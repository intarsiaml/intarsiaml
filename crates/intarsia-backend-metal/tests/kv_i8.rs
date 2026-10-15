//! int8 block-scaled KV cache (`--kv-dtype int8`): layout `[Nkv, cap, D]` int8 followed by `[Nkv, cap, D / 32]` bf16 scales per
//! plane (core `state::kv_plane_bytes`). Host references: the quantiser, an f64 attention over the dequantised cache, the
//! bf16 flash kernels on the bf16 dequantisation, and the g0b-style rows == 1-row identity.

#![cfg(target_os = "macos")]

mod common;

use common::{backend as plain_backend, *};
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
use intarsia_backend_metal::kernels::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::state::kv_plane_bytes;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const B: usize = 32;

/// The backend with the int8 KV kernels listed (they are hidden unless an int8 cache is requested, see `kernels::int8_kv_enabled`).
fn backend() -> MetalBackend {
    std::env::set_var("INTARSIA_METAL_KV_INT8", "1");
    plain_backend()
}

fn tr(b: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}
fn zeros(be: &MetalBackend, n: usize) -> Buffer {
    let b = be.alloc(n, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0u8; n], 0).unwrap();
    b
}
fn dl(be: &MetalBackend, b: &Buffer, off: usize, n: usize) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; n];
    be.download(b, off, &mut raw, 0).unwrap();
    raw
}
fn bf16_vals(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}

/// The quantiser of kernels/metal/common.h on one 32-element block of bf16-valued floats: (q, bf16 scale bits).
fn quant_block(x: &[f32]) -> (Vec<i8>, u16) {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let sbits = f32_to_bf16(amax / 127.0);
    let s = bf16_to_f32(sbits);
    let q = x.iter().map(|v| if s == 0.0 { 0 } else { (v / s).round_ties_even().clamp(-127.0, 127.0) as i8 }).collect();
    (q, sbits)
}

/// Quantise a `[rows, D]` bf16 matrix: (data bytes, scale bits per block).
fn quant_rows(vals: &[f32], d: usize) -> (Vec<u8>, Vec<u16>) {
    let (mut data, mut sc) = (Vec::new(), Vec::new());
    for blk in vals.chunks(B) {
        let (q, s) = quant_block(blk);
        data.extend(q.iter().map(|v| *v as u8));
        sc.push(s);
    }
    let _ = d;
    (data, sc)
}

/// One plane's bytes: int8 data then the scales.
fn plane(data: &[u8], sc: &[u16]) -> Vec<u8> {
    let mut v = data.to_vec();
    v.extend(sc.iter().flat_map(|s| s.to_le_bytes()));
    v
}

fn dequant_plane_bf16(p: &[u8], nkv: usize, cap: usize, d: usize) -> Vec<u8> {
    let n = nkv * cap * d;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let blk = i / B;
        let s = bf16_to_f32(u16::from_le_bytes([p[n + 2 * blk], p[n + 2 * blk + 1]]));
        out.extend(f32_to_bf16(p[i] as i8 as f32 * s).to_le_bytes());
    }
    out
}

/// A random int8 cache plane: data uniform in [-127, 127], scales spanning several octaves.
fn rand_plane(nkv: usize, cap: usize, d: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    let mut nxt = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let n = nkv * cap * d;
    let data: Vec<u8> = (0..n).map(|_| ((nxt() % 255) as i32 - 127) as i8 as u8).collect();
    let sc: Vec<u16> = (0..n / B).map(|_| f32_to_bf16(0.002 * 2f32.powi((nxt() % 5) as i32))).collect();
    plane(&data, &sc)
}

#[test]
fn plane_bytes_helper_is_exact_and_zero_extra_for_other_dtypes() {
    assert_eq!(kv_plane_bytes(8, 100, 128, DType::Bf16), 8 * 100 * 128 * 2);
    assert_eq!(kv_plane_bytes(8, 100, 128, DType::Fp8E4m3), 8 * 100 * 128);
    assert_eq!(kv_plane_bytes(8, 100, 128, DType::I8), 8 * 100 * 128 + 2 * 8 * 100 * 4);
}

#[test]
fn kv_append_i8_is_the_host_quantiser() {
    let be = backend();
    let (nkv, d, cap) = (8usize, 128usize, 40usize);
    for (t, pos0) in [(1usize, 0usize), (1, 17), (6, 0), (6, 9), (33, 3)] {
        let src = rand_bf16(t * nkv * d, 3 + t as u64);
        let sb = upload(&be, &src);
        let plane_bytes = kv_plane_bytes(nkv, cap, d, DType::I8);
        let init: Vec<u8> = (0..plane_bytes).map(|i| (i * 7 + 3) as u8).collect();
        let cb = be.alloc(plane_bytes, MemKind::Device).unwrap();
        be.upload(&cb, 0, &init, 0).unwrap();
        let at = OpAttrs { pos0, kv_cap: cap, n_kv_heads: nkv, head_dim: d, ..Default::default() };
        let cl = OpCall::new(Op::KvAppend, vec![tr(&sb, 0, DType::Bf16, &[t, nkv, d])], vec![tr(&cb, 0, DType::I8, &[nkv, cap, d])], at);
        be.launch(KV_APPEND_I8, &cl, 0).unwrap();
        let got = dl(&be, &cb, 0, plane_bytes);
        let mut want = init.clone();
        let vals: Vec<f32> = src.iter().map(|b| bf16_to_f32(*b)).collect();
        let mut bad = 0;
        for ti in 0..t {
            for h in 0..nkv {
                let row = &vals[(ti * nkv + h) * d..][..d];
                let (data, sc) = quant_rows(row, d);
                let r = h * cap + pos0 + ti;
                want[r * d..(r + 1) * d].copy_from_slice(&data);
                for (bi, s) in sc.iter().enumerate() {
                    let o = nkv * cap * d + 2 * (r * (d / B) + bi);
                    want[o..o + 2].copy_from_slice(&s.to_le_bytes());
                }
            }
        }
        for i in 0..plane_bytes {
            if got[i] != want[i] {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "T {t} pos0 {pos0}: {bad} of {plane_bytes} bytes differ from the host quantiser");
    }
}

#[test]
fn qk_norm_rope_kv_i8_quantises_the_bf16_cache_and_forms_agree() {
    let be = backend();
    for (n, nkv, d, rd) in [(32usize, 8usize, 128usize, 128usize), (8, 2, 64, 64), (8, 2, 128, 32)] {
        let cap = 300usize;
        for (t, pos0) in [(1usize, 0usize), (1, 41), (5, 0), (5, 77), (33, 100), (8, 250)] {
            let q = upload(&be, &rand_bf16(t * n * d, 11));
            let kk = upload(&be, &rand_bf16(t * nkv * d, 12));
            let vv = upload(&be, &rand_bf16(t * nkv * d, 13));
            let (wq, wk) = (upload(&be, &rand_bf16(d, 14)), upload(&be, &rand_bf16(d, 15)));
            let at = |p: usize| OpAttrs {
                pos0: p,
                theta: 1_000_000.0,
                eps: 1e-6,
                w_offset: 0.0,
                head_dim: d,
                rotary_dim: rd,
                n_heads: n,
                n_kv_heads: nkv,
                kv_cap: cap,
                ..Default::default()
            };
            let mk = |rows: usize, r0: usize, qo: &Buffer, kc: &Buffer, vc: &Buffer, p: usize, dt: DType| {
                OpCall::new(
                    Op::QkNormRopeKv,
                    vec![
                        tr(&q, r0 * n * d * 2, DType::Bf16, &[rows, n, d]),
                        tr(&kk, r0 * nkv * d * 2, DType::Bf16, &[rows, nkv, d]),
                        tr(&vv, r0 * nkv * d * 2, DType::Bf16, &[rows, nkv, d]),
                        tr(&wq, 0, DType::Bf16, &[d]),
                        tr(&wk, 0, DType::Bf16, &[d]),
                    ],
                    vec![tr(qo, r0 * n * d * 2, DType::Bf16, &[rows, n, d]), tr(kc, 0, dt, &[nkv, cap, d]), tr(vc, 0, dt, &[nkv, cap, d])],
                    at(p),
                )
            };
            let w = format!("{n}/{nkv}/{d}/{rd} T {t} pos0 {pos0}");
            // reference: the bf16 kernel, then the host quantiser on the rows it wrote
            let (qo0, kcb, vcb) = (zeros(&be, t * n * d * 2), zeros(&be, nkv * cap * d * 2), zeros(&be, nkv * cap * d * 2));
            be.launch(QK_NORM_ROPE_KV_BF16, &mk(t, 0, &qo0, &kcb, &vcb, pos0, DType::Bf16), 0).unwrap();
            let (kr, vr) = (bf16_vals(&dl(&be, &kcb, 0, nkv * cap * d * 2)), bf16_vals(&dl(&be, &vcb, 0, nkv * cap * d * 2)));
            let pb = kv_plane_bytes(nkv, cap, d, DType::I8);
            // one t-row call and t 1-row calls into int8 planes
            let (qo1, kc1, vc1) = (zeros(&be, t * n * d * 2), zeros(&be, pb), zeros(&be, pb));
            let (qo2, kc2, vc2) = (zeros(&be, t * n * d * 2), zeros(&be, pb), zeros(&be, pb));
            be.launch(QK_NORM_ROPE_KV_I8, &mk(t, 0, &qo1, &kc1, &vc1, pos0, DType::I8), 0).unwrap();
            for r in 0..t {
                be.launch(QK_NORM_ROPE_KV_I8, &mk(1, r, &qo2, &kc2, &vc2, pos0 + r, DType::I8), 0).unwrap();
            }
            let (k1, v1, k2, v2) = (dl(&be, &kc1, 0, pb), dl(&be, &vc1, 0, pb), dl(&be, &kc2, 0, pb), dl(&be, &vc2, 0, pb));
            assert!(k1 == k2 && v1 == v2, "{w}: a {t}-row call (prefill form for D=128) writes other bytes than {t} 1-row calls");
            assert!(dl(&be, &qo1, 0, t * n * d * 2) == dl(&be, &qo0, 0, t * n * d * 2), "{w}: q output differs from the bf16 kernel");
            for (name, got, refv) in [("k", &k1, &kr), ("v", &v1, &vr)] {
                let mut bad = 0;
                for h in 0..nkv {
                    for row in pos0..pos0 + t {
                        let r = h * cap + row;
                        let (data, sc) = quant_rows(&refv[r * d..(r + 1) * d], d);
                        if got[r * d..(r + 1) * d] != data[..] {
                            bad += 1;
                        }
                        for (bi, s) in sc.iter().enumerate() {
                            let o = nkv * cap * d + 2 * (r * (d / B) + bi);
                            if got[o..o + 2] != s.to_le_bytes() {
                                bad += 1;
                            }
                        }
                    }
                }
                assert_eq!(bad, 0, "{w}: {name} cache has {bad} head rows / scales that differ from the host quantiser of the bf16 kernel's rows");
            }
        }
    }
}

struct Kv {
    be: MetalBackend,
    n: usize,
    nkv: usize,
    d: usize,
    cap: usize,
    k8: Buffer,
    v8: Buffer,
    kb: Buffer,
    vb: Buffer,
    kp: Vec<u8>,
    vp: Vec<u8>,
}

impl Kv {
    fn new(n: usize, nkv: usize, d: usize, cap: usize, seed: u64) -> Kv {
        let be = backend();
        let (kp, vp) = (rand_plane(nkv, cap, d, seed), rand_plane(nkv, cap, d, seed + 1));
        let (k8, v8) = (be.alloc(kp.len(), MemKind::Device).unwrap(), be.alloc(vp.len(), MemKind::Device).unwrap());
        be.upload(&k8, 0, &kp, 0).unwrap();
        be.upload(&v8, 0, &vp, 0).unwrap();
        let (kdq, vdq) = (dequant_plane_bf16(&kp, nkv, cap, d), dequant_plane_bf16(&vp, nkv, cap, d));
        let (kb, vb) = (be.alloc(kdq.len(), MemKind::Device).unwrap(), be.alloc(vdq.len(), MemKind::Device).unwrap());
        be.upload(&kb, 0, &kdq, 0).unwrap();
        be.upload(&vb, 0, &vdq, 0).unwrap();
        Kv { be, n, nkv, d, cap, k8, v8, kb, vb, kp, vp }
    }
    fn attrs(&self, pos0: usize) -> OpAttrs {
        OpAttrs { pos0, scale: 1.0 / (self.d as f32).sqrt(), n_heads: self.n, n_kv_heads: self.nkv, head_dim: self.d, kv_cap: self.cap, ..Default::default() }
    }
    fn attn(&self, kernel: &str, op: Op, q: &Buffer, rows: usize, pos0: usize, i8: bool) -> Vec<u8> {
        let len = rows * self.n * self.d;
        let out = zeros(&self.be, len * 2);
        let (dt, k, v) = if i8 { (DType::I8, &self.k8, &self.v8) } else { (DType::Bf16, &self.kb, &self.vb) };
        let inputs = vec![tr(q, 0, DType::Bf16, &[rows, self.n, self.d]), tr(k, 0, dt, &[self.nkv, self.cap, self.d]), tr(v, 0, dt, &[self.nkv, self.cap, self.d])];
        let mut cl = OpCall::new(op, inputs, vec![tr(&out, 0, DType::Bf16, &[rows, self.n, self.d])], self.attrs(pos0));
        let need = self.be.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(&cl)));
        if need > 0 {
            let b = self.be.alloc(need, MemKind::Device).unwrap();
            self.be.upload(&b, 0, &vec![0xEEu8; need], 0).unwrap();
            cl.scratch = Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] });
        }
        self.be.launch(kernel, &cl, 0).unwrap();
        dl(&self.be, &out, 0, len * 2)
    }
    fn deq(&self, plane: &[u8], h: usize, j: usize, i: usize) -> f64 {
        let (nkv, cap, d) = (self.nkv, self.cap, self.d);
        let r = h * cap + j;
        let s = bf16_to_f32(u16::from_le_bytes([plane[nkv * cap * d + 2 * (r * (d / B) + i / B)], plane[nkv * cap * d + 2 * (r * (d / B) + i / B) + 1]]));
        (plane[r * d + i] as i8 as f32 * s) as f64
    }
}

#[test]
fn decode_i8_matches_f64_reference() {
    for (n, nkv, d, cap) in [(32usize, 8usize, 128usize, 4096usize), (8, 8, 64, 2304), (16, 2, 256, 1536)] {
        let kv = Kv::new(n, nkv, d, cap, 31 + d as u64);
        let qv = rand_bf16(n * d, 5);
        let q = upload(&kv.be, &qv);
        let qf: Vec<f32> = qv.iter().map(|b| bf16_to_f32(*b)).collect();
        let mut worst = 0f64;
        for pos in [0usize, 1, 31, 127, 128, 129, 700, 1023, 1024, 1500, cap - 1] {
            let got = bf16_vals(&kv.attn(ATTN_DECODE_I8_SPLITKV, Op::AttnDecode, &q, 1, pos, true));
            let g = n / nkv;
            let scale = 1.0 / (d as f64).sqrt();
            for h in 0..n {
                let kvh = h / g;
                let mut s: Vec<f64> = (0..=pos).map(|j| (0..d).map(|i| qf[h * d + i] as f64 * kv.deq(&kv.kp, kvh, j, i)).sum::<f64>() * scale).collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut l = 0.0;
                for x in s.iter_mut() {
                    *x = (*x - m).exp();
                    l += *x;
                }
                for i in 0..d {
                    let w = (0..=pos).map(|j| s[j] * kv.deq(&kv.vp, kvh, j, i)).sum::<f64>() / l;
                    let tol = w.abs() * 2f64.powi(-8) * 1.5 + 2e-3;
                    let e = (got[h * d + i] as f64 - w).abs();
                    assert!(e <= tol, "{n}/{nkv}/{d} pos {pos} head {h} el {i}: got {} want {w}", got[h * d + i]);
                    worst = worst.max(e / tol);
                }
            }
        }
        eprintln!("decode i8 {n}/{nkv}/{d} cap {cap}: worst error {worst:.3} of tolerance");
    }
}

#[test]
fn rows_i8_are_bitwise_the_one_row_i8_launches() {
    for (n, nkv, d, cap) in [(32usize, 8usize, 128usize, 4096usize), (8, 2, 64, 2304), (16, 2, 256, 1536), (12, 4, 32, 1024)] {
        let kv = Kv::new(n, nkv, d, cap, 4242 + d as u64);
        let qall = rand_bf16(8 * n * d, 99);
        let q = upload(&kv.be, &qall);
        let len = n * d;
        let mut checked = 0;
        let mut positions: Vec<usize> = vec![0, 1, 2, 7, 31, 32, 33, 126, 127, 128, 129, 480, 481, 1023, 1024, 1025, cap / 3, cap / 2];
        positions.extend([1792, 1793, 960, 1440, 1920].into_iter().filter(|p| *p < cap));
        for k in [2usize, 3, 5, 8] {
            for &pos0 in positions.iter().filter(|p| **p + k <= cap) {
                let got = kv.attn(ATTN_PREFILL_I8_SPLITKV_ROWS, Op::AttnPrefill, &q, k, pos0, true);
                for r in 0..k {
                    let qr = upload(&kv.be, &qall[r * len..(r + 1) * len]);
                    let want = kv.attn(ATTN_DECODE_I8_SPLITKV, Op::AttnDecode, &qr, 1, pos0 + r, true);
                    assert!(got[r * len * 2..(r + 1) * len * 2] == want[..], "{n}/{nkv}/{d} k {k} pos0 {pos0} row {r}");
                }
                checked += 1;
            }
        }
        eprintln!("rows i8 {n}/{nkv}/{d} cap {cap}: {checked} (k, pos0) windows bitwise equal to the 1-row int8 launches");
    }
}

#[test]
fn prefill_i8_is_bitwise_the_bf16_flash_kernels_on_the_staged_cache() {
    let (n, nkv, d, cap) = (32usize, 8usize, 128usize, 3200usize);
    let kv = Kv::new(n, nkv, d, cap, 777);
    let has_nax = kv.be.has_metal4();
    for &(t, pos0) in &[(2usize, 0usize), (3, 7), (16, 0), (47, 100), (64, 1), (130, 0), (200, 1100), (5, 3000), (512, 2048)] {
        if t + pos0 > cap {
            continue;
        }
        let q = upload(&kv.be, &rand_bf16(t * n * d, 9 + t as u64));
        let mut pairs = vec![(ATTN_PREFILL_I8_FLASH, ATTN_PREFILL_BF16_FLASH)];
        if has_nax {
            pairs.push((ATTN_PREFILL_I8_FLASH_NAX, ATTN_PREFILL_BF16_FLASH_NAX));
        }
        for (i8k, bfk) in pairs {
            let a = kv.attn(i8k, Op::AttnPrefill, &q, t, pos0, true);
            let b = kv.attn(bfk, Op::AttnPrefill, &q, t, pos0, false);
            assert!(a == b, "{i8k} T {t} pos0 {pos0} differs from {bfk} on the bf16 dequantisation of the cache");
        }
    }
}

#[test]
fn i8_kernels_are_registered_only_for_int8_caches() {
    // listed once an int8 cache is requested (hidden by default, `kernels::int8_kv_enabled`; process-wide env, so the
    // "hidden by default" half is checked by `kernels test` in the gates, not here)
    let be = backend();
    let ks = be.kernels();
    let s = |m: usize, kv: DType, out: DType| ShapeInfo {
        m,
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        act: Some(DType::Bf16),
        weight: Some(kv),
        out: Some(out),
        ..Default::default()
    };
    for (name, op, m) in [
        (QK_NORM_ROPE_KV_I8, Op::QkNormRopeKv, 1usize),
        (ATTN_DECODE_I8_SPLITKV, Op::AttnDecode, 1),
        (ATTN_PREFILL_I8_FLASH, Op::AttnPrefill, 64),
        (ATTN_PREFILL_I8_SPLITKV_ROWS, Op::AttnPrefill, 4),
    ] {
        let k = ks.iter().find(|k| k.name == name && k.op == op).unwrap();
        assert!(k.accepts(&s(m, DType::I8, DType::Bf16)), "{name} takes an int8 cache");
        assert!(!k.accepts(&s(m, DType::Bf16, DType::Bf16)) && !k.accepts(&s(m, DType::Fp8E4m3, DType::Bf16)), "{name} must take only int8");
    }
    let k = ks.iter().find(|k| k.name == KV_APPEND_I8).unwrap();
    assert!(k.accepts(&s(1, DType::Bf16, DType::I8)) && !k.accepts(&s(1, DType::Bf16, DType::Bf16)));
    for k in ks.iter().filter(|k| !k.name.contains("_i8")) {
        if matches!(k.op, Op::KvAppend | Op::QkNormRopeKv | Op::AttnDecode | Op::AttnPrefill) {
            let out = if k.op == Op::KvAppend { DType::I8 } else { DType::Bf16 };
            assert!(!k.accepts(&s(4, DType::I8, out)) || k.name.contains("f32"), "{} takes an int8 cache", k.name);
        }
    }
}
