//! fp8 e4m3 KV cache (`--kv-dtype fp8_e4m3`): the Metal conversions are bit-exact counterparts of the cpu oracle's
//! (`fp8_e4m3_to_f32` / `f32_to_fp8_e4m3`), checked over every code / every bf16 value.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, run_flat_shader};
use intarsia_core::dtype::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};

#[test]
fn decode_matches_the_oracle_for_all_256_codes() {
    let be = backend();
    let input: Vec<u8> = (0..=255u8).collect();
    let out = run_flat_shader(&be, "fp8_test_decode", &input, 256 * 4, 256).unwrap();
    for (code, c) in out.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        if code & 0x7f == 0x7f {
            continue; // the NaN codes: never written (the encoder saturates), the oracle says NaN, the kernel 480
        }
        let want = fp8_e4m3_to_f32(code as u8);
        assert_eq!(got.to_bits(), want.to_bits(), "code {code:#04x}: got {got} want {want}");
    }
}

#[test]
fn encode_matches_the_oracle_for_every_bf16_value() {
    let be = backend();
    // all 65536 bf16 patterns widened to f32 (NaNs included), then rounding-boundary probes in f32: every midpoint between
    // adjacent e4m3 values, one f32 ulp either side of it, the subnormal / normal boundary, saturation and its neighbours
    let mut vals: Vec<f32> = (0..=65535u32).map(|b| f32::from_bits(b << 16)).collect();
    for code in 0..126u8 {
        let (lo, hi) = (fp8_e4m3_to_f32(code), fp8_e4m3_to_f32(code + 1));
        let mid = (lo as f64 + hi as f64) / 2.0;
        for m in [mid as f32, f32::from_bits((mid as f32).to_bits() - 1), f32::from_bits((mid as f32).to_bits() + 1)] {
            vals.push(m);
            vals.push(-m);
        }
    }
    vals.extend([448.0, 447.99, 448.01, 464.0, 480.0, 1e9, -1e9, f32::INFINITY, f32::NEG_INFINITY, 1e-9, -1e-9, 0.0, -0.0]);
    // raw f32 patterns (xorshift), as a bf16 round never produces them
    let mut x = 0x9E3779B97F4A7C15u64;
    for _ in 0..200_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        vals.push(f32::from_bits((x >> 16) as u32));
    }
    let input: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_flat_shader(&be, "fp8_test_encode", &input, vals.len(), vals.len()).unwrap();
    let mut bad = 0;
    for (v, g) in vals.iter().zip(&out) {
        let want = f32_to_fp8_e4m3(*v);
        if *g != want {
            bad += 1;
            if bad < 10 {
                eprintln!("encode {v:e} ({:#010x}): got {g:#04x} want {want:#04x}", v.to_bits());
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {} values encode differently from the oracle", vals.len());
    eprintln!("fp8 encode: {} values (all 65536 bf16 patterns + boundaries + random f32) identical to the oracle", vals.len());
    let _ = bf16_to_f32(0);
    let _ = rand_bf16(1, 1);
}

// ---------------------------------------------------------------------------------------------------------------------
// kernels on fp8 caches
// ---------------------------------------------------------------------------------------------------------------------

use intarsia_backend_metal::bench::f32_to_bf16;
use intarsia_backend_metal::kernels::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

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

/// Random fp8 cache bytes: values of widely varying magnitude (subnormals, large codes), encoded by the oracle.
fn rand_fp8(n: usize, seed: u64) -> Vec<u8> {
    rand_bf16(n, seed)
        .into_iter()
        .enumerate()
        .map(|(i, b)| f32_to_fp8_e4m3(bf16_to_f32(b) * 2f32.powi((i % 13) as i32 - 9) * 4.0))
        .collect()
}

/// The bf16 cache holding the decoded value of every fp8 byte.
fn decode_bf16(fp8: &[u8]) -> Vec<u8> {
    fp8.iter().flat_map(|&b| f32_to_bf16(fp8_e4m3_to_f32(b)).to_le_bytes()).collect()
}

fn scratch_for(be: &MetalBackend, kernel: &str, cl: &OpCall) -> Option<TensorRef> {
    let need = be.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(cl)));
    (need > 0).then(|| {
        let b = be.alloc(need, MemKind::Device).unwrap();
        be.upload(&b, 0, &vec![0xEEu8; need], 0).unwrap();
        TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] }
    })
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
}

impl Kv {
    fn new(n: usize, nkv: usize, d: usize, cap: usize, seed: u64) -> Kv {
        let be = backend();
        let (k8, v8) = (rand_fp8(nkv * cap * d, seed), rand_fp8(nkv * cap * d, seed + 1));
        let (k8b, v8b) = (be.alloc(k8.len(), MemKind::Device).unwrap(), be.alloc(v8.len(), MemKind::Device).unwrap());
        be.upload(&k8b, 0, &k8, 0).unwrap();
        be.upload(&v8b, 0, &v8, 0).unwrap();
        let (kb, vb) = (be.alloc(2 * k8.len(), MemKind::Device).unwrap(), be.alloc(2 * v8.len(), MemKind::Device).unwrap());
        be.upload(&kb, 0, &decode_bf16(&k8), 0).unwrap();
        be.upload(&vb, 0, &decode_bf16(&v8), 0).unwrap();
        Kv { be, n, nkv, d, cap, k8: k8b, v8: v8b, kb, vb }
    }

    fn attrs(&self, pos0: usize) -> OpAttrs {
        OpAttrs {
            pos0,
            scale: 1.0 / (self.d as f32).sqrt(),
            n_heads: self.n,
            n_kv_heads: self.nkv,
            head_dim: self.d,
            kv_cap: self.cap,
            ..Default::default()
        }
    }

    fn cache(&self, f8: bool) -> Vec<TensorRef> {
        let (dt, k, v) = if f8 { (DType::Fp8E4m3, &self.k8, &self.v8) } else { (DType::Bf16, &self.kb, &self.vb) };
        vec![tr(k, 0, dt, &[self.nkv, self.cap, self.d]), tr(v, 0, dt, &[self.nkv, self.cap, self.d])]
    }

    /// One attention launch (`op` over `rows` query rows at pos0) on the fp8 or the decoded bf16 cache; output bf16 bits.
    fn attn(&self, kernel: &str, op: Op, q: &Buffer, rows: usize, pos0: usize, f8: bool) -> Vec<u8> {
        let len = rows * self.n * self.d;
        let out = zeros(&self.be, len * 2);
        let mut inputs = vec![tr(q, 0, DType::Bf16, &[rows, self.n, self.d])];
        inputs.extend(self.cache(f8));
        let mut cl = OpCall::new(op, inputs, vec![tr(&out, 0, DType::Bf16, &[rows, self.n, self.d])], self.attrs(pos0));
        cl.scratch = scratch_for(&self.be, kernel, &cl);
        self.be.launch(kernel, &cl, 0).unwrap();
        dl(&self.be, &out, 0, len * 2)
    }
}

fn bf16_vals(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}

/// f64 attention over the decoded cache for one query row at `pos` (causal: keys 0..=pos), all heads.
fn ref_row(kv: &Kv, qv: &[f32], pos: usize, k8: &[u8], v8: &[u8]) -> Vec<f64> {
    let (n, nkv, d, cap) = (kv.n, kv.nkv, kv.d, kv.cap);
    let g = n / nkv;
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0; n * d];
    for h in 0..n {
        let kvh = h / g;
        let mut s: Vec<f64> = (0..=pos)
            .map(|j| (0..d).map(|i| qv[h * d + i] as f64 * fp8_e4m3_to_f32(k8[(kvh * cap + j) * d + i]) as f64).sum::<f64>() * scale)
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0.0;
        for x in s.iter_mut() {
            *x = (*x - m).exp();
            l += *x;
        }
        for i in 0..d {
            out[h * d + i] =
                (0..=pos).map(|j| s[j] * fp8_e4m3_to_f32(v8[(kvh * cap + j) * d + i]) as f64).sum::<f64>() / l;
        }
    }
    out
}

fn fetch(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<u8> {
    dl(be, b, 0, n)
}

#[test]
fn decode_f8_matches_f64_reference_and_the_bf16_kernel_on_the_decoded_cache() {
    // split-KV decode (1, 2, 8 splits, partial last tile, group 4 and group 1) over a random fp8 cache
    for (n, nkv, d, cap) in [(32usize, 8usize, 128usize, 4096usize), (8, 8, 64, 2304), (16, 2, 256, 1536)] {
        let kv = Kv::new(n, nkv, d, cap, 31 + d as u64);
        let q = upload(&kv.be, &rand_bf16(n * d, 5));
        let k8 = fetch(&kv.be, &kv.k8, nkv * cap * d);
        let v8 = fetch(&kv.be, &kv.v8, nkv * cap * d);
        let qv = bf16_vals(&fetch(&kv.be, &q, n * d * 2));
        let (mut worst, mut diff_bf16, mut tot) = (0f64, 0usize, 0usize);
        for pos in [0usize, 1, 31, 127, 128, 129, 700, 1023, 1024, 1500, cap - 1] {
            let got = bf16_vals(&kv.attn(ATTN_DECODE_F8_SPLITKV, Op::AttnDecode, &q, 1, pos, true));
            let bfk = kv.attn(ATTN_DECODE_BF16_SPLITKV, Op::AttnDecode, &q, 1, pos, false);
            let want = ref_row(&kv, &qv, pos, &k8, &v8);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let tol = w.abs() * 2f64.powi(-8) * 1.5 + 1e-3;
                assert!((*g as f64 - w).abs() <= tol, "{n}/{nkv}/{d} pos {pos} el {i}: got {g} want {w}");
                worst = worst.max((*g as f64 - w).abs() / tol);
            }
            diff_bf16 += bf16_vals(&bfk).iter().zip(&got).filter(|(a, b)| a != b).count();
            tot += got.len();
        }
        eprintln!("decode f8 {n}/{nkv}/{d} cap {cap}: worst error {worst:.3} of tolerance; {diff_bf16}/{tot} outputs differ from the bf16 kernel on the decoded cache (summation order of the 4-unit step)");
    }
}

#[test]
fn rows_f8_are_bitwise_the_one_row_f8_launches() {
    // the g0b proof for the fp8 kernels: row r of a k-row call == the 1-row launch at position pos0 + r on the same cache
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
                let got = kv.attn(ATTN_PREFILL_F8_SPLITKV_ROWS, Op::AttnPrefill, &q, k, pos0, true);
                for r in 0..k {
                    let qr = upload(&kv.be, &qall[r * len..(r + 1) * len]);
                    let want = kv.attn(ATTN_DECODE_F8_SPLITKV, Op::AttnDecode, &qr, 1, pos0 + r, true);
                    assert!(got[r * len * 2..(r + 1) * len * 2] == want[..], "{n}/{nkv}/{d} k {k} pos0 {pos0} row {r}");
                }
                checked += 1;
            }
        }
        eprintln!("rows f8 {n}/{nkv}/{d} cap {cap}: {checked} (k, pos0) windows bitwise equal to the 1-row fp8 launches");
    }
}

#[test]
fn prefill_f8_is_bitwise_the_bf16_flash_kernels_on_the_decoded_cache() {
    // the staging decode is exact, so the fp8 prefill kernels give the bf16 flash kernels' bits on the decoded cache
    let (n, nkv, d, cap) = (32usize, 8usize, 128usize, 3200usize);
    let kv = Kv::new(n, nkv, d, cap, 777);
    let has_nax = kv.be.has_metal4();
    for &(t, pos0) in &[(2usize, 0usize), (3, 7), (16, 0), (47, 100), (64, 1), (130, 0), (200, 1100), (1, 5), (5, 3000), (512, 2048)] {
        if t + pos0 > cap || t < 2 {
            continue;
        }
        let q = upload(&kv.be, &rand_bf16(t * n * d, 9 + t as u64));
        let pairs: Vec<(&str, &str)> = if has_nax {
            vec![(ATTN_PREFILL_F8_FLASH_NAX, ATTN_PREFILL_BF16_FLASH_NAX), (ATTN_PREFILL_F8_FLASH, ATTN_PREFILL_BF16_FLASH)]
        } else {
            vec![(ATTN_PREFILL_F8_FLASH, ATTN_PREFILL_BF16_FLASH)]
        };
        for (f8k, bfk) in pairs {
            let a = kv.attn(f8k, Op::AttnPrefill, &q, t, pos0, true);
            let b = kv.attn(bfk, Op::AttnPrefill, &q, t, pos0, false);
            assert!(a == b, "{f8k} T {t} pos0 {pos0} differs from {bfk} on the decoded cache");
        }
    }
}

#[test]
fn kv_append_f8_is_the_oracle_encode_of_the_bf16_rows() {
    let be = backend();
    let (nkv, d, cap) = (8usize, 128usize, 40usize);
    for (t, pos0) in [(1usize, 0usize), (1, 17), (6, 0), (6, 9), (33, 3)] {
        let src = rand_bf16(t * nkv * d, 3 + t as u64);
        let sb = upload(&be, &src);
        let init: Vec<u8> = (0..nkv * cap * d).map(|i| (i * 7 + 3) as u8).collect();
        let cb = be.alloc(init.len(), MemKind::Device).unwrap();
        be.upload(&cb, 0, &init, 0).unwrap();
        let at = OpAttrs { pos0, kv_cap: cap, n_kv_heads: nkv, head_dim: d, ..Default::default() };
        let cl = OpCall::new(Op::KvAppend, vec![tr(&sb, 0, DType::Bf16, &[t, nkv, d])], vec![tr(&cb, 0, DType::Fp8E4m3, &[nkv, cap, d])], at);
        be.launch(KV_APPEND_F8, &cl, 0).unwrap();
        let got = dl(&be, &cb, 0, init.len());
        let mut want = init.clone();
        for ti in 0..t {
            for h in 0..nkv {
                for e in 0..d {
                    want[(h * cap + pos0 + ti) * d + e] = f32_to_fp8_e4m3(bf16_to_f32(src[(ti * nkv + h) * d + e]));
                }
            }
        }
        assert!(got == want, "T {t} pos0 {pos0}");
    }
}

#[test]
fn qk_norm_rope_kv_f8_is_the_encoded_bf16_cache_and_k_rows_equal_k_single_rows() {
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
                    vec![
                        tr(qo, r0 * n * d * 2, DType::Bf16, &[rows, n, d]),
                        tr(kc, 0, dt, &[nkv, cap, d]),
                        tr(vc, 0, dt, &[nkv, cap, d]),
                    ],
                    at(p),
                )
            };
            // reference: the bf16 kernel, then the oracle encode of its cache bytes
            let (qo0, kcb, vcb) = (zeros(&be, t * n * d * 2), zeros(&be, nkv * cap * d * 2), zeros(&be, nkv * cap * d * 2));
            be.launch(QK_NORM_ROPE_KV_BF16, &mk(t, 0, &qo0, &kcb, &vcb, pos0, DType::Bf16), 0).unwrap();
            let (kcr, vcr) = (dl(&be, &kcb, 0, nkv * cap * d * 2), dl(&be, &vcb, 0, nkv * cap * d * 2));
            let enc = |raw: &[u8]| -> Vec<u8> {
                raw.chunks_exact(2).map(|c| f32_to_fp8_e4m3(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))).collect()
            };
            // one k-row call, and k 1-row calls, both into fp8 caches (a sentinel in the unwritten rows)
            let fill = |b: &Buffer| be.upload(b, 0, &vec![0xA5u8; nkv * cap * d], 0).unwrap();
            let (qo1, kc1, vc1) = (zeros(&be, t * n * d * 2), zeros(&be, nkv * cap * d), zeros(&be, nkv * cap * d));
            let (qo2, kc2, vc2) = (zeros(&be, t * n * d * 2), zeros(&be, nkv * cap * d), zeros(&be, nkv * cap * d));
            for b in [&kc1, &vc1, &kc2, &vc2] {
                fill(b);
            }
            be.launch(QK_NORM_ROPE_KV_F8, &mk(t, 0, &qo1, &kc1, &vc1, pos0, DType::Fp8E4m3), 0).unwrap();
            for r in 0..t {
                be.launch(QK_NORM_ROPE_KV_F8, &mk(1, r, &qo2, &kc2, &vc2, pos0 + r, DType::Fp8E4m3), 0).unwrap();
            }
            let (k1, v1, k2, v2) = (
                dl(&be, &kc1, 0, nkv * cap * d),
                dl(&be, &vc1, 0, nkv * cap * d),
                dl(&be, &kc2, 0, nkv * cap * d),
                dl(&be, &vc2, 0, nkv * cap * d),
            );
            let w = format!("{n}/{nkv}/{d}/{rd} T {t} pos0 {pos0}");
            assert!(k1 == k2 && v1 == v2, "{w}: a {t}-row call writes other bytes than {t} 1-row calls");
            assert!(dl(&be, &qo1, 0, t * n * d * 2) == dl(&be, &qo0, 0, t * n * d * 2), "{w}: q output differs from the bf16 kernel");
            let (ke, ve) = (enc(&kcr), enc(&vcr));
            for h in 0..nkv {
                for row in 0..cap {
                    let range = (h * cap + row) * d..(h * cap + row + 1) * d;
                    if row >= pos0 && row < pos0 + t {
                        assert!(k1[range.clone()] == ke[range.clone()], "{w}: k row {row} head {h} is not the encoded bf16 row");
                        assert!(v1[range.clone()] == ve[range.clone()], "{w}: v row {row} head {h} is not the encoded bf16 row");
                    } else {
                        assert!(k1[range.clone()].iter().all(|b| *b == 0xA5) && v1[range].iter().all(|b| *b == 0xA5), "{w}: row {row} outside the append was written");
                    }
                }
            }
        }
    }
}

#[test]
fn f8_kernels_are_registered_only_for_fp8_caches_and_leave_the_bf16_ones_alone() {
    let be = backend();
    let ks = be.kernels();
    let s = |op: Op, m: usize, kv: DType| ShapeInfo {
        m,
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        act: Some(DType::Bf16),
        weight: Some(kv),
        out: Some(if op == Op::KvAppend { kv } else { DType::Bf16 }),
        ..Default::default()
    };
    for (name, op, m) in [
        (KV_APPEND_F8, Op::KvAppend, 1usize),
        (QK_NORM_ROPE_KV_F8, Op::QkNormRopeKv, 1),
        (ATTN_DECODE_F8_SPLITKV, Op::AttnDecode, 1),
        (ATTN_PREFILL_F8_FLASH, Op::AttnPrefill, 64),
        (ATTN_PREFILL_F8_SPLITKV_ROWS, Op::AttnPrefill, 4),
    ] {
        let k = ks.iter().find(|k| k.name == name).unwrap();
        assert!(k.accepts(&s(op, m, DType::Fp8E4m3)), "{name} takes an fp8 cache");
        assert!(!k.accepts(&s(op, m, DType::Bf16)), "{name} must not take a bf16 cache");
    }
    for k in ks.iter().filter(|k| !k.name.contains("_f8")) {
        for op in [Op::KvAppend, Op::QkNormRopeKv, Op::AttnDecode, Op::AttnPrefill] {
            if k.op == op {
                assert!(!k.accepts(&s(op, 4, DType::Fp8E4m3)) || k.name.contains("f32"), "{} takes an fp8 cache", k.name);
            }
        }
    }
}
