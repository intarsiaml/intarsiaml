#![cfg(target_os = "macos")]
//! Rows-exact gate of the Inkling-2-Mini attention kernels (`ink_kv_prologue_bf16`, `rel_sink_attn_bf16`, inkattn.rs): an
//! R-row InkKvPrologue + RelSinkAttn pair equals, bit for bit, R one-row pairs at positions pos0 .. pos0 + R made in order
//! (outputs q / rel / k / v / o and the final conv rings and K / V caches), at the real shapes (24 / 8 heads, head_dim 128,
//! d_rel 16, extent 1024, K 4, conv ring 19 rows): sliding layers (window 1024, ring of 1040 rows, positions before, at and
//! across the ring wrap) and full layers (linear cache, dynamic taps, tau, positions across the rel extent). The one-row
//! calls run once with host positions and once with a device position (`OpCall::pos`): both must match the R-row call.
//! Numerics against the f64 reference are `intarsia kernels test --backend metal --op ink_kv_prologue|rel_sink_attn`.

mod common;
use common::*;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::inkling2::ops::nq;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const H: usize = 24;
const HK: usize = 8;
const DH: usize = 128;
const DR: usize = 16;
const E: usize = 1024;
const K: usize = 4;
const RR: usize = K - 1 + 16;

fn tr(b: &Buffer, off: usize, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: DType::Bf16, shape: shape.to_vec() }
}

fn read(be: &impl Backend, b: &Buffer, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn zeros(be: &impl Backend, n: usize) -> Buffer {
    let b = be.alloc(n * 2, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0u8; n * 2], 0).unwrap();
    b
}

fn scaled(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<u16> {
    use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
    rand_bf16(n, seed).iter().map(|&v| f32_to_bf16(lo + (hi - lo) * (bf16_to_f32(v) * 0.5 + 0.5))).collect()
}

struct Layer {
    window: usize,
    ring: bool,
    cr: usize,
    dynamic: bool,
}

/// Everything one layer visit touches; `state` = (k ring, v ring, k cache, v cache).
struct Run {
    outs: Vec<Vec<u16>>,
    state: Vec<Vec<u16>>,
}

struct Inputs {
    pq: Vec<u16>,
    pk: Vec<u16>,
    pv: Vec<u16>,
    w: Vec<Vec<u16>>, // k taps, v taps, q scale, r scale, rel proj
    state0: Vec<Vec<u16>>,
}

fn make(l: &Layer, t: usize, seed: u64) -> Inputs {
    let ckv = HK * DH;
    let pw = if l.dynamic { ckv + HK * K } else { ckv };
    Inputs {
        pq: rand_bf16(t * nq(H, DH, DR), seed),
        pk: rand_bf16(t * pw, seed + 1),
        pv: rand_bf16(t * pw, seed + 2),
        w: vec![
            rand_bf16(K * ckv, seed + 3),
            rand_bf16(K * ckv, seed + 4),
            scaled(H * DH, seed + 5, 0.5, 6.0),
            scaled(H * DR, seed + 6, 0.5, 3.0),
            rand_bf16(DR * E, seed + 7),
        ],
        state0: vec![rand_bf16(RR * ckv, seed + 8), rand_bf16(RR * ckv, seed + 9), rand_bf16(HK * l.cr * DH, seed + 10), rand_bf16(HK * l.cr * DH, seed + 11)],
    }
}

fn attrs(l: &Layer, pos0: usize) -> OpAttrs {
    let mut a = OpAttrs { pos0, n_heads: H, n_kv_heads: HK, head_dim: DH, window: l.window, scale: (DH as f32).powf(-0.5), eps: 1e-6, ..OpAttrs::default() };
    a.ink.qk_clip = 6.0;
    a.ink.rel_clip = 1.5;
    a.ink.d_rel = DR;
    a.ink.kv_ring = l.ring;
    a.ink.dyn_conv = l.dynamic;
    if l.dynamic {
        (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
    }
    a
}

/// Run the visit as calls of `rows` rows each (rows divides T); `dev`: device positions.
fn run(be: &intarsia_backend_metal::MetalBackend, l: &Layer, inp: &Inputs, t: usize, pos0: usize, rows: usize, dev: bool, fused: bool) -> Run {
    let ckv = HK * DH;
    let pw = inp.pk.len() / t;
    let nqw = nq(H, DH, DR);
    let (pq, pk, pv) = (upload(be, &inp.pq), upload(be, &inp.pk), upload(be, &inp.pv));
    let w: Vec<Buffer> = inp.w.iter().map(|v| upload(be, v)).collect();
    let st: Vec<Buffer> = inp.state0.iter().map(|v| upload(be, v)).collect();
    let (q, rel, k, v, o) = (zeros(be, t * H * DH), zeros(be, t * H * E), zeros(be, t * ckv), zeros(be, t * ckv), zeros(be, t * H * DH));
    let posb = be.alloc(16, MemKind::Device).unwrap();
    for c in 0..t / rows {
        let r0 = c * rows;
        let p0 = pos0 + r0;
        let mut a = attrs(l, p0);
        let pos = if dev {
            be.upload(&posb, 0, &(p0 as i32).to_le_bytes(), 0).unwrap();
            a.pos0 = 0; // must not be read
            Some(TensorRef { buf: posb.clone(), offset: 0, dtype: DType::I32, shape: vec![1] })
        } else {
            None
        };
        if fused {
            let mut f = OpCall::new(
                Op::InkAttnDecode,
                vec![
                    tr(&pq, r0 * nqw * 2, &[rows, nqw]),
                    tr(&pk, r0 * pw * 2, &[rows, pw]),
                    tr(&pv, r0 * pw * 2, &[rows, pw]),
                    tr(&w[0], 0, &[K, ckv]),
                    tr(&w[1], 0, &[K, ckv]),
                    tr(&w[2], 0, &[H, DH]),
                    tr(&w[3], 0, &[H, DR]),
                    tr(&w[4], 0, &[DR, E]),
                    tr(&st[2], 0, &[HK, l.cr, DH]),
                    tr(&st[3], 0, &[HK, l.cr, DH]),
                ],
                vec![
                    tr(&o, r0 * H * DH * 2, &[rows, H * DH]),
                    tr(&st[2], 0, &[HK, l.cr, DH]),
                    tr(&st[3], 0, &[HK, l.cr, DH]),
                    tr(&st[0], 0, &[RR, ckv]),
                    tr(&st[1], 0, &[RR, ckv]),
                ],
                a,
            );
            f.pos = pos;
            let need = be.kernels().iter().find(|k| k.name == "ink_attn_decode_bf16").unwrap().scratch_bytes(&ShapeInfo::from_call(&f));
            let sb = be.alloc(need, MemKind::Device).unwrap();
            f.scratch = Some(TensorRef { buf: sb, offset: 0, dtype: DType::U8, shape: vec![need] });
            be.launch("ink_attn_decode_bf16", &f, 0).unwrap();
            be.synchronize(0).unwrap();
            continue;
        }
        let mut pro = OpCall::new(
            Op::InkKvPrologue,
            vec![
                tr(&pq, r0 * nqw * 2, &[rows, nqw]),
                tr(&pk, r0 * pw * 2, &[rows, pw]),
                tr(&pv, r0 * pw * 2, &[rows, pw]),
                tr(&w[0], 0, &[K, ckv]),
                tr(&w[1], 0, &[K, ckv]),
                tr(&w[2], 0, &[H, DH]),
                tr(&w[3], 0, &[H, DR]),
                tr(&w[4], 0, &[DR, E]),
            ],
            vec![
                tr(&q, r0 * H * DH * 2, &[rows, H, DH]),
                tr(&rel, r0 * H * E * 2, &[rows, H, E]),
                tr(&k, r0 * ckv * 2, &[rows, HK, DH]),
                tr(&v, r0 * ckv * 2, &[rows, HK, DH]),
                tr(&st[0], 0, &[RR, ckv]),
                tr(&st[1], 0, &[RR, ckv]),
            ],
            a.clone(),
        );
        pro.pos = pos.clone();
        be.launch("ink_kv_prologue_bf16", &pro, 0).unwrap();
        let mut att = OpCall::new(
            Op::RelSinkAttn,
            vec![
                tr(&q, r0 * H * DH * 2, &[rows, H, DH]),
                tr(&k, r0 * ckv * 2, &[rows, HK, DH]),
                tr(&v, r0 * ckv * 2, &[rows, HK, DH]),
                tr(&st[2], 0, &[HK, l.cr, DH]),
                tr(&st[3], 0, &[HK, l.cr, DH]),
                tr(&rel, r0 * H * E * 2, &[rows, H, E]),
                tr(&pq, r0 * nqw * 2, &[rows, nqw]),
            ],
            vec![tr(&o, r0 * H * DH * 2, &[rows, H * DH]), tr(&st[2], 0, &[HK, l.cr, DH]), tr(&st[3], 0, &[HK, l.cr, DH])],
            a,
        );
        att.pos = pos;
        let need = be.kernels().iter().find(|k| k.name == "rel_sink_attn_bf16").unwrap().scratch_bytes(&ShapeInfo::from_call(&att));
        let sb = be.alloc(need, MemKind::Device).unwrap();
        att.scratch = Some(TensorRef { buf: sb, offset: 0, dtype: DType::U8, shape: vec![need] });
        be.launch("rel_sink_attn_bf16", &att, 0).unwrap();
        be.synchronize(0).unwrap();
    }
    let outs = vec![read(be, &q, t * H * DH), read(be, &rel, t * H * E), read(be, &k, t * ckv), read(be, &v, t * ckv), read(be, &o, t * H * DH)];
    let state = st.iter().zip(&inp.state0).map(|(b, s)| read(be, b, s.len())).collect();
    Run { outs, state }
}

fn same(a: &Run, b: &Run, what: &str) {
    for (i, (x, y)) in a.outs.iter().zip(&b.outs).enumerate() {
        if i < 4 && (x.iter().all(|&v| v == 0) || y.iter().all(|&v| v == 0)) {
            continue; // a fused run has no q / rel / k / v
        }
        let bad = x.iter().zip(y).filter(|(p, q)| p != q).count();
        assert_eq!(bad, 0, "{what}: output {} (q, rel, k, v, o) differs in {bad} of {} values", i, x.len());
    }
    for (i, (x, y)) in a.state.iter().zip(&b.state).enumerate() {
        let bad = x.iter().zip(y).filter(|(p, q)| p != q).count();
        assert_eq!(bad, 0, "{what}: state {} (k ring, v ring, k cache, v cache) differs in {bad} of {} values", i, x.len());
    }
}

#[test]
fn rows_exact_real_shapes() {
    let be = backend();
    let swa = Layer { window: 1024, ring: true, cr: 1040, dynamic: false };
    let full = Layer { window: 0, ring: false, cr: 4096, dynamic: true };
    let mut n = 0;
    // full at 8180..: 32 splits of 288 keys, the trailing ones empty
    let full_long = Layer { window: 0, ring: false, cr: 8300, dynamic: true };
    for (l, poss) in [(&swa, vec![0usize, 1030, 2100, 1020]), (&full, vec![3, 1018, 3000, 1500]), (&full_long, vec![8180])] {
        for &t in &[2usize, 4, 7, 16] {
            for &pos0 in poss.iter() {
                let inp = make(l, t, (t * 977 + pos0) as u64);
                let multi = run(&be, l, &inp, t, pos0, t, false, false);
                let one = run(&be, l, &inp, t, pos0, 1, false, false);
                let one_dev = run(&be, l, &inp, t, pos0, 1, true, false);
                let what = format!("window {} ring {} T={t} pos0={pos0}", l.window, l.ring);
                same(&multi, &one, &format!("{what} (host positions)"));
                same(&multi, &one_dev, &format!("{what} (device position)"));
                // InkAttnDecode: R rows, R x 1 row (host and device position) all bitwise the unfused pair
                for (rows, dev) in [(t, false), (1, false), (1, true), (t, true)] {
                    if dev && rows == t && t > 1 {
                        continue; // a device position covers the whole call: same as host for R rows
                    }
                    let f = run(&be, l, &inp, t, pos0, rows, dev, true);
                    same(&multi, &f, &format!("{what} (InkAttnDecode, {rows}-row calls, device position {dev})"));
                }
                // the attention output is not trivially zero
                assert!(multi.outs[4].iter().any(|&x| x & 0x7fff != 0), "{what}: zero output");
                n += 1;
            }
        }
    }
    eprintln!("{n} rows-exact cases bitwise");
}
