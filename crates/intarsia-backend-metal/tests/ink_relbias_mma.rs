#![cfg(target_os = "macos")]
//! `INTARSIA_INK_RELBIAS_MMA` (opt-in, lane ink-metal brief 49): a prefill InkKvPrologue + RelSinkAttn (NAX) pair with the
//! packed rel form (r and relproj, the bias built per key tile on the accelerators) against the materialised-bias pair, at
//! the real shapes and row counts that are not a multiple of the tiles (query block 64, key tile 32). The prologue's q / k / v,
//! the conv rings and the K / V caches must be bitwise; the attention output differs only through the f32 order of the
//! 16-term bias dot (bf16 rounding point kept), so its max difference is bounded and far below the bias's own effect
//! (the same call with relproj zeroed).
mod common;

use common::*;
use intarsia_backend_metal::inkattn::{set_relbias_mma, INK_KV_PROLOGUE_BF16, REL_SINK_ATTN_BF16_NAX};
use intarsia_backend_metal::bench::bf16_to_f32;
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

struct Layer {
    window: usize,
    ring: bool,
    cr: usize,
    dynamic: bool,
}

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
    use intarsia_backend_metal::bench::f32_to_bf16;
    rand_bf16(n, seed).iter().map(|&v| f32_to_bf16(lo + (hi - lo) * (bf16_to_f32(v) * 0.5 + 0.5))).collect()
}

/// pq, pk, pv, weights (k taps, v taps, q scale, r scale, rel proj), state (k ring, v ring, k cache, v cache).
fn inputs(l: &Layer, t: usize, seed: u64) -> (Vec<Vec<u16>>, Vec<Vec<u16>>, Vec<Vec<u16>>) {
    let ckv = HK * DH;
    let pw = if l.dynamic { ckv + HK * K } else { ckv };
    let x = vec![rand_bf16(t * nq(H, DH, DR), seed), rand_bf16(t * pw, seed + 1), rand_bf16(t * pw, seed + 2)];
    let w = vec![
        rand_bf16(K * ckv, seed + 3),
        rand_bf16(K * ckv, seed + 4),
        scaled(H * DH, seed + 5, 0.5, 6.0),
        scaled(H * DR, seed + 6, 0.5, 3.0),
        rand_bf16(DR * E, seed + 7),
    ];
    let s = vec![rand_bf16(RR * ckv, seed + 8), rand_bf16(RR * ckv, seed + 9), rand_bf16(HK * l.cr * DH, seed + 10), rand_bf16(HK * l.cr * DH, seed + 11)];
    (x, w, s)
}

/// One prefill pair of T rows at pos0; returns (q, k, v, out, the first T * H * 16 rel elements) and the final state.
fn run(be: &intarsia_backend_metal::MetalBackend, l: &Layer, x: &[Vec<u16>], w: &[Vec<u16>], s0: &[Vec<u16>], t: usize, pos0: usize) -> (Vec<Vec<u16>>, Vec<Vec<u16>>) {
    let ckv = HK * DH;
    let pw = x[1].len() / t;
    let nqw = nq(H, DH, DR);
    let (pq, pk, pv) = (upload(be, &x[0]), upload(be, &x[1]), upload(be, &x[2]));
    let wb: Vec<Buffer> = w.iter().map(|v| upload(be, v)).collect();
    let st: Vec<Buffer> = s0.iter().map(|v| upload(be, v)).collect();
    let (q, rel, k, v, o) = (zeros(be, t * H * DH), zeros(be, t * H * E), zeros(be, t * ckv), zeros(be, t * ckv), zeros(be, t * H * DH));
    let mut a = OpAttrs { pos0, n_heads: H, n_kv_heads: HK, head_dim: DH, window: l.window, scale: (DH as f32).powf(-0.5), eps: 1e-6, ..OpAttrs::default() };
    a.ink.qk_clip = 6.0;
    a.ink.rel_clip = 1.5;
    a.ink.d_rel = DR;
    a.ink.kv_ring = l.ring;
    a.ink.dyn_conv = l.dynamic;
    if l.dynamic {
        (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
    }
    let pro = OpCall::new(
        Op::InkKvPrologue,
        vec![
            tr(&pq, 0, &[t, nqw]),
            tr(&pk, 0, &[t, pw]),
            tr(&pv, 0, &[t, pw]),
            tr(&wb[0], 0, &[K, ckv]),
            tr(&wb[1], 0, &[K, ckv]),
            tr(&wb[2], 0, &[H, DH]),
            tr(&wb[3], 0, &[H, DR]),
            tr(&wb[4], 0, &[DR, E]),
        ],
        vec![tr(&q, 0, &[t, H, DH]), tr(&rel, 0, &[t, H, E]), tr(&k, 0, &[t, HK, DH]), tr(&v, 0, &[t, HK, DH]), tr(&st[0], 0, &[RR, ckv]), tr(&st[1], 0, &[RR, ckv])],
        a.clone(),
    );
    be.launch(INK_KV_PROLOGUE_BF16, &pro, 0).unwrap();
    let mut att = OpCall::new(
        Op::RelSinkAttn,
        vec![
            tr(&q, 0, &[t, H, DH]),
            tr(&k, 0, &[t, HK, DH]),
            tr(&v, 0, &[t, HK, DH]),
            tr(&st[2], 0, &[HK, l.cr, DH]),
            tr(&st[3], 0, &[HK, l.cr, DH]),
            tr(&rel, 0, &[t, H, E]),
            tr(&pq, 0, &[t, nqw]),
        ],
        vec![tr(&o, 0, &[t, H * DH]), tr(&st[2], 0, &[HK, l.cr, DH]), tr(&st[3], 0, &[HK, l.cr, DH])],
        a,
    );
    let need = be.kernels().iter().find(|k| k.name == REL_SINK_ATTN_BF16_NAX).unwrap().scratch_bytes(&ShapeInfo::from_call(&att)).max(16);
    let sb = be.alloc(need, MemKind::Device).unwrap();
    att.scratch = Some(TensorRef { buf: sb, offset: 0, dtype: DType::U8, shape: vec![need] });
    be.launch(REL_SINK_ATTN_BF16_NAX, &att, 0).unwrap();
    be.synchronize(0).unwrap();
    let outs = vec![read(be, &q, t * H * DH), read(be, &k, t * ckv), read(be, &v, t * ckv), read(be, &o, t * H * DH), read(be, &rel, t * H * DR)];
    let state = st.iter().zip(s0).map(|(b, s)| read(be, b, s.len())).collect();
    (outs, state)
}

fn max_diff(a: &[u16], b: &[u16]) -> (f32, usize) {
    let mut m = 0.0f32;
    let mut n = 0;
    for (&x, &y) in a.iter().zip(b) {
        let d = (bf16_to_f32(x) - bf16_to_f32(y)).abs();
        assert!(d.is_finite(), "non-finite output");
        if x != y {
            n += 1;
        }
        m = m.max(d);
    }
    (m, n)
}

#[test]
fn packed_rel_bias_matches_the_materialised_bias() {
    let be = backend();
    assert!(be.has_metal4(), "this test needs the metal4.0 library");
    let swa = Layer { window: 1024, ring: true, cr: 1040, dynamic: false };
    let full = Layer { window: 0, ring: false, cr: 4096, dynamic: true };
    let mut worst = 0.0f32;
    for (l, cases) in [(&swa, vec![(37usize, 0usize), (100, 1030), (201, 2100)]), (&full, vec![(37, 3), (100, 1000), (201, 3000), (130, 1500)])] {
        for &(t, pos0) in &cases {
            let (x, w, s0) = inputs(l, t, (t * 131 + pos0) as u64);
            set_relbias_mma(Some(false));
            let (mo, ms) = run(&be, l, &x, &w, &s0, t, pos0);
            set_relbias_mma(Some(true));
            let (po, ps) = run(&be, l, &x, &w, &s0, t, pos0);
            // the bias's own effect: the materialised path with relproj zeroed
            set_relbias_mma(Some(false));
            let mut w0 = w.clone();
            w0[4] = vec![0; DR * E];
            let (zo, _) = run(&be, l, &x, &w0, &s0, t, pos0);
            set_relbias_mma(None);
            let what = format!("window {} T={t} pos0={pos0}", l.window);
            for (i, name) in ["q", "k", "v"].iter().enumerate() {
                assert_eq!(mo[i], po[i], "{what}: prologue {name} must be bitwise");
            }
            for (i, (a, b)) in ms.iter().zip(&ps).enumerate() {
                assert!(a == b, "{what}: state {i} (k ring, v ring, k cache, v cache) must be bitwise");
            }
            let (d, n) = max_diff(&mo[3], &po[3]);
            let (dz, _) = max_diff(&mo[3], &zo[3]);
            eprintln!("{what}: packed vs materialised max |diff| {d:.3e} ({n} of {} values differ); bias effect {dz:.3e}", mo[3].len());
            assert!(po[3].iter().any(|&v| v & 0x7fff != 0), "{what}: zero output");
            // the packed form really ran: the rel output's head holds r, not the materialised bias
            assert!(mo[4] != po[4], "{what}: the rel output is the same with the switch on (packed path not taken)");
            assert!(dz > 0.01, "{what}: the bias has no visible effect ({dz})");
            assert!(d <= 2e-3 && d * 20.0 < dz, "{what}: packed bias off by {d} (bias effect {dz})");
            worst = worst.max(d);
        }
    }
    eprintln!("worst max |diff| {worst:.3e}");
}
