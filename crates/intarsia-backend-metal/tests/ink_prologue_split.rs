#![cfg(target_os = "macos")]
//! Lane ink-prefill: InkKvPrologue of prefill chunks as `ink_kv_prologue_norel_bf16` + `ink_rel_proj_bf16`
//! (on unless `INTARSIA_INK_PROLOGUE_SPLIT=0`, inkprefill.rs) equals `ink_kv_prologue_bf16` bit for bit: q, rel, k, v and both conv
//! rings, at the real shapes (24 / 8 heads, head_dim 128, d_rel 16, extent 1024, K 4), static and dynamic taps with tau,
//! the in-kernel ring write (ring 1040 rows) and the separate ring pass (ring 19 rows), 17 .. 4100 rows, a partial
//! 8-row block, positions 0 and past the tau floor / ring wrap.

mod common;
use common::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::inkling2::ops::nq;
use intarsia_core::{DType, Op, OpAttrs, OpCall};

const H: usize = 24;
const HK: usize = 8;
const DH: usize = 128;
const DR: usize = 16;
const E: usize = 1024;
const K: usize = 4;

fn tr(b: &Buffer, off: usize, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: DType::Bf16, shape: shape.to_vec() }
}

fn read(be: &impl Backend, b: &Buffer, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn filled(be: &impl Backend, n: usize) -> Buffer {
    let b = be.alloc(n * 2, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xA5u8; n * 2], 0).unwrap(); // poison: every element must be written
    b
}

fn scaled(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<u16> {
    use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
    rand_bf16(n, seed).iter().map(|&v| f32_to_bf16(lo + (hi - lo) * (bf16_to_f32(v) * 0.5 + 0.5))).collect()
}

/// One prologue call of `t` rows at `pos0`; returns q, rel, k, v, k ring, v ring.
fn run(be: &MetalBackend, t: usize, pos0: usize, rr: usize, dynamic: bool, split: bool, seed: u64) -> Vec<Vec<u16>> {
    let ckv = HK * DH;
    let pw = if dynamic { ckv + HK * K } else { ckv };
    let nqw = nq(H, DH, DR);
    let (pq, pk, pv) = (upload(be, &rand_bf16(t * nqw, seed)), upload(be, &rand_bf16(t * pw, seed + 1)), upload(be, &rand_bf16(t * pw, seed + 2)));
    let w = [
        upload(be, &rand_bf16(K * ckv, seed + 3)),
        upload(be, &rand_bf16(K * ckv, seed + 4)),
        upload(be, &scaled(H * DH, seed + 5, 0.5, 6.0)),
        upload(be, &scaled(H * DR, seed + 6, 0.5, 3.0)),
        upload(be, &rand_bf16(DR * E, seed + 7)),
    ];
    let (kr, vr) = (upload(be, &rand_bf16(rr * ckv, seed + 8)), upload(be, &rand_bf16(rr * ckv, seed + 9)));
    let (q, rel, k, v) = (filled(be, t * H * DH), filled(be, t * H * E), filled(be, t * ckv), filled(be, t * ckv));
    let mut a = OpAttrs { pos0, n_heads: H, n_kv_heads: HK, head_dim: DH, eps: 1e-6, ..OpAttrs::default() };
    a.ink.qk_clip = 6.0;
    a.ink.rel_clip = 1.5;
    a.ink.d_rel = DR;
    a.ink.dyn_conv = dynamic;
    if dynamic {
        (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
    }
    let call = OpCall::new(
        Op::InkKvPrologue,
        vec![
            tr(&pq, 0, &[t, nqw]),
            tr(&pk, 0, &[t, pw]),
            tr(&pv, 0, &[t, pw]),
            tr(&w[0], 0, &[K, ckv]),
            tr(&w[1], 0, &[K, ckv]),
            tr(&w[2], 0, &[H, DH]),
            tr(&w[3], 0, &[H, DR]),
            tr(&w[4], 0, &[DR, E]),
        ],
        vec![tr(&q, 0, &[t, H, DH]), tr(&rel, 0, &[t, H, E]), tr(&k, 0, &[t, HK, DH]), tr(&v, 0, &[t, HK, DH]), tr(&kr, 0, &[rr, ckv]), tr(&vr, 0, &[rr, ckv])],
        a,
    );
    call.check().unwrap();
    let runs0 = intarsia_backend_metal::inkprefill::prologue_split_runs();
    intarsia_backend_metal::inkprefill::set_prologue_split(Some(split));
    be.launch("ink_kv_prologue_bf16", &call, 0).unwrap();
    be.synchronize(0).unwrap();
    intarsia_backend_metal::inkprefill::set_prologue_split(None);
    // the split kernels must have run exactly when asked (no split-off vs split-off comparison)
    assert_eq!(intarsia_backend_metal::inkprefill::prologue_split_runs() - runs0, usize::from(split), "T={t}: split path taken = {split}?");
    vec![read(be, &q, t * H * DH), read(be, &rel, t * H * E), read(be, &k, t * ckv), read(be, &v, t * ckv), read(be, &kr, rr * ckv), read(be, &vr, rr * ckv)]
}

#[test]
fn prologue_split_is_bitwise_the_prologue() {
    let be = MetalBackend::new().expect("metal");
    let names = ["q", "rel", "k", "v", "k ring", "v ring"];
    let mut n = 0;
    for &(t, pos0, rr, dynamic) in &[
        (17usize, 0usize, 1040usize, false),
        (100, 37, 1040, true),
        (333, 900, 19, true),
        (1037, 0, 1040, false),
        (2048, 0, 1040, false),
        (2048, 2048, 1040, true),
        (2048, 5000, 19, true),
        (1890, 6144, 1040, false),
        // lane ink-prologue2: 4096-row chunks (INTARSIA_INK_LONG_CHUNK=1), one with a partial 8-row block
        (4096, 0, 1040, false),
        (4096, 8192, 19, true),
        (4100, 4099, 1040, true),
    ] {
        let seed = 0x5917 + n as u64 * 31;
        let a = run(&be, t, pos0, rr, dynamic, false, seed);
        let b = run(&be, t, pos0, rr, dynamic, true, seed);
        for ((x, y), w) in a.iter().zip(&b).zip(names) {
            let diff = x.iter().zip(y).filter(|(p, q)| p != q).count();
            assert_eq!(diff, 0, "T={t} pos0={pos0} rr={rr} dyn={dynamic}: {w} differs in {diff} of {} elements", x.len());
        }
        eprintln!("T={t:5} pos0={pos0:5} rr={rr:5} dyn={dynamic}: bitwise");
        n += 1;
    }
}
