//! The device sampler's Metal kernel (`gumbel_argmax_f32`, `Op::GumbelArgmax`, lane ink-hostgap) on the GPU: the tokens of
//! the cpu definition (`intarsia_core::gumbel::argmax_row`) with and without the softcap, at V % 4 != 0 and Inkling-2's 200058
//! rows; a row draws the same token in a 5-row call as alone and from a key at an offset; and the Metal draws follow
//! softmax(softcap(l) / T) (chi-square at T 0.7 and 1.0, softcap on and off).

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::kernels::GUMBEL_ARGMAX_F32;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::gumbel::{argmax_row, key_row};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, Op, OpAttrs, OpCall};

fn put_f32(be: &MetalBackend, v: &[f32]) -> Buffer {
    let b = be.alloc(v.len() * 4 + 16, MemKind::Device).unwrap();
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    be.upload(&b, 0, &bytes, 0).unwrap();
    b
}

fn put_keys(be: &MetalBackend, k: &[[i32; 4]]) -> Buffer {
    let b = be.alloc(k.len() * 16 + 16, MemKind::Device).unwrap();
    let bytes: Vec<u8> = k.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    be.upload(&b, 0, &bytes, 0).unwrap();
    b
}

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

/// One `GumbelArgmax` of `r` rows of width `v` (logits at row offset `lrow`, keys at row offset `krow`), on the GPU.
fn draw(be: &MetalBackend, l: &Buffer, lrow: usize, k: &Buffer, krow: usize, r: usize, v: usize, cap: f32) -> Vec<i32> {
    let ob = be.alloc(r * 4 + 16, MemKind::Device).unwrap();
    let call = OpCall::new(
        Op::GumbelArgmax,
        vec![t(l, lrow * v * 4, DType::F32, &[r, v]), t(k, krow * 16, DType::I32, &[r, 4])],
        vec![t(&ob, 0, DType::I32, &[r])],
        OpAttrs { scale: cap, ..Default::default() },
    );
    call.check().unwrap();
    be.launch(GUMBEL_ARGMAX_F32, &call, 0).unwrap();
    let mut raw = vec![0u8; r * 4];
    be.download(&ob, 0, &mut raw, 0).unwrap();
    raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn metal_draws_are_the_reference_draws_and_rows_are_independent() {
    let be = backend();
    let mut rng = SplitMix64::new(21);
    for v in [101usize, 4099, 200058] {
        let r = 5;
        let l: Vec<f32> = (0..r * v).map(|_| rng.next_f32() * 60.0 - 20.0).collect();
        let keys: Vec<[i32; 4]> = (0..r).map(|j| key_row(rng.next_u64(), 17 + 1000 * j, [1.0f32, 0.7, 1.3, 1.0, 0.5][j])).collect();
        let (lb, kb) = (put_f32(&be, &l), put_keys(&be, &keys));
        for cap in [0.0f32, 30.0] {
            let want: Vec<i32> = (0..r).map(|j| argmax_row(&l[j * v..(j + 1) * v], cap, keys[j]) as i32).collect();
            assert_eq!(draw(&be, &lb, 0, &kb, 0, r, v, cap), want, "V {v} cap {cap}: 5 rows");
            for j in 0..r {
                assert_eq!(draw(&be, &lb, j, &kb, j, 1, v, cap), vec![want[j]], "V {v} cap {cap}: row {j} alone, key at an offset");
            }
        }
    }
}

/// Chi-square of `n` Metal draws of one row (one key per draw: positions 0..n, one call of `n` rows) against
/// softmax(c(l) / T).
fn chi_square(be: &MetalBackend, logits: &[f32], cap: f32, temp: f32, n: usize, seed: u64) -> f64 {
    let v = logits.len();
    let flat: Vec<f32> = (0..n).flat_map(|_| logits.iter().copied()).collect();
    let keys: Vec<[i32; 4]> = (0..n).map(|p| key_row(seed, p, temp)).collect();
    let got = draw(be, &put_f32(be, &flat), 0, &put_keys(be, &keys), 0, n, v, cap);
    let c: Vec<f64> =
        logits.iter().map(|&l| if cap > 0.0 { (cap * (l / cap).tanh()) as f64 } else { l as f64 } / temp as f64).collect();
    let m = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = c.iter().map(|x| (x - m).exp()).sum();
    let mut counts = vec![0usize; v];
    for g in got {
        counts[g as usize] += 1;
    }
    c.iter()
        .zip(&counts)
        .map(|(x, &o)| {
            let e = n as f64 * (x - m).exp() / z;
            (o as f64 - e).powi(2) / e
        })
        .sum()
}

#[test]
fn metal_draws_follow_the_softmax_at_t07_and_t1_with_the_softcap() {
    let be = backend();
    // 8 tokens, 7 degrees of freedom: chi-square above 24.3 has p = 0.001; with the cap (30) the logits reach past it
    let capped = [31.0f32, 29.5, 28.0, 30.2, 26.0, 33.0, 29.9, 27.5];
    let plain = capped.map(|l| l / 10.0);
    for t in [0.7f32, 1.0] {
        for (logits, cap, seed) in [(&capped, 30.0f32, 7u64), (&plain, 0.0, 11)] {
            let x2 = chi_square(&be, logits, cap, t, 40_000, seed);
            assert!(x2 < 24.3, "T {t} cap {cap}: chi-square {x2:.2}");
        }
    }
}
