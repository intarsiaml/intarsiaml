#![cfg(target_os = "macos")]
//! Lane dm: `logits_rows_scalar_m<M>` (logits_rows.rs) is bitwise the 1-row `Logits` launch per row.

mod common;
use common::*;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall};

/// An M-row (`INTARSIA_METAL_LOGITS_ROWS` default) `logits_bf16_gemv_rows` call over a weight at byte 8 mod 16 (the 27B lm_head) or 2 mod 4,
/// or with K not a multiple of 8, runs `logits_rows_scalar_m<M>`; every row is bitwise the 1-row `logits_bf16_gemv` launch on that
/// activation row (which takes `gemv_f32o_scalar` for these operands), M = 2..=16, N with a partial threadgroup, K across tiles.
#[test]
fn logits_rows_scalar_is_bitwise_the_one_row_launch() {
    std::env::remove_var("INTARSIA_METAL_LOGITS_ROWS");
    let be = backend();
    for &(n, k, woff) in &[(1000usize, 5120usize, 8usize), (333, 700, 2), (77, 1030, 8)] {
        let w = rand_bf16(n * k, (n + k) as u64);
        let mut wpad = vec![0u16; woff / 2];
        wpad.extend_from_slice(&w);
        let wb = upload(&be, &wpad);
        let wt = tref(&wb, woff, &[n, k]);
        for m in 2..=16usize {
            let xb = upload(&be, &rand_bf16(m * k, (m * 7 + k) as u64));
            let x = tref(&xb, 0, &[m, k]);
            let f32buf = |rows: usize| {
                let b = be.alloc(rows * n * 4 + 16, MemKind::Device).unwrap();
                be.upload(&b, 0, &vec![0xEEu8; rows * n * 4], 0).unwrap();
                b
            };
            let (ym, y1) = (f32buf(m), f32buf(m));
            let ft = |b: &Buffer, r0: usize, rows: usize| TensorRef { buf: b.clone(), offset: r0 * n * 4, dtype: DType::F32, shape: vec![rows, n] };
            let call = OpCall::new(Op::Logits, vec![x.clone(), wt.clone()], vec![ft(&ym, 0, m)], OpAttrs::default());
            be.launch("logits_bf16_gemv_rows", &call, 0).unwrap();
            for r in 0..m {
                let c1 = OpCall::new(Op::Logits, vec![x.rows(r, 1), wt.clone()], vec![ft(&y1, r, 1)], OpAttrs::default());
                be.launch("logits_bf16_gemv", &c1, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let (mut a, mut b) = (vec![0u8; m * n * 4], vec![0u8; m * n * 4]);
            be.download(&ym, 0, &mut a, 0).unwrap();
            be.download(&y1, 0, &mut b, 0).unwrap();
            assert!(a == b, "n={n} k={k} woff={woff} m={m}: not bitwise the 1-row launches");
        }
    }
}

/// Timing, not a gate (`-- --ignored --nocapture`, exclusive slot): the 27B head shape (248320 x 5120 bf16) at byte offset 0 (the 16-byte
/// `gemv_f32o_v`) and 8 (the scalar `gemv_f32o_scalar`, as the real lm_head), M = 1, and `logits_bf16_gemv_rows` at M = 2, 3, 4, 8, 16.
#[test]
#[ignore]
fn bench_head_alignment() {
    let be = backend();
    let (n, k) = (248320usize, 5120usize);
    let w: Vec<u16> = rand_bf16(n * k + 8, 7);
    let wb = upload(&be, &w);
    let xb = upload(&be, &rand_bf16(16 * k, 9));
    let yb = be.alloc(16 * n * 4, MemKind::Device).unwrap();
    let time = |f: &dyn Fn()| {
        let mut best = f64::MAX;
        for _ in 0..3 {
            be.synchronize(0).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                f();
            }
            be.synchronize(0).unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1e5);
        }
        best
    };
    for off in [0usize, 8] {
        let wt = tref(&wb, off, &[n, k]);
        for m in [1usize, 2, 3, 4, 8, 16] {
            let x = tref(&xb, 0, &[m, k]);
            let y = TensorRef { buf: yb.clone(), offset: 0, dtype: DType::F32, shape: vec![m, n] };
            let c = OpCall::new(Op::Logits, vec![x, wt.clone()], vec![y], OpAttrs::default());
            let name = if m == 1 { "logits_bf16_gemv" } else { "logits_bf16_gemv_rows" };
            let us = time(&|| be.launch(name, &c, 0).unwrap());
            eprintln!("LRBENCH head {n}x{k} offset {off} M={m} {name}: {us:.0} us");
        }
    }
}
