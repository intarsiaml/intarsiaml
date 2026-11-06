//! Lane df (27B DFlash gap): cost of an M-row verify pass's big kernels on Metal at the Qwen3.8-27B shapes (timing, not a gate).
//! `cargo test --release -p intarsia-backend-metal --test df_verify_bench -- --ignored --nocapture`, in an exclusive fleetq slot on endor.
//! Prints, per kernel and M = 1..8, the wall time of one M-row launch, against M one-row launches.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

fn up(be: &MetalBackend, data: &[u8]) -> Buffer {
    let b = be.alloc(data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, data, 0).unwrap();
    b
}

fn t(buf: &Buffer, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: 0, dtype: dt, shape: shape.to_vec() }
}

/// Wall time per call of `f` (it enqueues; one synchronize per `reps` calls), best of 3 rounds, in microseconds.
fn time_us(be: &MetalBackend, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        be.synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        be.synchronize(0).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
    }
    best
}

fn bf16_rows(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..m * k).flat_map(|_| f32_to_bf16(r.unit() * 0.5).to_le_bytes()).collect()
}

#[test]
#[ignore]
fn df_verify_kernels_27b() {
    let be = backend();
    // MXFP4 W4A16 GEMV (matmul_mxfp4_gemv: _m1 at M = 1, _m8 above), 27B tp=1 shapes: FFN gate/up and down, GDN in-proj, attention q
    for (n, k) in [(17408usize, 5120usize), (5120, 17408), (12288, 5120), (5120, 6144)] {
        let mut r = Rng(7);
        let w: Vec<u8> = (0..n * k / 2).map(|_| (r.next_u64() >> 40) as u8).collect();
        let sc: Vec<u8> = (0..n * k / 32).map(|_| 119 + ((r.next_u64() >> 40) % 12) as u8).collect();
        let (wb, sb, ab) = (up(&be, &w), up(&be, &sc), up(&be, &0f32.to_le_bytes()));
        let xb = up(&be, &bf16_rows(8, k, 3));
        let yb = be.alloc(8 * n * 2 + 16, MemKind::Device).unwrap();
        let call = |m: usize, r0: usize| {
            let mut cl = OpCall::new(
                Op::MatmulMxfp4,
                vec![
                    TensorRef { buf: xb.clone(), offset: r0 * k * 2, dtype: DType::Bf16, shape: vec![m, k] },
                    t(&wb, DType::U8, &[n, k / 2]),
                    t(&sb, DType::U8, &[n, k / 32]),
                    t(&ab, DType::F32, &[1]),
                ],
                vec![TensorRef { buf: yb.clone(), offset: r0 * n * 2, dtype: DType::Bf16, shape: vec![m, n] }],
                OpAttrs { mxfp4_w4a16: true, ..OpAttrs::default() },
            );
            let need = be
                .kernels()
                .iter()
                .find(|q| q.name == "matmul_mxfp4_gemv")
                .unwrap()
                .scratch_bytes(&ShapeInfo::from_call(&cl));
            let s = be.alloc(need + 16, MemKind::Device).unwrap();
            cl.scratch = Some(t(&s, DType::U8, &[need.max(16)]));
            cl
        };
        let one: Vec<OpCall> = (0..8).map(|r| call(1, r)).collect();
        let t1 = time_us(&be, 50, || be.launch("matmul_mxfp4_gemv", &one[0], 0).unwrap());
        for m in 1..=8usize {
            let cm = call(m, 0);
            let rows = intarsia_backend_metal::mxfp4::with_mr(0, || {
                time_us(&be, 50, || be.launch("matmul_mxfp4_gemv", &cm, 0).unwrap())
            });
            let mr: Vec<String> = [1usize, 2, 4, 8]
                .iter()
                .filter(|_| m >= 2)
                .map(|&p| {
                    let us = intarsia_backend_metal::mxfp4::with_mr(p, || {
                        time_us(&be, 50, || be.launch("matmul_mxfp4_gemv", &cm, 0).unwrap())
                    });
                    format!("mr {} {us:.1} us ({:.2}x M=1)", if p == 8 { "mma".to_string() } else { format!("p{p}") }, us / t1)
                })
                .collect();
            let serial = time_us(&be, 50, || {
                for c in &one[..m] {
                    be.launch("matmul_mxfp4_gemv", c, 0).unwrap();
                }
            });
            eprintln!(
                "DFBENCH mxfp4 {n}x{k} M={m}: rows launch {rows:.1} us ({:.2}x M=1, {:.0} GB/s), {m} one-row launches {serial:.1} us; {}",
                rows / t1,
                (n * k / 2 + n * k / 32) as f64 / rows / 1e3,
                mr.join(", ")
            );
        }
    }
    if std::env::var_os("DFBENCH_NO_LOGITS").is_some() {
        return;
    }
    // bf16 lm_head logits (f32 out), 248320 x 5120: logits_bf16_gemv (M = 1), logits_bf16_gemv_rows (M = 2..8, the verify head),
    // logits_bf16 (the generic kernel)
    let (n, k) = (248320usize, 5120usize);
    let mut r = Rng(11);
    let w: Vec<u8> = (0..n * k).flat_map(|_| f32_to_bf16(r.unit() * 0.02).to_le_bytes()).collect();
    let wb = up(&be, &w);
    drop(w);
    let xb = up(&be, &bf16_rows(8, k, 5));
    let yb = be.alloc(8 * n * 4 + 16, MemKind::Device).unwrap();
    let lcall = |m: usize, r0: usize| {
        OpCall::new(
            Op::Logits,
            vec![
                TensorRef { buf: xb.clone(), offset: r0 * k * 2, dtype: DType::Bf16, shape: vec![m, k] },
                t(&wb, DType::Bf16, &[n, k]),
            ],
            vec![TensorRef { buf: yb.clone(), offset: r0 * n * 4, dtype: DType::F32, shape: vec![m, n] }],
            OpAttrs::default(),
        )
    };
    let c1 = lcall(1, 0);
    let t1 = time_us(&be, 10, || be.launch("logits_bf16_gemv", &c1, 0).unwrap());
    eprintln!("DFBENCH logits {n}x{k} M=1: logits_bf16_gemv {t1:.1} us ({:.0} GB/s)", (n * k * 2) as f64 / t1 / 1e3);
    for m in 2..=8usize {
        let cm = lcall(m, 0);
        let rows = time_us(&be, 10, || be.launch("logits_bf16_gemv_rows", &cm, 0).unwrap());
        let gen = time_us(&be, 10, || be.launch("logits_bf16", &cm, 0).unwrap());
        eprintln!("DFBENCH logits {n}x{k} M={m}: logits_bf16_gemv_rows {rows:.1} us ({:.2}x M=1), logits_bf16 (generic) {gen:.1} us", rows / t1);
    }
}
