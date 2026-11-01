//! DeepSeek-V4.1 `transpose01`, `engram_gather` and `engram_merge` (lane tp2, kernels/cuda/dsv41_engram.cu) at the model's shapes
//! (wo_a groups [T, 4, 4096], i32 / f32 route tables, 24 Engram columns of Dh 256, S = 4 streams of D = 5120) against the cpu
//! reference or, for byte movement, against a host transpose; plus an ignored microbenchmark:
//! `INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41_engram -- --ignored --nocapture`.
//! The admission sweep (`intarsia kernels test --backend cuda --op <op>`) is the gate; these add the real shapes.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::judge;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::ops::OpAttrs;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpCall};
use std::sync::Arc;
use std::time::Instant;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("dsv41_engram", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

fn call(op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs) -> HostCall {
    HostCall { op, inputs, outputs, attrs }
}

/// Host byte transpose of the two leading axes.
fn transpose_bytes(x: &HostTensor) -> HostTensor {
    let (a, b, c) = (x.shape[0], x.shape[1], x.shape[2]);
    let row = c * x.dtype.size_bytes();
    let mut y = vec![0u8; x.data.len()];
    for p in 0..a {
        for q in 0..b {
            y[(q * a + p) * row..(q * a + p + 1) * row].copy_from_slice(&x.data[(p * b + q) * row..(p * b + q + 1) * row]);
        }
    }
    HostTensor::new(x.dtype, vec![b, a, c], y).unwrap()
}

/// transpose01 is byte movement: every dtype, odd rows (unaligned vector widths), the wo_a and route-table shapes.
#[test]
fn transpose01_is_a_byte_transpose() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(7);
    let mut failed = Vec::new();
    let shapes: [(usize, usize, usize); 9] =
        [(1, 4, 4096), (4, 1, 4096), (37, 4, 4096), (4, 37, 4096), (257, 6, 1), (7, 257, 1), (3, 5, 3), (5, 3, 6), (2, 129, 66)];
    for dt in [DType::Bf16, DType::F16, DType::F32, DType::I32, DType::U8] {
        for (a, b, c) in shapes {
            let n = a * b * c * dt.size_bytes();
            let x = HostTensor::new(dt, vec![a, b, c], (0..n).map(|_| rng.next_u64() as u8).collect()).unwrap();
            let want = transpose_bytes(&x);
            let hc = call(Op::Transpose01, vec![x], vec![OutSpec::new(dt, &[b, a, c])], OpAttrs::default());
            let got = run_host(&*cuda, "transpose01", &hc).unwrap();
            let ok = got[0].data == want.data;
            println!("{:<4} [{a},{b},{c}] {dt}", if ok { "ok" } else { "FAIL" });
            if !ok {
                failed.push(format!("[{a},{b},{c}] {dt}"));
            }
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

fn gather_call(rng: &mut SplitMix64, t: usize, dt: DType) -> HostCall {
    // 24 columns of Dh 256: 256 finite e4m3 bytes (either sign), then 8 E8M0 scales near 2^0
    let (c, dh) = (24usize, 256usize);
    let r = dh + dh / 32;
    let data: Vec<u8> = (0..t * c * r)
        .map(|j| {
            let v = rng.next_u64();
            if j % r >= dh {
                120 + (v % 10) as u8
            } else {
                (v % 0x7E) as u8 | ((v >> 8) as u8 & 0x80)
            }
        })
        .collect();
    call(
        Op::EngramGather,
        vec![HostTensor::new(DType::U8, vec![t, c, r], data).unwrap()],
        vec![OutSpec::new(dt, &[t, c * dh])],
        OpAttrs::default(),
    )
}

fn merge_call(rng: &mut SplitMix64, t: usize, (dh, dkv, dw, dout): (DType, DType, DType, DType)) -> HostCall {
    let (s, d) = (4usize, 5120usize);
    call(
        Op::EngramMerge,
        vec![
            rand_t(rng, dh, &[t, s, d], 1.0),
            rand_t(rng, dkv, &[t, (s + 1) * d], 1.0),
            rand_t(rng, dw, &[s, d], 1.0),
            rand_t(rng, dw, &[s, d], 1.0),
        ],
        vec![OutSpec::new(dout, &[t, s, d])],
        OpAttrs { eps: 1e-6, ..Default::default() },
    )
}

/// engram_gather and engram_merge at the model's shapes (incl. a mixed-dtype merge) against the cpu reference.
#[test]
fn engram_real_shapes_against_the_cpu_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let kernels = cuda.kernels();
    let mut rng = SplitMix64::new(11);
    let (b, f) = (DType::Bf16, DType::F32);
    let mut cases: Vec<(String, &str, HostCall)> = Vec::new();
    for t in [1usize, 3, 129] {
        for dt in [b, DType::F16, f] {
            cases.push((format!("engram_gather T={t} {dt}"), "engram_gather", gather_call(&mut rng, t, dt)));
        }
        for ty in [(b, b, b, b), (f, f, f, f), (DType::F16, DType::F16, DType::F16, DType::F16), (b, f, f, b), (f, b, b, f)] {
            cases.push((format!("engram_merge T={t} {ty:?}"), "engram_merge", merge_call(&mut rng, t, ty)));
        }
    }
    let mut failed = Vec::new();
    for (label, kernel, hc) in cases {
        let want = run_host(&cpu, &format!("ref_{}", hc.op.as_str()), &hc).unwrap();
        let got = run_host(&*cuda, kernel, &hc).unwrap();
        let kern = kernels.iter().find(|k| k.name == kernel).unwrap();
        let (ok, ma, rl) = judge(kern, &got, &want);
        println!("{:<4} {label:<62} max_abs={ma:.3e} rel_l2={rl:.3e}", if ok { "ok" } else { "FAIL" });
        if !ok {
            failed.push(label);
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

fn device_call(b: &dyn Backend, hc: &HostCall) -> OpCall {
    let put = |t: &HostTensor| {
        let buf = b.alloc(t.data.len().max(16), MemKind::Device).unwrap();
        b.upload(&buf, 0, &t.data, 0).unwrap();
        TensorRef::new(buf, t.dtype, &t.shape)
    };
    let inputs = hc.inputs.iter().map(put).collect();
    let outputs = hc.outputs.iter().map(|o| put(&HostTensor::zeros(o.dtype, &o.shape))).collect();
    let c = OpCall::new(hc.op, inputs, outputs, hc.attrs.clone());
    c.check().unwrap();
    c
}

fn time_us(b: &dyn Backend, kernel: &str, c: &OpCall, iters: usize) -> f64 {
    for _ in 0..3 {
        b.launch(kernel, c, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        b.launch(kernel, c, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

/// us per call of the three kernels at decode (T = 1) and a 2048-row prefill chunk.
#[test]
#[ignore]
fn bench_engram_and_transpose() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(3);
    let b = DType::Bf16;
    for t in [1usize, 2048] {
        let wo_a = call(Op::Transpose01, vec![rand_t(&mut rng, b, &[t, 4, 4096], 1.0)], vec![OutSpec::new(b, &[4, t, 4096])], OpAttrs::default());
        let back = call(Op::Transpose01, vec![rand_t(&mut rng, b, &[4, t, 4096], 1.0)], vec![OutSpec::new(b, &[t, 4, 4096])], OpAttrs::default());
        let route = call(
            Op::Transpose01,
            vec![HostTensor::new(DType::I32, vec![t, 6, 1], vec![0; t * 24]).unwrap()],
            vec![OutSpec::new(DType::I32, &[6, t, 1])],
            OpAttrs::default(),
        );
        for (label, kernel, hc) in [
            (format!("transpose01 [{t},4,4096] bf16"), "transpose01", wo_a),
            (format!("transpose01 [4,{t},4096] bf16"), "transpose01", back),
            (format!("transpose01 [{t},6,1] i32"), "transpose01", route),
            (format!("engram_gather T={t} bf16"), "engram_gather", gather_call(&mut rng, t, b)),
            (format!("engram_merge T={t} bf16"), "engram_merge", merge_call(&mut rng, t, (b, b, b, b))),
            (format!("engram_merge T={t} bf16, f32 weights"), "engram_merge", merge_call(&mut rng, t, (b, b, DType::F32, b))),
        ] {
            let dc = device_call(&*cuda, &hc);
            println!("{label:<48} {:>10.1} us", time_us(&*cuda, kernel, &dc, if t == 1 { 500 } else { 50 }));
        }
    }
}
