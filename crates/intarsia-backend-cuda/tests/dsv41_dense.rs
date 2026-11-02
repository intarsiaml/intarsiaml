//! DeepSeek-V4.1 f16-weight dense linears (`logits_f16w` / `matmul_f16w`, kernels/cuda/dsv41_dense.cu, lane dsv41/tp2): the
//! router (`ffn.gate`, [384, 5120] f16, f32 logits) and the indexer `weights_proj` ([32, 5120] f16) at decode and prefill
//! row counts against the cpu reference, row bits independent of the batch, and an ignored microbenchmark:
//! `INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41_dense -- --ignored --nocapture`.
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
            common::skip("dsv41_dense", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

const K: usize = 5120;

fn case(rng: &mut SplitMix64, m: usize, n: usize, xd: DType, yd: DType) -> HostCall {
    let x = HostTensor::from_f32_as(xd, &[m, K], &rng.normal_vec(m * K, 1.0));
    let w = HostTensor::from_f32_as(DType::F16, &[n, K], &rng.normal_vec(n * K, 1.0 / (K as f32).sqrt()));
    let op = if yd == DType::F32 { Op::Logits } else { Op::Matmul };
    HostCall { op, inputs: vec![x, w], outputs: vec![OutSpec::new(yd, &[m, n])], attrs: OpAttrs::default() }
}

fn kernel_of(yd: DType) -> &'static str {
    if yd == DType::F32 {
        "logits_f16w"
    } else {
        "matmul_f16w"
    }
}

#[test]
fn real_shapes_against_the_cpu_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let kernels = cuda.kernels();
    let mut rng = SplitMix64::new(4101);
    let mut failed = Vec::new();
    for (n, what) in [(384usize, "router"), (32, "indexer weights_proj")] {
        for m in [1usize, 2, 15, 16, 17, 64, 512] {
            for (xd, yd) in [(DType::Bf16, DType::F32), (DType::Bf16, DType::Bf16), (DType::F32, DType::F32)] {
                let hc = case(&mut rng, m, n, xd, yd);
                let kernel = kernel_of(yd);
                let want = run_host(&cpu, &format!("ref_{}", hc.op.as_str()), &hc).unwrap();
                let got = run_host(&*cuda, kernel, &hc).unwrap();
                let kern = kernels.iter().find(|k| k.name == kernel).unwrap();
                let (ok, ma, rl) = judge(kern, &got, &want);
                println!("{:<4} {what} M={m} x={xd:?} y={yd:?} max_abs={ma:.3e} rel_l2={rl:.3e}", if ok { "ok" } else { "FAIL" });
                if !ok {
                    failed.push(format!("{what} M={m} {xd:?}->{yd:?}"));
                }
            }
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Row r of an M-row call is bitwise the one-row call on that row (decode == prefill for the router).
#[test]
fn rows_do_not_depend_on_the_batch() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(4102);
    for n in [384usize, 32] {
        let big = case(&mut rng, 37, n, DType::Bf16, DType::F32);
        let all = run_host(&*cuda, "logits_f16w", &big).unwrap();
        for r in [0usize, 5, 16, 36] {
            let row = HostTensor { dtype: DType::Bf16, shape: vec![1, K], data: big.inputs[0].data[r * K * 2..(r + 1) * K * 2].to_vec() };
            let one = HostCall { op: Op::Logits, inputs: vec![row, big.inputs[1].clone()], outputs: vec![OutSpec::new(DType::F32, &[1, n])], attrs: OpAttrs::default() };
            let got = run_host(&*cuda, "logits_f16w", &one).unwrap();
            assert_eq!(got[0].data, all[0].data[r * n * 4..(r + 1) * n * 4], "N={n} row {r}: one-row call differs from the {}-row call", 37);
        }
    }
}

fn device_call(b: &dyn Backend, hc: &HostCall, copies: usize) -> Vec<OpCall> {
    let put = |t: &HostTensor| {
        let buf = b.alloc(t.data.len().max(16), MemKind::Device).unwrap();
        b.upload(&buf, 0, &t.data, 0).unwrap();
        TensorRef::new(buf, t.dtype, &t.shape)
    };
    let x = put(&hc.inputs[0]);
    let y = put(&HostTensor::zeros(hc.outputs[0].dtype, &hc.outputs[0].shape));
    (0..copies)
        .map(|_| {
            let mut c = OpCall::new(hc.op, vec![x.clone(), put(&hc.inputs[1])], vec![y.clone()], hc.attrs.clone());
            c.check().unwrap();
            c
        })
        .collect()
}

/// us per call, cycling over enough weight copies (64 MiB) that every call reads its weights from DRAM, like decode.
#[test]
#[ignore]
fn bench_router_and_weights_proj() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(4103);
    println!("{:<44} {:>9} {:>9}", "(bf16 x, f16 w, K=5120)", "us/call", "GB/s(w)");
    for (n, what) in [(384usize, "router"), (32, "weights_proj")] {
        for m in [1usize, 4, 512] {
            let yd = if what == "router" { DType::F32 } else { DType::Bf16 };
            let hc = case(&mut rng, m, n, DType::Bf16, yd);
            let wbytes = n * K * 2;
            let copies = ((64usize << 20) / wbytes).clamp(1, 64);
            let calls = device_call(&*cuda, &hc, copies);
            let k = kernel_of(yd);
            for c in &calls {
                cuda.launch(k, c, 0).unwrap();
            }
            cuda.synchronize(0).unwrap();
            let reps = 20;
            let t0 = Instant::now();
            for _ in 0..reps {
                for c in &calls {
                    cuda.launch(k, c, 0).unwrap();
                }
            }
            cuda.synchronize(0).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / (reps * calls.len()) as f64;
            println!("{:<44} {:>9.1} {:>9.0}", format!("{what} N={n} M={m}"), us, wbytes as f64 / 1e3 / us);
        }
    }
}
