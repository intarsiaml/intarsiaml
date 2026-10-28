//! Lane VRC: timing of the W4A8 MXFP4 decode GEMV (`matmul_mxfp4_a8_gemv_rows`, `matmul_mxfp4_pair_a8_gemv_rows`) at M = 1, 3, 8 on
//! the Qwen3.8-27B tp=2 per-rank shapes. Weights rotate over enough copies to defeat the infinity cache. Prints us/launch, GB/s of
//! weight bytes (e2m1 + e8m0 scales) and the ratio to M = 1. Runs only with `VRC_BENCH=1` (it is a measurement, not a test).
//!   VRC_BENCH=1 cargo test --release -p intarsia-backend-hip --test verify_gemv_bench -- --nocapture
#![cfg(ik_hip)]

use intarsia_core::backend::{Backend, MemKind};
use intarsia_core::exec::HostCall;
use intarsia_core::ops::{OpCall, ShapeInfo};
use intarsia_core::admission::gdn_conv_delta_case;
use intarsia_core::quant::cases::{matmul_mxfp4_case, matmul_mxfp4_pair_case};
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, TensorRef};
use std::time::Instant;

fn put(b: &dyn Backend, t: &intarsia_core::HostTensor) -> TensorRef {
    let buf = b.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    b.zero(&buf, 0).unwrap();
    b.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn calls(b: &dyn Backend, hc: &HostCall, copies: usize, kernel: &str) -> Vec<OpCall> {
    let x = put(b, &hc.inputs[0]);
    let mut v = Vec::new();
    for _ in 0..copies {
        let mut inputs = vec![x.clone()];
        inputs.extend(hc.inputs[1..].iter().map(|t| put(b, t)));
        let outputs: Vec<TensorRef> = hc.outputs.iter().map(|o| put(b, &intarsia_core::HostTensor::zeros(o.dtype, &o.shape))).collect();
        let mut call = OpCall::new(hc.op, inputs, outputs, hc.attrs.clone());
        call.check().unwrap();
        let k = b.kernels().iter().find(|k| k.name == kernel && k.op == hc.op).unwrap();
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            let buf = b.alloc(need.max(16), MemKind::Device).unwrap();
            call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
        }
        v.push(call);
    }
    v
}

#[test]
fn verify_gemv_microbench() {
    if std::env::var_os("VRC_BENCH").is_none() {
        eprintln!("skipping: VRC_BENCH unset");
        return;
    }
    let Ok(b) = intarsia_backend_hip::create(std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) else {
        eprintln!("skipping: no HIP device");
        return;
    };
    let mut rng = SplitMix64::new(77);
    // (name, pair?, N0, N1, K): the 27B tp=2 per-rank decode shapes
    let shapes: [(&str, bool, usize, usize, usize); 5] = [
        ("gate+up pair 8704+8704 K5120", true, 8704, 8704, 5120),
        ("down 5120 K8704", false, 5120, 0, 8704),
        ("qkv+z 8192 K5120", false, 8192, 0, 5120),
        ("q+gate 6144 K5120", false, 6144, 0, 5120),
        ("out/o 5120 K3072", false, 5120, 0, 3072),
    ];
    let ms: Vec<usize> = std::env::var("VRC_MS").ok().map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 2, 3, 4, 8]);
    for (name, pair, n0, n1, k) in shapes {
        let wbytes = (n0 + n1) * (k / 2 + k / 32);
        let copies = (400usize << 20).div_ceil(wbytes).max(2);
        let mut base = 0f64;
        for &m in &ms {
            let (hc, kernel) = if pair {
                (matmul_mxfp4_pair_case(&mut rng, m, n0, n1, k, Mxfp4Act::Fp8, DType::Bf16, DType::F32), "matmul_mxfp4_pair_a8_gemv_rows")
            } else {
                (matmul_mxfp4_case(&mut rng, m, n0, k, Mxfp4Act::Fp8, DType::Bf16, DType::F32), "matmul_mxfp4_a8_gemv_rows")
            };
            let cs = calls(&*b, &hc, copies, kernel);
            for c in cs.iter().take(2) {
                b.launch(kernel, c, 0).unwrap();
            }
            b.synchronize(0).unwrap();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                for c in &cs {
                    b.launch(kernel, c, 0).unwrap();
                }
                b.synchronize(0).unwrap();
                best = best.min(t0.elapsed().as_secs_f64() * 1e6 / cs.len() as f64);
            }
            if m == 1 {
                base = best;
            }
            println!(
                "VRCBENCH {name:<32} M={m:<2} {best:>8.1} us  {:>6.0} GB/s  x{:.2} of M=1",
                wbytes as f64 / best / 1e3,
                best / base
            );
        }
    }
    // gdn_conv_delta_bf16 at the 27B tp=2 per-rank shape (8 key / 24 value heads of 128, 4 taps): one launch of T tokens
    for t in [1usize, 2, 3, 4, 8] {
        let hc = gdn_conv_delta_case(&mut rng, t, 8, 24, 128, 128, 4, DType::Bf16);
        let kernel = "gdn_conv_delta_bf16";
        let cs = calls(&*b, &hc, 4, kernel);
        for c in &cs {
            b.launch(kernel, c, 0).unwrap();
        }
        b.synchronize(0).unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..50 {
                for c in &cs {
                    b.launch(kernel, c, 0).unwrap();
                }
            }
            b.synchronize(0).unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1e6 / (50 * cs.len()) as f64);
        }
        println!("VRCBENCH gdn_conv_delta T={t:<2} {best:>8.1} us/launch");
    }
}
