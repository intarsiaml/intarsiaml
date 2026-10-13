//! Matmul microbenchmark on the Qwen3-8B shapes: decode GEMV bandwidth (the number the HIP track is gated on)
//! and prefill GEMM TFLOPS. Needs a GPU; ignored by default:
//!   cargo test --release -p intarsia-backend-hip --test matmul_bench -- --ignored --nocapture
//! `INTARSIA_HIP_GEMV=R,U` and `INTARSIA_HIP_GEMM_TILE=32|64|128` select kernel variants for tuning sweeps;
//! `INTARSIA_HIP_BW=627` is the roofline in GB/s (default: the measured 627 of the R9700).
#![cfg(ik_hip)]

use intarsia_core::{Backend, DType, MemKind, Op, OpAttrs, OpCall, TensorRef};
use std::time::Instant;

fn device() -> Option<std::sync::Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn fill(be: &dyn Backend, bytes: usize) -> intarsia_core::Buffer {
    let b = be.alloc(bytes, MemKind::Device).unwrap();
    // small finite bf16 values (exponent ~2^-3..2^-1) so the data is not all zeros (power) and never NaN
    let pat: Vec<u8> = (0..bytes).map(|i| if i % 2 == 1 { 0x3c + (i / 2 % 5) as u8 } else { (i * 37 % 251) as u8 }).collect();
    be.upload(&b, 0, &pat, 0).unwrap();
    b
}

struct Run {
    us: f64,
    kernel: String,
}

/// Time `op(x[m,k], w[n,k])` over rotating weight copies so the 64 MB MALL / L2 cannot hold the weights.
fn time_matmul(be: &dyn Backend, op: Op, m: usize, n: usize, k: usize, kernel: &str, out_dtype: DType) -> Run {
    let wbytes = n * k * 2;
    let copies = (768usize << 20).div_ceil(wbytes).clamp(2, 64);
    let x = fill(be, m * k * 2);
    let ws: Vec<_> = (0..copies).map(|_| fill(be, wbytes)).collect();
    let out = be.alloc(m * n * out_dtype.size_bytes(), MemKind::Device).unwrap();
    let calls: Vec<OpCall> = ws
        .iter()
        .map(|w| {
            OpCall::new(
                op,
                vec![TensorRef::new(x.clone(), DType::Bf16, &[m, k]), TensorRef::new(w.clone(), DType::Bf16, &[n, k])],
                vec![TensorRef::new(out.clone(), out_dtype, &[m, n])],
                OpAttrs::default(),
            )
        })
        .collect();
    for c in calls.iter().take(3) {
        be.launch(kernel, c, 0).unwrap();
    }
    be.synchronize(0).unwrap();
    let rounds = (200 / copies).max(3);
    let t0 = Instant::now();
    for _ in 0..rounds {
        for c in &calls {
            be.launch(kernel, c, 0).unwrap();
        }
    }
    be.synchronize(0).unwrap();
    Run { us: t0.elapsed().as_secs_f64() * 1e6 / (rounds * copies) as f64, kernel: kernel.to_string() }
}

#[test]
#[ignore]
fn gemv_bandwidth_on_qwen3_8b_shapes() {
    let Some(be) = device() else { return };
    let roof: f64 = std::env::var("INTARSIA_HIP_BW").ok().and_then(|v| v.parse().ok()).unwrap_or(627.0);
    // (name, N, K, per-token multiplicity: layers x count)
    let layer: &[(&str, usize, usize, f64)] = &[
        ("q_proj   ", 4096, 4096, 36.0),
        ("k_proj   ", 1024, 4096, 36.0),
        ("v_proj   ", 1024, 4096, 36.0),
        ("o_proj   ", 4096, 4096, 36.0),
        ("gate_proj", 12288, 4096, 36.0),
        ("up_proj  ", 12288, 4096, 36.0),
        ("down_proj", 4096, 12288, 36.0),
    ];
    println!("roofline {roof} GB/s; weights rotate over >= 768 MB so the cache cannot hold them");
    let (mut bytes_tok, mut us_tok) = (0.0f64, 0.0f64);
    for &(name, n, k, mult) in layer {
        let r = time_matmul(&*be, Op::Matmul, 1, n, k, "matmul_bf16_gemv", DType::Bf16);
        let bytes = (n * k * 2) as f64;
        let gbs = bytes / r.us / 1e3;
        println!("{name} N={n:6} K={k:6}: {:8.1} us  {:6.1} GB/s  {:5.1}% of {roof}  [{}]", r.us, gbs, gbs / roof * 100.0, r.kernel);
        bytes_tok += bytes * mult;
        us_tok += r.us * mult;
    }
    let lm = time_matmul(&*be, Op::Logits, 1, 151936, 4096, "logits_bf16_gemv", DType::F32);
    let lm_bytes = (151936usize * 4096 * 2) as f64;
    println!("lm_head   N=151936 K=  4096: {:8.1} us  {:6.1} GB/s  {:5.1}% of {roof}", lm.us, lm_bytes / lm.us / 1e3, lm_bytes / lm.us / 1e3 / roof * 100.0);
    bytes_tok += lm_bytes;
    us_tok += lm.us;
    let agg = bytes_tok / us_tok / 1e3;
    println!(
        "per token: {:.2} GB of weights in {:.2} ms of GEMV = {:.1} GB/s ({:.1}% of {roof}); weight-stream floor {:.1} tok/s, GEMV-only ceiling {:.1} tok/s",
        bytes_tok / 1e9,
        us_tok / 1e3,
        agg,
        agg / roof * 100.0,
        roof * 1e9 / bytes_tok,
        1e6 / us_tok
    );
}

#[test]
#[ignore]
fn gemm_throughput_on_qwen3_8b_shapes() {
    let Some(be) = device() else { return };
    println!("prefill GEMM (matmul_bf16, WMMA); TFLOPS counts 2*M*N*K");
    for m in [64usize, 256, 512, 1024, 2048] {
        for &(name, n, k) in &[("q/o ", 4096usize, 4096usize), ("gate", 12288, 4096), ("down", 4096, 12288)] {
            let r = time_matmul(&*be, Op::Matmul, m, n, k, "matmul_bf16", DType::Bf16);
            let tf = 2.0 * (m * n * k) as f64 / r.us / 1e6;
            println!("M={m:5} {name} N={n:6} K={k:6}: {:9.1} us  {:6.1} TFLOPS  [{}]", r.us, tf, r.kernel);
        }
    }
}
