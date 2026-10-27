//! W4A8 decode with the activation quantisation fused into the GEMV block (`INTARSIA_HIP_MXFP4_A8_FUSE=1`, kernels/hip/mxfp4_a8.hip
//! `mxfp4_a8_rows_fused_kernel`): the rows launchers (`matmul_mxfp4_a8_gemv_rows`, `matmul_mxfp4_pair_a8_gemv_rows`) at M = 1..16,
//! K in {5120, 6144, 17408, 2080}, N in {17, 1024, 5120, 301} against the f64 reference of the same math (rel L2 < 3e-5), and
//! batch invariance (rows at M = 5 are bitwise the rows run at M = 1).
//!
//! The knob is read once per process, so this file is its own test binary and has ONE test: it sets the knob to 1 before the first
//! launch unless the caller already set it (INTARSIA_HIP_MXFP4_A8_FUSE=0 runs the very same cases on the pre-pass path). The fused
//! kernel is used where its LDS size fits (al16(M*K) + al16(4*M) <= 60000: K = 5120 up to M = 11, K = 6144 up to M = 9, K = 17408
//! up to M = 3, K = 2080 all M); larger M fall back to the pre-pass inside the same process.
//!
//! "Fused == pre-pass, bitwise" is checked by running this file twice and comparing the printed line (an FNV-64 over every output
//! byte of every launch, in order):
//!   INTARSIA_HIP_MXFP4_A8_FUSE=0 cargo test --release -p intarsia-backend-hip --test mxfp4_a8_fuse -- --nocapture | grep FUSEHASH
//!   INTARSIA_HIP_MXFP4_A8_FUSE=1 cargo test --release -p intarsia-backend-hip --test mxfp4_a8_fuse -- --nocapture | grep FUSEHASH
//!
//! Needs a GPU: skips without one.
#![cfg(ik_hip)]

use intarsia_core::backend::Backend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{matmul_mxfp4_case, matmul_mxfp4_pair_case};
use intarsia_core::quant::reference::matmul_mxfp4_a8;
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor};
use std::sync::Arc;

const KNOB: &str = "INTARSIA_HIP_MXFP4_A8_FUSE";

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            assert!(std::env::var("INTARSIA_REQUIRE_GPU").is_err(), "INTARSIA_REQUIRE_GPU set but no HIP device: {e}");
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += (*x as f64 - y) * (*x as f64 - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

/// rows r0..r0+m of the M-row call `big`, outputs re-shaped to the narrower call (every output is [m, n_i])
fn rows_of(big: &HostCall, r0: usize, m: usize, k: usize) -> HostCall {
    let x = &big.inputs[0];
    let mut hc = big.clone();
    hc.inputs[0] = HostTensor::new(x.dtype, vec![m, k], x.data[r0 * k * 2..(r0 + m) * k * 2].to_vec()).unwrap();
    hc.outputs = big.outputs.iter().map(|o| OutSpec::new(o.dtype, &[m, o.shape[1]])).collect();
    hc
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

#[test]
fn a8_fused_is_bitwise_the_unfused() {
    // before the first launch: the launcher reads the knob once per process
    if std::env::var_os(KNOB).is_none() {
        std::env::set_var(KNOB, "1");
    }
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(9701);
    let mut hash = 0xcbf29ce484222325u64;
    let mut launches = 0usize;
    for k in [5120usize, 6144, 17408, 2080] {
        for n in [17usize, 1024, 5120, 301] {
            let n1 = n / 3 + 5; // the pair's second matrix: the warps' row pairs straddle the two
            let single = matmul_mxfp4_case(&mut rng, 16, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::F32);
            let pair = matmul_mxfp4_pair_case(&mut rng, 16, n, n1, k, Mxfp4Act::Fp8, DType::Bf16, DType::F32);
            // the f64 reference once for the 16 rows; row r of any M-row call is row r of this (the math is per row)
            let x = single.inputs[0].to_f32();
            let want = matmul_mxfp4_a8(&x, 16, k, &single.inputs[1].data, &single.inputs[2].data, n);
            let px = pair.inputs[0].to_f32();
            let want0 = matmul_mxfp4_a8(&px, 16, k, &pair.inputs[1].data, &pair.inputs[2].data, n);
            let want1 = matmul_mxfp4_a8(&px, 16, k, &pair.inputs[3].data, &pair.inputs[4].data, n1);
            // outputs of the M = 1 call per row, for the batch-invariance check
            let mut solo: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
            for m in 1..=16usize {
                let got = run_host(&*hip, "matmul_mxfp4_a8_gemv_rows", &rows_of(&single, 0, m, k)).unwrap().remove(0);
                let r = rel_l2(&got.to_f32(), &want[..m * n]);
                assert!(r < 3e-5, "gemv_rows M={m} N={n} K={k}: rel_l2 {r:.3e}");
                fnv(&mut hash, &got.data);
                let o = run_host(&*hip, "matmul_mxfp4_pair_a8_gemv_rows", &rows_of(&pair, 0, m, k)).unwrap();
                let (r0, r1) = (rel_l2(&o[0].to_f32(), &want0[..m * n]), rel_l2(&o[1].to_f32(), &want1[..m * n1]));
                assert!(r0 < 3e-5 && r1 < 3e-5, "pair_rows M={m} N={n}+{n1} K={k}: rel_l2 {r0:.3e} / {r1:.3e}");
                fnv(&mut hash, &o[0].data);
                fnv(&mut hash, &o[1].data);
                launches += 2;
                if m == 1 {
                    for r in 0..5 {
                        let g = run_host(&*hip, "matmul_mxfp4_a8_gemv_rows", &rows_of(&single, r, 1, k)).unwrap().remove(0);
                        let p = run_host(&*hip, "matmul_mxfp4_pair_a8_gemv_rows", &rows_of(&pair, r, 1, k)).unwrap();
                        solo.push((g.data, p[0].data.clone(), p[1].data.clone()));
                    }
                }
                if m == 5 {
                    for (r, (g, p0, p1)) in solo.iter().enumerate() {
                        assert!(got.data[r * n * 4..(r + 1) * n * 4] == g[..], "gemv_rows N={n} K={k}: row {r} at M=5 differs from M=1");
                        assert!(o[0].data[r * n * 4..(r + 1) * n * 4] == p0[..], "pair out0 N={n}+{n1} K={k}: row {r} at M=5 differs from M=1");
                        assert!(o[1].data[r * n1 * 4..(r + 1) * n1 * 4] == p1[..], "pair out1 N={n}+{n1} K={k}: row {r} at M=5 differs from M=1");
                    }
                }
            }
        }
    }
    println!("FUSEMODE {KNOB}={} launches={launches}", std::env::var(KNOB).unwrap_or_default());
    println!("FUSEHASH {hash:016x}");
}
