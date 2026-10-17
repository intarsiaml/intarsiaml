//! `matmul_fp8_pair_gemv` at every block shape it can run (R rows per warp 1, 2, 4; W warps per block 4, 8, 16, forced through
//! INTARSIA_HIP_FP8_PAIR): each must be *bitwise* the two `matmul_fp8_gemv_v2` launches it replaces, whatever the split of
//! the rows into blocks (every row is summed in the same order). The shapes are the 35B's two pairs and edges for every block
//! size: a first segment that ends mid-block, a second shorter than a block, one row, K past the prefetch window, W8A8 and
//! W8A16 segments in one call.
//!
//! Its own test binary with one test: it sets an environment variable, which no other test thread may read at the same time.
//! scripts/hip-decode/mutate.py (`pair-nb0`, `pair-grid`) breaks the kernel's and the launcher's block arithmetic for the
//! default shape; fused_decode.rs fails on both (verified), this file is expected to as well.
//!
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_core::exec::run_host;
use intarsia_core::quant::cases::matmul_fp8_pair_case;
use intarsia_core::quant::fp8::scales_tensor;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, Op, OpAttrs};

#[test]
fn matmul_fp8_pair_every_block_shape_is_bitwise_v2() {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let hip = match intarsia_backend_hip::create(ord) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            return;
        }
    };
    let mut rng = SplitMix64::new(63);
    let mut checked = 0;
    for (n0, n1, k) in [
        (8192usize, 4096usize, 2048usize),
        (512, 512, 2048),
        (17, 33, 48),
        (130, 7, 1040),
        (1, 1, 16),
        (517, 259, 4112),
        (64, 48, 64),
    ] {
        // (segment 0 W8A8, segment 1 W8A8 / W8A16, input scale of segment 1)
        for (aq0, aq1, is1) in [(true, true, 0.3f32), (true, false, 0.0), (false, true, 0.25)] {
            let mut hc = matmul_fp8_pair_case(&mut rng, 1, n0, n1, k, aq0, aq1, DType::Bf16, DType::Bf16);
            let s = hc.inputs[4].to_f32();
            hc.inputs[4] = scales_tensor(s[0] * 1.7, if aq1 { is1 } else { 0.0 });
            let mut want = Vec::new();
            for (j, w, sc) in [(0usize, 1usize, 2usize), (1, 3, 4)] {
                let one = intarsia_core::exec::HostCall {
                    op: Op::MatmulFp8,
                    inputs: vec![hc.inputs[0].clone(), hc.inputs[w].clone(), hc.inputs[sc].clone()],
                    outputs: vec![hc.outputs[j].clone()],
                    attrs: OpAttrs::default(),
                };
                want.push(run_host(&*hip, "matmul_fp8_gemv_v2", &one).unwrap().remove(0));
            }
            for r in [1, 2, 4] {
                for w in [4, 8, 16] {
                    std::env::set_var("INTARSIA_HIP_FP8_PAIR", format!("{r},{w}"));
                    let got = run_host(&*hip, "matmul_fp8_pair_gemv", &hc).unwrap();
                    for j in 0..2 {
                        assert!(
                            got[j].data == want[j].data,
                            "pair n0={n0} n1={n1} k={k} aq={aq0}/{aq1} R={r} W={w}: output {j} differs from matmul_fp8_gemv_v2"
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    std::env::remove_var("INTARSIA_HIP_FP8_PAIR");
    assert_eq!(checked, 7 * 3 * 9);
}
