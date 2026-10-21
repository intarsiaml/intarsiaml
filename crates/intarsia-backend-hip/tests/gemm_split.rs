//! Round 10 (docs/results/2026-10-19-hip-round10.md): a long bf16 GEMM / one-row GEMV launch is cut along independent
//! output rows (kernels/hip/matmul.hip `split_rows`) so the short dispatch after it does not pay the gfx1201 ~260 us
//! window (INTARSIA_HIP_GEMM_SPLIT: tail / even / 0, opt-in, off by default; _SPLIT_MIN_US trigger, _SPLIT_TAIL_US tail length).
//! Pins the piece plan on the host (the 8B's shapes, the trigger, the kill switch, both modes) and checks on the device
//! that every cut gives every bit of the single launch: GEMM rows that are not a tile multiple, a last piece of one row
//! under a full M >= 128 (the tile must come from the full M), the f32 logits output, and the GEMV over weight rows.
//!
//! Its own test binary with a single test: it sets environment variables, which no other test may race with.
#![cfg(ik_hip)]

use intarsia_backend_hip::gemm_split_rows;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, Op, OpAttrs};

const VARS: [&str; 3] = ["INTARSIA_HIP_GEMM_SPLIT", "INTARSIA_HIP_GEMM_SPLIT_MIN_US", "INTARSIA_HIP_GEMM_SPLIT_TAIL_US"];

fn set(mode: Option<&str>, min: Option<&str>, tail: Option<&str>) {
    for (v, x) in VARS.iter().zip([mode, min, tail]) {
        match x {
            Some(x) => std::env::set_var(v, x),
            None => std::env::remove_var(v),
        }
    }
}

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

fn rand_t(rng: &mut SplitMix64, shape: &[usize]) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(n, 1.0))
}

fn plan_rule() {
    // default: off, one launch (the A/B found no win)
    set(None, None, None);
    assert_eq!(gemm_split_rows(2048, 4096, 12288, false), vec![0, 2048], "off by default");
    assert_eq!(gemm_split_rows(1, 151936, 4096, true), vec![0, 151936], "off by default");
    // tail mode, default trigger 1100 us and tail 400 us
    set(Some("tail"), None, None);
    assert_eq!(gemm_split_rows(1059, 12288, 4096, false), vec![0, 640, 1059], "8B gate / up at ~1k tokens: tail of 4 tile rows");
    assert_eq!(gemm_split_rows(1059, 4096, 12288, false), vec![0, 640, 1059], "8B down at ~1k tokens");
    assert_eq!(gemm_split_rows(2048, 4096, 12288, false), vec![0, 1536, 2048], "8B down at a 2048-token chunk");
    assert_eq!(gemm_split_rows(2048, 6144, 4096, false), vec![0, 2048], "8B qkv at 2048 tokens (~1.02 ms, no stall) stays one launch");
    assert_eq!(gemm_split_rows(1024, 12288, 4096, false), vec![0, 1024], "estimate 1031 us: under the trigger");
    assert_eq!(gemm_split_rows(100, 1 << 20, 4096, false), vec![0, 100], "one tile row cannot be cut");
    assert_eq!(gemm_split_rows(1, 151936, 4096, true), vec![0, 121856, 151936], "8B lm_head GEMV: tail of 30 x 1024 rows");
    assert_eq!(gemm_split_rows(2, 151936, 4096, true), vec![0, 151936], "multi-row GEMV is never cut");
    assert_eq!(gemm_split_rows(1, 32768, 4096, true), vec![0, 32768], "a 433 us GEMV stays one launch");
    // the kill switch
    for off in ["0", "off"] {
        set(Some(off), None, None);
        assert_eq!(gemm_split_rows(2048, 4096, 12288, false), vec![0, 2048], "INTARSIA_HIP_GEMM_SPLIT={off}");
        assert_eq!(gemm_split_rows(1, 151936, 4096, true), vec![0, 151936], "INTARSIA_HIP_GEMM_SPLIT={off}");
    }
    // trigger and tail length
    set(Some("tail"), Some("2100"), None);
    assert_eq!(gemm_split_rows(2048, 4096, 12288, false), vec![0, 2048], "estimate 2062 us under a 2100 us trigger");
    set(Some("tail"), Some("2000"), Some("100"));
    assert_eq!(gemm_split_rows(2048, 4096, 12288, false), vec![0, 1920, 2048], "tail of one tile row");
    set(Some("tail"), None, Some("100000"));
    assert_eq!(gemm_split_rows(2048, 4096, 12288, false), vec![0, 128, 2048], "the first piece keeps at least one tile row");
    // even: pieces of at most ~900 us
    set(Some("even"), None, None);
    assert_eq!(gemm_split_rows(2048, 4096, 12288, false), vec![0, 640, 1280, 2048]);
    assert_eq!(gemm_split_rows(1059, 12288, 4096, false), vec![0, 512, 1059]);
    assert_eq!(gemm_split_rows(1, 151936, 4096, true), vec![0, 50176, 101376, 151936]);
    set(None, None, None);
}

#[test]
fn gemm_split_plan_and_bits() {
    plan_rule();
    let Some(hip) = device() else {
        set(None, None, None);
        return;
    };
    let mut rng = SplitMix64::new(1010);
    // (op, kernel, out dtype, M, N, K)
    let cases = [
        (Op::Matmul, "matmul_bf16", DType::Bf16, 300usize, 256usize, 128usize),
        (Op::Matmul, "matmul_bf16", DType::Bf16, 129, 64, 64),
        (Op::Matmul, "matmul_bf16", DType::Bf16, 1059, 512, 256),
        (Op::Matmul, "matmul_bf16", DType::Bf16, 2048, 384, 512),
        (Op::Matmul, "matmul_bf16", DType::Bf16, 777, 200, 96),
        (Op::Logits, "logits_bf16", DType::F32, 300, 1000, 128),
        (Op::Logits, "logits_bf16_gemv", DType::F32, 1, 5000, 256),
        (Op::Logits, "logits_bf16_gemv", DType::F32, 1, 20480, 512),
        (Op::Matmul, "matmul_bf16_gemv", DType::Bf16, 1, 9000, 384),
    ];
    // forced cuts (trigger 0): tail of one granule, tail of most of the launch, even
    let modes: [(&str, &str); 4] = [("tail", "0.000001"), ("tail", "1000000"), ("tail", "3"), ("even", "400")];
    let mut n_cases = 0usize;
    for &(op, kernel, odt, m, n, k) in &cases {
        let gemv = kernel.ends_with("_gemv");
        let hc = HostCall {
            op,
            inputs: vec![rand_t(&mut rng, &[m, k]), rand_t(&mut rng, &[n, k])],
            outputs: vec![OutSpec::new(odt, &[m, n])],
            attrs: OpAttrs::default(),
        };
        set(Some("0"), None, None);
        let one = run_host(&*hip, kernel, &hc).unwrap_or_else(|e| panic!("{kernel} M={m} N={n} K={k}: {e}"));
        assert!(one[0].to_f32().iter().any(|v| *v != 0.0), "{kernel} M={m} N={n} K={k}: all zero");
        for (mode, tail) in modes {
            // even mode with a tiny budget is not reachable from the env; force the trigger and use the tail knob only
            set(Some(mode), Some("0"), Some(tail));
            let rows = gemm_split_rows(m as i64, n as i64, k as i64, gemv);
            let cut = rows.len() > 2;
            let got = run_host(&*hip, kernel, &hc).unwrap();
            assert!(
                one[0].data == got[0].data,
                "{kernel} M={m} N={n} K={k} mode={mode} tail={tail} cuts={rows:?}: the pieces changed the output"
            );
            if cut {
                n_cases += 1;
            }
        }
    }
    set(None, None, None);
    assert!(n_cases >= 20, "too few cut launches compared ({n_cases})");
    println!("gemm split: {n_cases} cut launches bitwise");
}
