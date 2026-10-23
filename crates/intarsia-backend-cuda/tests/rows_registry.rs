//! Selection of the CUDA kernels through `Registry::select` itself, without a GPU (the kernel table is data).
//!
//! 1. Ordinary calls (`exact_rows` false) pick EXACTLY what they picked before the `rows_exact` kernels were listed
//!    (hybrid continuous batching, stage R0). MTP verify rows, prefill tail chunks and the solo decode step all depend on it:
//!    `ORDINARY_PINS` was generated on the tree without any `rows_exact` kernel (6167babd) and pasted here, not derived.
//! 2. Row-exact calls (`exact_rows`) of the five ops that have row-exact kernels (Matmul, Logits, MatmulFp8, MatmulNvfp4,
//!    MoeExperts) take only those, M = 1..=16: M = 1 is included (decision: an `exact_rows` one-row call must stay valid, e.g.
//!    a `forward_batch` step with one active row or a verify pass without drafts; Metal starts its rows kernels at 2 and fails
//!    an M = 1 call), and M > 16 fails instead of landing on a kernel whose bits depend on the row count.
//! 3. Ops without a row-exact kernel keep their ordinary choice under `exact_rows` (row-parallel ops; the fused ops have none
//!    by policy: at B >= 2 the model runs the unfused chain).
//!
//! The default environment is what is pinned (the INTARSIA_CUDA_* opt-ins change priorities and are read once per process).

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::exec::{HostCall, OutSpec};
use intarsia_core::quant::cases::{
    combine_case, experts_case, gated_rmsnorm_case, matmul_fp8_case, matmul_fp8_pair_case, matmul_nvfp4_case,
    matmul_nvfp4_glu_case, matmul_pair_case, topk_case,
};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, DeviceInfo, HostTensor, Op, OpAttrs, OpCall, Registry};

fn registry() -> Registry {
    let info = DeviceInfo {
        name: "pin".into(),
        backend: "cuda".into(),
        arch: intarsia_backend_cuda::kernels::ARCH.into(),
        total_mem: 0,
        free_mem: 0,
        unified: true,
        bandwidth_gbs: None,
    };
    Registry::new("cuda", &info, intarsia_backend_cuda::kernels::all())
}

fn put(cpu: &CpuBackend, t: &HostTensor) -> TensorRef {
    let buf = cpu.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    cpu.zero(&buf, 0).unwrap();
    if t.nbytes() > 0 {
        cpu.upload(&buf, 0, &t.data, 0).unwrap();
    }
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn op_call(cpu: &CpuBackend, hc: &HostCall) -> OpCall {
    let ins = hc.inputs.iter().map(|t| put(cpu, t)).collect();
    let outs = hc.outputs.iter().map(|o| put(cpu, &HostTensor::zeros(o.dtype, &o.shape))).collect();
    OpCall::new(hc.op, ins, outs, hc.attrs.clone())
}

fn bf16_matmul(rng: &mut SplitMix64, op: Op, m: usize, n: usize, k: usize, odt: DType) -> HostCall {
    let x = HostTensor::from_f32_as(DType::Bf16, &[m, k], &rng.normal_vec(m * k, 1.0));
    let w = HostTensor::from_f32_as(DType::Bf16, &[n, k], &rng.normal_vec(n * k, 1.0));
    HostCall { op, inputs: vec![x, w], outputs: vec![OutSpec::new(odt, &[m, n])], attrs: OpAttrs::default() }
}

/// (label, builder of the call at `m` rows). Small N / experts: the predicates look at M, K, the experts' H / I multiples and
/// the dtypes, which equal the 35B's and 8B's.
type Builder = Box<dyn Fn(&mut SplitMix64, usize) -> HostCall>;
fn cases() -> Vec<(&'static str, Builder)> {
    vec![
        ("matmul bf16 k=2048 -> bf16", Box::new(|r, m| bf16_matmul(r, Op::Matmul, m, 128, 2048, DType::Bf16))),
        ("matmul bf16 k=4104 -> bf16", Box::new(|r, m| bf16_matmul(r, Op::Matmul, m, 64, 4104, DType::Bf16))),
        ("logits bf16 k=4096 -> f32", Box::new(|r, m| bf16_matmul(r, Op::Logits, m, 128, 4096, DType::F32))),
        ("matmul_fp8 w8a8 k=2048 -> bf16", Box::new(|r, m| matmul_fp8_case(r, m, 128, 2048, true, DType::Bf16, DType::Bf16))),
        ("matmul_fp8 w8a8 k=4096 -> bf16", Box::new(|r, m| matmul_fp8_case(r, m, 128, 4096, true, DType::Bf16, DType::Bf16))),
        ("matmul_fp8 w8a16 k=2048 -> f32", Box::new(|r, m| matmul_fp8_case(r, m, 128, 2048, false, DType::Bf16, DType::F32))),
        ("matmul_nvfp4 k=2048 -> bf16", Box::new(|r, m| matmul_nvfp4_case(r, m, 128, 2048, DType::Bf16, DType::Bf16))),
        ("matmul_nvfp4 k=2048 -> f32 (head)", Box::new(|r, m| matmul_nvfp4_case(r, m, 128, 2048, DType::Bf16, DType::F32))),
        ("moe_experts silu top8", Box::new(|r, m| experts_case(r, m, 16, 8, 256, 128, DType::Bf16))),
        (
            "moe_experts gelu top8",
            Box::new(|r, m| {
                let mut hc = experts_case(r, m, 16, 8, 256, 128, DType::Bf16);
                hc.attrs.gelu = true;
                hc
            }),
        ),
        // ops that have no row-exact kernel: the choice must be the same under exact_rows
        ("moe_topk e=16 k=8", Box::new(|r, m| topk_case(r, m, 16, 8, false, DType::Bf16, true))),
        ("moe_combine h=256", Box::new(|r, m| combine_case(r, m, 256, DType::Bf16))),
        ("matmul_pair bf16 k=2048", Box::new(|r, m| matmul_pair_case(r, m, 32, 32, 2048, DType::Bf16))),
        (
            "matmul_fp8_pair k=2048",
            Box::new(|r, m| matmul_fp8_pair_case(r, m, 128, 128, 2048, true, true, DType::Bf16, DType::Bf16)),
        ),
        ("matmul_nvfp4_glu k=2048", Box::new(|r, m| matmul_nvfp4_glu_case(r, m, 128, 2048, DType::Bf16, DType::Bf16))),
        ("gated_rmsnorm r=4", Box::new(|r, m| gated_rmsnorm_case(r, m.max(1) * 4, 128, DType::Bf16))),
    ]
}

const MS: [usize; 28] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 40, 64, 255, 256, 257, 767, 768, 1000, 1024, 2048, 4096,
];

/// Run-length encoded picks of `exact_rows` false / true over `MS`: "1..=4 name; 5..=8 other" (or `-` for NoKernel).
fn table(exact_rows: bool) -> String {
    let reg = registry();
    let cpu = CpuBackend::new();
    let mut out = String::new();
    for (label, build) in cases() {
        let mut runs: Vec<(usize, usize, String)> = Vec::new();
        for m in MS {
            let mut rng = SplitMix64::new(7);
            let hc = build(&mut rng, m);
            let mut call = op_call(&cpu, &hc);
            call.exact_rows = exact_rows;
            let name = reg.select(&call).map(|k| k.name.clone()).unwrap_or_else(|_| "-".to_string());
            match runs.last_mut() {
                Some(l) if l.2 == name => l.1 = m,
                _ => runs.push((m, m, name)),
            }
        }
        out.push_str(&format!("{label}:"));
        for (a, b, n) in runs {
            out.push_str(&format!(" {a}..={b} {n};"));
        }
        out.push('\n');
    }
    out
}

fn default_env() -> bool {
    !std::env::vars().any(|(k, _)| k.starts_with("INTARSIA_CUDA_"))
}

/// Generated on 6167babd (no `rows_exact` kernel in the CUDA table) with `print_tables`, and pasted.
const ORDINARY_PINS: &str = "matmul bf16 k=2048 -> bf16: 1..=4 matmul_bf16_gemv; 5..=4096 matmul_bf16;
matmul bf16 k=4104 -> bf16: 1..=4 matmul_bf16_gemv; 5..=4096 matmul_bf16;
logits bf16 k=4096 -> f32: 1..=4 logits_bf16_gemv; 5..=4096 logits_bf16;
matmul_fp8 w8a8 k=2048 -> bf16: 1..=4 matmul_fp8_gemv; 5..=32 matmul_fp8_mma; 33..=255 matmul_fp8_gemm; 256..=4096 matmul_fp8_gemm_reg;
matmul_fp8 w8a8 k=4096 -> bf16: 1..=4 matmul_fp8_gemv; 5..=32 matmul_fp8_mma; 33..=255 matmul_fp8_gemm; 256..=4096 matmul_fp8_gemm_reg;
matmul_fp8 w8a16 k=2048 -> f32: 1..=4 matmul_fp8_gemv; 5..=32 matmul_fp8_mma; 33..=255 matmul_fp8_gemm; 256..=4096 matmul_fp8_gemm_reg;
matmul_nvfp4 k=2048 -> bf16: 1..=8 matmul_nvfp4_gemv; 9..=32 matmul_nvfp4_mma; 33..=255 matmul_nvfp4_gemm; 256..=4096 matmul_nvfp4_gemm_reg;
matmul_nvfp4 k=2048 -> f32 (head): 1..=8 matmul_nvfp4_gemv; 9..=32 matmul_nvfp4_mma; 33..=255 matmul_nvfp4_gemm; 256..=4096 matmul_nvfp4_gemm_reg;
moe_experts silu top8: 1..=1 moe_experts_decode_v2_bf16; 2..=31 moe_experts_bf16; 32..=767 moe_experts_mma_bf16; 768..=4096 moe_experts_w4a16_bf16;
moe_experts gelu top8: 1..=1 moe_experts_decode_v2_gelu_bf16; 2..=31 moe_experts_gelu_bf16; 32..=767 moe_experts_mma_gelu_bf16; 768..=4096 moe_experts_w4a16_gelu_bf16;
moe_topk e=16 k=8: 1..=4096 moe_topk_bf16;
moe_combine h=256: 1..=4096 moe_combine_bf16;
matmul_pair bf16 k=2048: 1..=1 matmul_pair_bf16_gemv; 2..=4096 -;
matmul_fp8_pair k=2048: 1..=1 matmul_fp8_pair_gemv; 2..=255 -; 256..=4096 matmul_fp8_pair_gemm_reg;
matmul_nvfp4_glu k=2048: 1..=1 matmul_nvfp4_glu_gemv; 2..=255 -; 256..=4096 matmul_nvfp4_glu_gemm;
gated_rmsnorm r=4: 1..=4096 gated_rmsnorm_bf16;
";

/// `exact_rows` calls: the row-exact kernel for M = 1..=16 (M = 1 included), none above 16; ops without a row-exact kernel
/// are unchanged (moe_topk, moe_combine, gated_rmsnorm) and the fused ops have none by policy (m >= 2: no kernel, so the
/// model runs the unfused chain, which is made of the row-exact ops above).
const ROWS_PINS: &str = "matmul bf16 k=2048 -> bf16: 1..=16 matmul_bf16_gemv_rows; 17..=4096 -;
matmul bf16 k=4104 -> bf16: 1..=16 matmul_bf16_gemv_rows; 17..=4096 -;
logits bf16 k=4096 -> f32: 1..=16 logits_bf16_gemv_rows; 17..=4096 -;
matmul_fp8 w8a8 k=2048 -> bf16: 1..=16 matmul_fp8_gemv_rows; 17..=4096 -;
matmul_fp8 w8a8 k=4096 -> bf16: 1..=16 matmul_fp8_gemv_rows; 17..=4096 -;
matmul_fp8 w8a16 k=2048 -> f32: 1..=16 matmul_fp8_gemv_rows; 17..=4096 -;
matmul_nvfp4 k=2048 -> bf16: 1..=16 matmul_nvfp4_gemv_rows; 17..=4096 -;
matmul_nvfp4 k=2048 -> f32 (head): 1..=16 matmul_nvfp4_gemv_rows; 17..=4096 -;
moe_experts silu top8: 1..=16 moe_experts_decode_v2_rows_bf16; 17..=4096 -;
moe_experts gelu top8: 1..=16 moe_experts_decode_v2_gelu_rows_bf16; 17..=4096 -;
moe_topk e=16 k=8: 1..=4096 moe_topk_bf16;
moe_combine h=256: 1..=4096 moe_combine_bf16;
matmul_pair bf16 k=2048: 1..=1 matmul_pair_bf16_gemv; 2..=4096 -;
matmul_fp8_pair k=2048: 1..=1 matmul_fp8_pair_gemv; 2..=255 -; 256..=4096 matmul_fp8_pair_gemm_reg;
matmul_nvfp4_glu k=2048: 1..=1 matmul_nvfp4_glu_gemv; 2..=255 -; 256..=4096 matmul_nvfp4_glu_gemm;
gated_rmsnorm r=4: 1..=4096 gated_rmsnorm_bf16;
";

fn lines(s: &str) -> Vec<&str> {
    s.lines().collect()
}

#[test]
fn ordinary_calls_pick_what_they_picked_before_the_rows_kernels() {
    if !default_env() {
        return; // the INTARSIA_CUDA_* opt-ins change priorities; the default selection is what is pinned
    }
    assert_eq!(lines(&table(false)), lines(ORDINARY_PINS), "an ordinary call (exact_rows off) selects another kernel");
}

#[test]
fn exact_rows_calls_pick_the_rows_kernels_only() {
    if !default_env() {
        return;
    }
    assert_eq!(lines(&table(true)), lines(ROWS_PINS), "the exact_rows selection moved");
}

/// Prints both tables (cargo test -p intarsia-backend-cuda --test rows_registry -- --ignored --nocapture): how the pins above
/// are regenerated.
#[test]
#[ignore = "prints the pin tables"]
fn print_tables() {
    println!("ORDINARY\n{}", table(false));
    println!("EXACT_ROWS\n{}", table(true));
}
