//! Kernel table and device dispatch of `Op::MatmulExl3` (EXL3 trellis weights, kernels/cuda/exl3.cu). The table is
//! data and compiles on every host; the dispatch (`launch_exl3`, called from `device.rs::launch`) needs CUDA.
//!
//! `matmul_exl3_gemv` serves every row count: the launcher runs `M` rows as slices of at most 128 (the rotated input of
//! one slice at a time is the scratch), and every 16-row pass of a block decodes the weight tiles again, so it is a decode
//! kernel first. `matmul_exl3_gemm` is the prefill kernel: a block decodes each weight tile once for 64 or 128 rows and
//! runs mma against them, in slices of 2048 rows. Its output is bit for bit the GEMV's (same mma chains, same sum order),
//! so which of the two runs changes no row's bits; it is registered from [`GEMM_MIN_TOKENS`] rows up. Both read the
//! codebook on the device and take the width from the trellis shape, so one registration covers every width and
//! codebook the op defines.

use crate::kernels::ARCH;
use intarsia_core::quant::exl3::valid_k2_any;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("matmul_exl3_gemv", Op::MatmulExl3),
    ("matmul_exl3_gemm", Op::MatmulExl3),
    ("moe_experts_exl3", Op::MoeExperts),
    ("matmul_exl3_splitk", Op::MatmulExl3),
    ("matmul_exl3_cp", Op::MatmulExl3),
];

/// Rows `matmul_exl3_splitk` takes (one 16-row pass, kernels/cuda/exl3.cu).
pub const SPLITK_MAX_ROWS: usize = 16;

/// Splits the scratch of `matmul_exl3_splitk` has room for (`kMaxSplits`).
pub const SPLITK_MAX_SPLITS: usize = 16;

/// Scratch bytes of `matmul_exl3_splitk`: the rotated input fp16 `[M, K]` rounded up to 256 bytes, then the f32 partials
/// `[16, M, N]` (mirrors `ik_matmul_exl3_splitk_scratch_bytes`).
pub fn splitk_scratch_bytes(s: &ShapeInfo) -> usize {
    ((s.m * s.k * 2 + 255) & !255) + SPLITK_MAX_SPLITS * s.m * s.n * 4
}

/// Rows below which `moe_experts_exl3` runs its decode kernel in one slice (kernels/cuda/exl3_moe.cu, `kDecodeRows`).
pub const MOE_DECODE_ROWS: usize = 64;
/// Rows of one prompt slice of `moe_experts_exl3` (`kPromptRows`).
pub const MOE_PROMPT_ROWS: usize = 2048;
/// Most experts a `moe_experts_exl3` call (`kMaxE`).
pub const MOE_MAX_EXPERTS: usize = 1024;

/// K splits of the gate / up projection of `moe_experts_exl3` (`gateup_splits`): the most of 4, 2, 1 that divide its
/// `H / 16` k tiles among 4 warps.
pub fn moe_gateup_splits(h: usize) -> usize {
    [4, 2, 1].into_iter().find(|sk| (h / 16) % (4 * sk) == 0).unwrap_or(0)
}

/// Scratch bytes of `moe_experts_exl3` (kernels/cuda/exl3_moe.cu `layout`, region by region, each rounded up to 256): for a
/// slice of R rows (all T below `MOE_DECODE_ROWS`, else up to `MOE_PROMPT_ROWS`) and P = R K pairs: the grouping's ints
/// (3E + 1 + P + 2 (P/32 + min(E, P) + 1)), the rotated gate / up inputs (2 x fp16 [P, H]; the down partials reuse them),
/// the gate / up partials (f32 [2, SK, P, I], SK = 1 for prompt slices) and the activation (fp16 [P, I]).
pub fn moe_scratch_bytes(s: &ShapeInfo) -> usize {
    let r256 = |b: usize| (b + 255) & !255;
    let (t, h, i, e, k) = (s.m, s.k, s.n, s.experts, s.top_k);
    if t == 0 {
        return 0;
    }
    let decode = t < MOE_DECODE_ROWS;
    let r = if decode { t } else { t.min(MOE_PROMPT_ROWS) };
    let p = r * k;
    let wlb = p / 32 + e.min(p) + 1;
    let skz = if decode { moe_gateup_splits(h) } else { 1 };
    r256(4 * (3 * e + 1 + p + 2 * wlb)) + 2 * r256(p * h * 2) + r256(2 * skz * p * i * 4) + r256(p * i * 2)
}

/// `moe_experts_exl3`'s budget of bf16 / f16 outputs. Two fp16 roundings separate it from the f64 reference: the rotated
/// input `H (x * suh)` of gate / up and the rotated activation `H (act * suh_d)` of down (each up to 2^-11 relative per
/// element), on top of the output rounding. Set from the measured sweep (crates/intarsia-backend-cuda/tests/exl3_moe.rs).
const MOE_BF16_OUT_TOL: Tol = Tol::new(1.5e-2, 3.5e-3);
/// ... of f32 outputs (the two fp16 roundings alone).
const MOE_F32_OUT_TOL: Tol = Tol::new(2e-3, 1e-3);

fn moe_shape_ok(s: &ShapeInfo) -> bool {
    s.m >= 1
        && s.k % 128 == 0
        && s.n % 128 == 0
        && s.n > 0
        && moe_gateup_splits(s.k) > 0
        && (1..=MOE_MAX_EXPERTS).contains(&s.experts)
        && (1..=32).contains(&s.top_k)
        && matches!(s.out, Some(DType::Bf16 | DType::F16 | DType::F32))
}

/// Rows of one slice of a GEMV launch (kernels/cuda/exl3.cu, `kSliceRows`).
pub const SLICE_ROWS: usize = 128;

/// Rows of one slice of a GEMM launch (`kGemmSliceRows`).
pub const GEMM_SLICE_ROWS: usize = 512;

/// Rows from which the GEMM is registered. The GEMV decodes the weights once per 16-row pass, so its time steps up at 17,
/// 33, 49 .. rows; the GEMM costs about the same from 1 to 64 rows (one 64-row block). Measured on spark1 (GB10, cold
/// weights, the four Qwen3-8B shapes, 4-bit 3inst, docs/results/2026-10-21-m2-exl3/raw/cuda-gemm-prefill.log): at 16 rows the
/// GEMV is 2x faster (42.6 vs 84.9 us at 4096x4096); at 32 rows the GEMM wins on three shapes (62.9 vs 68.4, 50.2 vs 63.0,
/// 174.3 vs 205.9 us) and loses 8% on 12288x4096 (179.1 vs 165.7); at 48 rows it takes 0.51-0.57x of the GEMV's time on every
/// shape. So 33.
pub const GEMM_MIN_TOKENS: usize = 33;

/// Scratch bytes of the GEMV: the rotated input of one slice, fp16 `[min(M, 128), K]`, rounded up to 256 bytes.
pub fn scratch_bytes(s: &ShapeInfo) -> usize {
    (s.m.min(SLICE_ROWS) * s.k * 2 + 255) & !255
}

/// Scratch bytes of the GEMM: the rotated input of one slice, fp16 `[min(M, 512), K]`, rounded up to 256 bytes.
pub fn gemm_scratch_bytes(s: &ShapeInfo) -> usize {
    (s.m.min(GEMM_SLICE_ROWS) * s.k * 2 + 255) & !255
}

/// Budget of the bf16-rounded outputs. The kernel rounds the rotated input `H (x * suh)` to fp16 before the tensor-core
/// multiply (the cpu reference keeps it in f64), a relative error of up to 2^-11 per element that reaches the output
/// almost undamped, and a bf16 output then flips an ulp wherever that moves a value across a rounding boundary. Measured
/// on spark1 over the full `matmul_exl3` sweep (242 cases, bf16 and f32 outputs, `intarsia kernels test --backend cuda
/// --op matmul_exl3`, docs/results/2026-10-21-m2-exl3/raw/cuda-admission-full.log): worst bf16 rel_l2 1.68e-3 (M=1 N=256
/// K=384, a small case: the Qwen3-8B shapes sit at 0.86 to 1.09e-3), worst max_abs 6.3e-3 of max(1, max|ref|). rel_l2 gets
/// ~2x that; max_abs stays at the shared 1.5e-2 (2.4x). The shared quantised-matmul budget (rel_l2 1e-3) refuses the kernel.
/// tests/exl3.rs shows the gap is the input rounding: against a reference that rounds `H (x * suh)` to fp16 too, f32
/// outputs agree to 1e-6 .. 3e-5 (2e-4 against the f64 reference).
const BF16_OUT_TOL: Tol = Tol::new(1.5e-2, 3.5e-3);

/// Budget of the f32 outputs: the same input rounding with no output rounding on top. Measured over the same sweep: worst
/// rel_l2 2.43e-4 (M=2 N=128 K=256), worst max_abs 3.2e-4; rel_l2 gets 2x, max_abs stays at the shared 1e-3.
const F32_OUT_TOL: Tol = Tol::new(1e-3, 5e-4);

/// Shapes both kernels take: K and N multiples of 128, an EXL3 width, a float output.
fn shape_ok(s: &ShapeInfo) -> bool {
    s.k % 128 == 0 && s.n % 128 == 0 && valid_k2_any(s.head_dim) && matches!(s.out, Some(DType::Bf16 | DType::F16 | DType::F32))
}

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "cuda").arch([ARCH]);
            match *name {
                "matmul_exl3_gemv" => k
                    .acts([DType::Bf16, DType::F16, DType::F32])
                    .weights([DType::I16])
                    .when(|s| s.m >= 1 && shape_ok(s))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_OUT_TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(scratch_bytes)
                    .priority(10),
                "matmul_exl3_gemm" => k
                    .acts([DType::Bf16, DType::F16, DType::F32])
                    .weights([DType::I16])
                    .when(|s| s.m >= GEMM_MIN_TOKENS && shape_ok(s))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_OUT_TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(gemm_scratch_bytes)
                    .priority(20),
                // split-K GEMV for small-N decode calls (lane dsv41/tp2). Priority below the GEMV, which accepts every shape this
                // does, so the plain selection never picks it and no model's plan changes; DeepSeek-V4.1's runtime selects it by
                // name for its small-grid one-row calls (`dsv41::runtime`, INTARSIA_DSV41_EXL3_SPLITK). Not bitwise the GEMV.
                "matmul_exl3_splitk" => k
                    .acts([DType::Bf16, DType::F16, DType::F32])
                    .weights([DType::I16])
                    .when(|s| (1..=SPLITK_MAX_ROWS).contains(&s.m) && shape_ok(s))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_OUT_TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(splitk_scratch_bytes)
                    .priority(-10),
                // split-K with the MoE kernel's cp.async loader; selected by name only, like matmul_exl3_splitk
                "matmul_exl3_cp" => k
                    .acts([DType::Bf16, DType::F16, DType::F32])
                    .weights([DType::I16])
                    .when(|s| (1..=SPLITK_MAX_ROWS).contains(&s.m) && shape_ok(s))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_OUT_TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(splitk_scratch_bytes)
                    .priority(-11),
                // EXL3 routed experts (Op::MoeExperts with an i16 trellis arena, quant::exl3_moe): decode windows below 64 rows,
                // prompt slices of 2048 rows above; every width and codebook (device data, per matrix)
                "moe_experts_exl3" => k
                    .acts([DType::Bf16, DType::F16, DType::F32])
                    .weights([DType::I16])
                    .when(moe_shape_ok)
                    .exact(Exactness::Deterministic)
                    .tol(MOE_BF16_OUT_TOL)
                    .tol_f32(MOE_F32_OUT_TOL)
                    .scratch(moe_scratch_bytes)
                    .priority(10),
                other => unreachable!("exl3 kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_cuda)]
pub mod launch {
    use crate::device::CudaBackend;
    use crate::ffi::{status_name, IkScratch, IkTensor};
    use intarsia_core::backend::StreamId;
    use intarsia_core::{Error, OpCall, Result};
    use std::ffi::{c_int, c_void};

    extern "C" {
        fn ik_matmul_exl3_gemv(
            x: *const IkTensor,
            trellis: *const IkTensor,
            suh: *const IkTensor,
            svh: *const IkTensor,
            cb: *const IkTensor,
            out: *mut IkTensor,
            scratch: *const IkScratch,
            s: *mut c_void,
        ) -> c_int;
        fn ik_matmul_exl3_gemm(
            x: *const IkTensor,
            trellis: *const IkTensor,
            suh: *const IkTensor,
            svh: *const IkTensor,
            cb: *const IkTensor,
            out: *mut IkTensor,
            scratch: *const IkScratch,
            s: *mut c_void,
        ) -> c_int;
        fn ik_matmul_exl3_splitk(
            x: *const IkTensor,
            trellis: *const IkTensor,
            suh: *const IkTensor,
            svh: *const IkTensor,
            cb: *const IkTensor,
            out: *mut IkTensor,
            scratch: *const IkScratch,
            s: *mut c_void,
        ) -> c_int;
        fn ik_matmul_exl3_cp(
            x: *const IkTensor,
            trellis: *const IkTensor,
            suh: *const IkTensor,
            svh: *const IkTensor,
            cb: *const IkTensor,
            out: *mut IkTensor,
            scratch: *const IkScratch,
            s: *mut c_void,
        ) -> c_int;
        fn ik_matmul_exl3_splitk_scratch_bytes(m: i64, k: i64, n: i64) -> i64;
        fn ik_moe_experts_exl3(
            x: *const IkTensor,
            ids: *const IkTensor,
            wts: *const IkTensor,
            arena: *const IkTensor,
            scales: *const IkTensor,
            meta: *const IkTensor,
            cb: *const IkTensor,
            limit: *const IkTensor,
            out: *mut IkTensor,
            scratch: *const IkScratch,
            s: *mut c_void,
        ) -> c_int;
        fn ik_moe_experts_exl3_scratch_bytes(t: i64, h: i64, i: i64, e: i64, k: i64) -> i64;
    }

    /// `ik_matmul_exl3_splitk_scratch_bytes`, for the test that `splitk_scratch_bytes` mirrors it.
    pub fn splitk_scratch_bytes_device(m: usize, k: usize, n: usize) -> usize {
        // SAFETY: pure host arithmetic
        unsafe { ik_matmul_exl3_splitk_scratch_bytes(m as i64, k as i64, n as i64) as usize }
    }

    /// The device layout's scratch size (kernels/cuda/exl3_moe.cu), for the test that `moe_scratch_bytes` mirrors it.
    pub fn moe_scratch_bytes_device(t: usize, h: usize, i: usize, e: usize, k: usize) -> usize {
        // SAFETY: pure host arithmetic
        unsafe { ik_moe_experts_exl3_scratch_bytes(t as i64, h as i64, i as i64, e as i64, k as i64) as usize }
    }

    impl CudaBackend {
        /// `Some(result)` when `kernel` is an EXL3 launcher, `None` otherwise.
        pub(crate) fn launch_exl3(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Option<Result<()>> {
            let (_, op) = super::TABLE.iter().find(|(n, _)| *n == kernel)?;
            Some(self.launch_exl3_inner(kernel, *op, call, stream))
        }

        fn launch_exl3_inner(&self, kernel: &str, op: intarsia_core::Op, call: &OpCall, stream: StreamId) -> Result<()> {
            if op != call.op {
                return Err(Error::invalid(format!("kernel '{kernel}' implements {op}, not {}", call.op)));
            }
            self.bind()?;
            let s = self.stream(stream)?;
            let inp: Vec<IkTensor> = call.inputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let mut out: Vec<IkTensor> = call.outputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let scratch = match &call.scratch {
                Some(t) => {
                    let t = Self::tensor(t)?;
                    Some(IkScratch { data: t.data, bytes: t.shape[0] })
                }
                None => None,
            };
            let i = |n: usize| &inp[n] as *const IkTensor;
            // SAFETY: tensors are bounds-checked views of live device buffers (held by `call`); the launcher validates
            // dtype/rank/stride/alignment/scratch and returns an error status instead of faulting, and never
            // synchronises or allocates.
            let sp = scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch);
            if kernel == "moe_experts_exl3" {
                if inp.len() != 8 || out.len() != 1 {
                    return Err(Error::invalid(format!("ik_{kernel}: 8 inputs and 1 output, got {} and {}", inp.len(), out.len())));
                }
                // SAFETY: as above
                let rc = unsafe { ik_moe_experts_exl3(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), out.as_mut_ptr(), sp, s) };
                return if rc != 0 { Err(Self::exl3_err(kernel, rc, call)) } else { Ok(()) };
            }
            let f = match kernel {
                "matmul_exl3_gemm" => ik_matmul_exl3_gemm,
                "matmul_exl3_splitk" => ik_matmul_exl3_splitk,
                "matmul_exl3_cp" => ik_matmul_exl3_cp,
                _ => ik_matmul_exl3_gemv,
            };
            let rc = unsafe {
                f(
                    i(0),
                    i(1),
                    i(2),
                    i(3),
                    i(4),
                    out.as_mut_ptr(),
                    scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch),
                    s,
                )
            };
            if rc != 0 {
                return Err(Self::exl3_err(kernel, rc, call));
            }
            Ok(())
        }

        fn exl3_err(kernel: &str, rc: c_int, call: &OpCall) -> Error {
            let dts: Vec<_> = call.inputs.iter().map(|t| t.dtype).collect();
            Error::backend(format!(
                "ik_{kernel} returned {} ({rc}); inputs {dts:?} shapes {:?}",
                status_name(rc),
                call.inputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
            ))
        }
    }
}
