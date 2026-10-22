//! Kernel table and device dispatch of `Op::MatmulExl3` (EXL3 trellis weights, kernels/cuda/exl3.cu). The table is
//! data and compiles on every host; the dispatch (`launch_exl3`, called from `device.rs::launch`) needs CUDA.
//!
//! `matmul_exl3_gemv` serves every row count: the launcher runs `M` rows as slices of at most 128 (the rotated input of
//! one slice at a time is the scratch), and every 16-row pass of a block decodes the weight tiles again, so it is a decode
//! kernel first. `matmul_exl3_gemm` is the prefill kernel: a block decodes each weight tile once for 64 or 128 rows and
//! runs mma against them, in slices of 512 rows. Its output is bit for bit the GEMV's (same mma chains, same sum order),
//! so which of the two runs changes no row's bits; it is registered from [`GEMM_MIN_TOKENS`] rows up. Both read the
//! codebook on the device and take the width from the trellis shape, so one registration covers every width and
//! codebook the op defines.

use crate::kernels::ARCH;
use intarsia_core::quant::exl3::valid_k2_any;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[("matmul_exl3_gemv", Op::MatmulExl3), ("matmul_exl3_gemm", Op::MatmulExl3)];

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
                other => unreachable!("exl3 kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_cuda)]
mod launch {
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
            let f = if kernel == "matmul_exl3_gemm" { ik_matmul_exl3_gemm } else { ik_matmul_exl3_gemv };
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
                let dts: Vec<_> = call.inputs.iter().map(|t| t.dtype).collect();
                return Err(Error::backend(format!(
                    "ik_{kernel} returned {} ({rc}); inputs {dts:?} shapes {:?}",
                    status_name(rc),
                    call.inputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
                )));
            }
            Ok(())
        }
    }
}
