//! Kernel table and device dispatch of the P12 (lossless 12-bit bf16, `intarsia_core::quant::p12`) GEMVs on CUDA (lane
//! ink-cuda-p12): `MatmulP12` and the P12 form of `InkInProj` (u8 codes as the weight). Kernels in kernels/cuda/inkling2_p12.cu,
//! ABI kernels/cuda/ik_cuda_abi_inkp12.h. Each is its bf16 twin's arithmetic on the decoded weight, bit for bit:
//!
//! * `matmul_p12_gemv` / `matmul_p12_logits_gemv` (bf16 / f32 out) == `matmul_bf16_gemv` / `logits_bf16_gemv`, ordinary calls of
//!   1..=4 rows only: where the bf16 call runs that GEMV. An ordinary call of 5..=16 rows finds no P12 kernel, so the model keeps
//!   the bf16 weight there (cuBLAS), and P12 on stays bitwise P12 off.
//! * `_rows`: the `rows_exact` registrations, == `matmul_bf16_gemv_rows` / `logits_bf16_gemv_rows`.
//! * `ink_in_proj_p12_mma[_rows]` == `ink_in_proj_bf16_mma[_rows]` (inkmix.rs), 1..=16 rows.
//!
//! The P12 copies are made only under `INTARSIA_INK_PACK12=1` on CUDA (`p12::decide`; default off here).

use crate::kernels::ARCH;
use intarsia_core::quant::p12::ROWS_MAX;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("matmul_p12_gemv", Op::MatmulP12),
    ("matmul_p12_gemv_rows", Op::MatmulP12),
    ("matmul_p12_logits_gemv", Op::MatmulP12),
    ("matmul_p12_logits_gemv_rows", Op::MatmulP12),
    ("ink_in_proj_p12_mma", Op::InkInProj),
    ("ink_in_proj_p12_mma_rows", Op::InkInProj),
];

const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// 1..=16 activation rows, K a multiple of 16 (the launchers' 8-element chunks in pairs).
fn rows_ok(s: &ShapeInfo) -> bool {
    (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0 && s.k <= 65536
}

/// The ordinary rows of `matmul_bf16_gemv` / `logits_bf16_gemv` (above them the bf16 call is cuBLAS).
fn gemv_ordinary(s: &ShapeInfo) -> bool {
    rows_ok(s) && s.m <= 4
}

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "cuda")
                .arch([ARCH])
                .acts([DType::Bf16])
                .weights([DType::U8])
                .exact(Exactness::Deterministic);
            match *name {
                "matmul_p12_gemv" => k.when(|s| gemv_ordinary(s) && s.out == Some(DType::Bf16)).tol(BF16_TOL).priority(10),
                "matmul_p12_gemv_rows" => {
                    k.when(|s| rows_ok(s) && s.out == Some(DType::Bf16)).tol(BF16_TOL).priority(-1).rows_exact()
                }
                // row m of an M-row call runs the one-row loop (as logits_bf16_gemv): row-invariant
                "matmul_p12_logits_gemv" => {
                    k.when(|s| gemv_ordinary(s) && s.out == Some(DType::F32)).tol(F32_TOL).priority(10).row_invariant()
                }
                "matmul_p12_logits_gemv_rows" => {
                    k.when(|s| rows_ok(s) && s.out == Some(DType::F32)).tol(F32_TOL).priority(-1).rows_exact()
                }
                "ink_in_proj_p12_mma" => k.when(rows_ok).tol(BF16_TOL).priority(10),
                "ink_in_proj_p12_mma_rows" => k.when(rows_ok).tol(BF16_TOL).priority(-1).rows_exact(),
                other => unreachable!("inkp12 kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_cuda)]
mod launch {
    use crate::device::CudaBackend;
    use crate::ffi::{status_name, IkTensor};
    use intarsia_core::backend::StreamId;
    use intarsia_core::{Error, Op, OpCall, Result};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type S = *mut c_void;

    extern "C" {
        fn ik_matmul_p12_gemv(x: T, codes: T, aux: T, out: TM, s: S) -> c_int;
        fn ik_matmul_p12_gemv_rows(x: T, codes: T, aux: T, out: TM, s: S) -> c_int;
        fn ik_matmul_p12_logits_gemv(x: T, codes: T, aux: T, out: TM, s: S) -> c_int;
        fn ik_matmul_p12_logits_gemv_rows(x: T, codes: T, aux: T, out: TM, s: S) -> c_int;
        fn ik_ink_in_proj_p12_mma(x: T, codes: T, aux: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_ink_in_proj_p12_mma_rows(x: T, codes: T, aux: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
    }

    impl CudaBackend {
        /// `Some(result)` when `kernel` is one of this module's launchers, `None` otherwise.
        pub(crate) fn launch_inkp12(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Option<Result<()>> {
            let (_, op) = super::TABLE.iter().find(|(n, _)| *n == kernel)?;
            Some(self.launch_inkp12_inner(kernel, *op, call, stream))
        }

        fn launch_inkp12_inner(&self, kernel: &str, op: Op, call: &OpCall, stream: StreamId) -> Result<()> {
            if op != call.op {
                return Err(Error::invalid(format!("kernel '{kernel}' implements {op}, not {}", call.op)));
            }
            if call.inputs.len() < 3 {
                return Err(Error::invalid(format!("{kernel}: needs x, codes, aux")));
            }
            self.bind()?;
            let s = self.stream(stream)?;
            let inp: Vec<IkTensor> = call.inputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let mut out: Vec<IkTensor> = call.outputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let i = |n: usize| &inp[n] as *const IkTensor;
            let o = out.as_mut_ptr();
            let on = |n: usize| o.wrapping_add(n);
            // SAFETY: tensors are bounds-checked views of live device buffers (held by `call`); the launchers validate dtype /
            // rank / shape and return an error status instead of faulting, and never synchronise or allocate.
            let rc = unsafe {
                match kernel {
                    "matmul_p12_gemv" => ik_matmul_p12_gemv(i(0), i(1), i(2), on(0), s),
                    "matmul_p12_gemv_rows" => ik_matmul_p12_gemv_rows(i(0), i(1), i(2), on(0), s),
                    "matmul_p12_logits_gemv" => ik_matmul_p12_logits_gemv(i(0), i(1), i(2), on(0), s),
                    "matmul_p12_logits_gemv_rows" => ik_matmul_p12_logits_gemv_rows(i(0), i(1), i(2), on(0), s),
                    "ink_in_proj_p12_mma" => ik_ink_in_proj_p12_mma(i(0), i(1), i(2), on(0), on(1), on(2), s),
                    "ink_in_proj_p12_mma_rows" => ik_ink_in_proj_p12_mma_rows(i(0), i(1), i(2), on(0), on(1), on(2), s),
                    other => return Err(Error::NoKernel(format!("cuda kernel '{other}'"))),
                }
            };
            if rc != 0 {
                return Err(Error::backend(format!(
                    "ik_{kernel} returned {} ({rc}); inputs {:?} shapes {:?}",
                    status_name(rc),
                    call.inputs.iter().map(|t| t.dtype).collect::<Vec<_>>(),
                    call.inputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
                )));
            }
            Ok(())
        }
    }
}
