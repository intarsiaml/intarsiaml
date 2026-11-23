//! Kernel table and device dispatch of the screened greedy head on CUDA (lane ink-cuda-attn, brief 54; contract
//! `intarsia_core::screen`, the Metal twins in intarsia-backend-metal's screen.rs):
//!
//! * `head_screen_f32` (HeadScreen) and `screen_argmax_f32` (ScreenArgmax): kernels/cuda/ink_screen.cu, the contract's exact
//!   f32 operations, bitwise the cpu reference.
//! * `logits_gather_p12` (LogitsGather over a P12 head): the serving P12 head kernel itself (`p12_gemv_kernel<1, float>`,
//!   which `matmul_p12_logits_gemv[_rows]` launches at one row) in its index mode, kernels/cuda/inkling2_p12.cu; so a gathered
//!   logit is the head's bit for bit. One activation row (the screen serves 1-row heads).
//!
//! ABI kernels/cuda/ik_cuda_abi_inkp12.h. Every kernel computes each activation row on its own: registered rows-exact (the
//! model runs the screen calls with `exact_rows` set).

use crate::kernels::ARCH;
use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

pub const TABLE: &[(&str, Op)] =
    &[("head_screen_f32", Op::HeadScreen), ("logits_gather_p12", Op::LogitsGather), ("screen_argmax_f32", Op::ScreenArgmax)];

/// f32 outputs of the gather: accumulation-order noise against the cpu's f64 reference.
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "cuda").arch([ARCH]);
            match *name {
                "head_screen_f32" => k.acts([DType::Bf16]).exact(Exactness::Exact).tol(Tol::EXACT).rows_exact(),
                // the P12 head kernel's form: K % 16 == 0, K <= 65536 (inkp12.rs rows_ok), one row
                "logits_gather_p12" => k
                    .acts([DType::Bf16])
                    .weights([DType::U8])
                    .when(|s| s.m == 1 && s.k % 16 == 0 && s.k <= 65536)
                    .exact(Exactness::Deterministic)
                    .tol(F32_TOL)
                    .rows_exact(),
                "screen_argmax_f32" => k.acts([DType::F32]).exact(Exactness::Exact).tol(Tol::EXACT).rows_exact(),
                other => unreachable!("inkscreen kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_cuda)]
mod launch {
    use crate::device::CudaBackend;
    use crate::ffi::{status_name, IkTensor};
    use intarsia_core::backend::StreamId;
    use intarsia_core::{Error, OpCall, Result};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type S = *mut c_void;

    extern "C" {
        fn ik_head_screen_f32(a: T, h: T, beta: T, cand: TM, cnt: TM, s: S) -> c_int;
        fn ik_logits_gather_p12(x: T, codes: T, cand: T, cnt: T, aux: T, y: TM, s: S) -> c_int;
        fn ik_screen_argmax_f32(cand: T, cnt: T, y: T, tok: TM, s: S) -> c_int;
    }

    impl CudaBackend {
        /// `Some(result)` when `kernel` is one of this module's launchers, `None` otherwise.
        pub(crate) fn launch_inkscreen(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Option<Result<()>> {
            let (_, op) = super::TABLE.iter().find(|(n, _)| *n == kernel)?;
            if *op != call.op {
                return Some(Err(Error::invalid(format!("kernel '{kernel}' implements {op}, not {}", call.op))));
            }
            Some(self.launch_inkscreen_inner(kernel, call, stream))
        }

        fn launch_inkscreen_inner(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
            let need = match kernel {
                "logits_gather_p12" => (5, 1),
                "head_screen_f32" => (3, 2),
                _ => (3, 1),
            };
            if call.inputs.len() != need.0 || call.outputs.len() != need.1 {
                return Err(Error::invalid(format!("{kernel}: needs {} inputs and {} outputs", need.0, need.1)));
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
                    "head_screen_f32" => ik_head_screen_f32(i(0), i(1), i(2), on(0), on(1), s),
                    // inputs [h, codes, cand, cnt, aux]
                    "logits_gather_p12" => ik_logits_gather_p12(i(0), i(1), i(2), i(3), i(4), on(0), s),
                    "screen_argmax_f32" => ik_screen_argmax_f32(i(0), i(1), i(2), on(0), s),
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
