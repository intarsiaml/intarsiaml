//! Kernel table and device dispatch of the Inkling-2-Mini experts (lane ink-cuda-mixmoe): `MoeExperts` with the SiTU
//! activation over NVFP4 (`.situ()` and only situ calls, so the 35B / Gemma NVFP4 calls keep their kernels), and
//! `MoeExpertsMxfp8` (silu / gelu / situ). Kernels in kernels/cuda/ink_moe.cu (ABI kernels/cuda/ik_cuda_abi_ink.h); the router
//! (`MoeRoute`) is DSv4.1's `moe_route` (dsv41.rs). Union-grouped decode: each expert's rows are read once per call for all
//! the pairs routed to it; a T-row call is bitwise T one-row calls, so the `_rows` twins are the same launchers.

use crate::kernels::ARCH;
use intarsia_core::quant::ops::moe_experts_scratch;
use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("moe_experts_situ_bf16", Op::MoeExperts),
    ("moe_experts_situ_bf16_rows", Op::MoeExperts),
    ("moe_experts_mxfp8_bf16", Op::MoeExpertsMxfp8),
    ("moe_experts_mxfp8_bf16_rows", Op::MoeExpertsMxfp8),
];

/// bf16 expert outputs: the quantised-MoE budget (quantmoe.rs `TOL`).
const MOE_TOL: Tol = Tol::new(1.5e-2, 1e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

pub const MAX_EXPERTS: usize = 1024;
pub const MAX_K: usize = 64;
/// Most rows of the `_rows` twins (`kernels::ROWS_MAX`).
pub const ROWS_MAX: usize = crate::kernels::ROWS_MAX;

pub fn all() -> Vec<KernelImpl> {
    let bf = [DType::Bf16];
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "cuda")
                .arch([ARCH])
                .acts(bf)
                .exact(Exactness::Deterministic)
                .tol(MOE_TOL)
                .tol_f32(F32_TOL)
                .scratch(moe_experts_scratch);
            match *name {
                "moe_experts_situ_bf16" => k
                    .weights([DType::U8])
                    .situ()
                    .when(|s| s.situ && s.experts <= MAX_EXPERTS && s.top_k <= MAX_K && s.k % 32 == 0 && s.n % 32 == 0),
                "moe_experts_situ_bf16_rows" => k
                    .weights([DType::U8])
                    .situ()
                    .when(|s| {
                        s.situ
                            && (1..=ROWS_MAX).contains(&s.m)
                            && s.experts <= MAX_EXPERTS
                            && s.top_k <= MAX_K
                            && s.k % 32 == 0
                            && s.n % 32 == 0
                    })
                    .priority(-1)
                    .rows_exact(),
                "moe_experts_mxfp8_bf16" => k
                    .weights([DType::Fp8E4m3])
                    .ext()
                    .situ()
                    .when(|s| s.experts <= MAX_EXPERTS && s.top_k <= MAX_K && s.k % 32 == 0 && s.n % 32 == 0),
                "moe_experts_mxfp8_bf16_rows" => k
                    .weights([DType::Fp8E4m3])
                    .ext()
                    .situ()
                    .when(|s| (1..=ROWS_MAX).contains(&s.m) && s.experts <= MAX_EXPERTS && s.top_k <= MAX_K && s.k % 32 == 0 && s.n % 32 == 0)
                    .priority(-1)
                    .rows_exact(),
                other => unreachable!("inkmoe kernel {other} has no registration"),
            }
        })
        .collect()
}

/// Benchmarks only: 1 launches only the gate+up kernel of an experts call, 2 only the down kernel (the output is garbage).
pub fn set_probe(v: i32) {
    #[cfg(ik_cuda)]
    {
        extern "C" {
            fn ik_ink_moe_probe(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_moe_probe(v) }
    }
    #[cfg(not(ik_cuda))]
    let _ = v;
}

#[cfg(ik_cuda)]
mod launch {
    use crate::device::CudaBackend;
    use crate::ffi::{status_name, IkScratch, IkTensor};
    use intarsia_core::backend::StreamId;
    use intarsia_core::{Error, Op, OpCall, Result};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type SC = *const IkScratch;
    type S = *mut c_void;

    extern "C" {
        fn ik_moe_experts_situ_bf16(x: T, ids: T, w: T, guw: T, gus: T, dnw: T, dns: T, gs: T, gc: f32, uc: f32, out: TM, sc: SC, s: S) -> c_int;
        fn ik_moe_experts_situ_bf16_rows(x: T, ids: T, w: T, guw: T, gus: T, dnw: T, dns: T, gs: T, gc: f32, uc: f32, out: TM, sc: SC, s: S) -> c_int;
        fn ik_moe_experts_mxfp8_bf16(x: T, ids: T, w: T, w13: T, s13: T, w2: T, s2: T, act: c_int, gc: f32, uc: f32, out: TM, sc: SC, s: S) -> c_int;
        fn ik_moe_experts_mxfp8_bf16_rows(x: T, ids: T, w: T, w13: T, s13: T, w2: T, s2: T, act: c_int, gc: f32, uc: f32, out: TM, sc: SC, s: S) -> c_int;
    }

    impl CudaBackend {
        /// `Some(result)` when `kernel` is one of this module's launchers, `None` otherwise.
        pub(crate) fn launch_inkmoe(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Option<Result<()>> {
            let (_, op) = super::TABLE.iter().find(|(n, _)| *n == kernel)?;
            Some(self.launch_inkmoe_inner(kernel, *op, call, stream))
        }

        fn launch_inkmoe_inner(&self, kernel: &str, op: Op, call: &OpCall, stream: StreamId) -> Result<()> {
            if op != call.op {
                return Err(Error::invalid(format!("kernel '{kernel}' implements {op}, not {}", call.op)));
            }
            let ink = &call.attrs.ink;
            if op == Op::MoeExperts && !ink.situ {
                return Err(Error::invalid(format!("cuda {kernel}: a call without ink.situ")));
            }
            self.bind()?;
            let s = self.stream(stream)?;
            let inp: Vec<IkTensor> = call.inputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let mut out: Vec<IkTensor> = call.outputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let scratch = match &call.scratch {
                Some(t) => {
                    let ik = Self::tensor(t)?;
                    Some(IkScratch { data: ik.data, bytes: t.nbytes() as i64 })
                }
                None => None,
            };
            let sc = scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch);
            let i = |n: usize| &inp[n] as *const IkTensor;
            let o = out.as_mut_ptr();
            let act: c_int = if ink.situ {
                2
            } else if call.attrs.gelu {
                1
            } else {
                0
            };
            let (gc, uc) = (ink.gate_cap, ink.up_cap);
            // SAFETY: tensors are bounds-checked views of live device buffers (held by `call`); the launchers validate dtype /
            // rank / shape / alignment / scratch and return an error status instead of faulting, and never synchronise or allocate.
            let rc = unsafe {
                match kernel {
                    "moe_experts_situ_bf16" => ik_moe_experts_situ_bf16(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), gc, uc, o, sc, s),
                    "moe_experts_situ_bf16_rows" => {
                        ik_moe_experts_situ_bf16_rows(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), gc, uc, o, sc, s)
                    }
                    "moe_experts_mxfp8_bf16" => ik_moe_experts_mxfp8_bf16(i(0), i(1), i(2), i(3), i(4), i(5), i(6), act, gc, uc, o, sc, s),
                    "moe_experts_mxfp8_bf16_rows" => {
                        ik_moe_experts_mxfp8_bf16_rows(i(0), i(1), i(2), i(3), i(4), i(5), i(6), act, gc, uc, o, sc, s)
                    }
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
