//! Kernel table and device dispatch of the Inkling-2 DSpark draft's ops that HIP had no kernel for (lane ink-hip-spec,
//! docs/results/2026-11-26-inkling2-hip-spec.md): `HcPre`, the draft ingest's four-tap mix (`model/dspark.rs`). Kernels in
//! kernels/hip/ink_spec.hip, ABI kernels/hip/ik_hip_abi_inkspec.h; contract `intarsia_core::dsv41::ops`, admitted against the cpu
//! reference by `intarsia kernels test --backend hip --op hc_pre`. The draft's other ops (Matmul, Logits, Rope, QkNorm, windowed
//! AttnDecode, SiluGlu, RmsNorm, Embedding, Add) and its device top-k (`Backend::topk_rows`, bf16 rows) are HIP's existing kernels.
//!
//! An op of `OPS` counts as ported once `TABLE` lists it (`kernels::unimplemented`).

use crate::kernels::ARCH;
use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

pub const TABLE: &[(&str, Op)] = &[("hc_pre_bf16", Op::HcPre)];

/// The ops this module is the home of.
pub const OPS: &[Op] = &[Op::HcPre];

const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| -> KernelImpl {
            let k = KernelImpl::new(*op, *name, "hip").arch([ARCH]).acts([DType::Bf16, DType::F32]).exact(Exactness::Deterministic);
            match *name {
                // CUDA's `hc_pre` registration (crates/intarsia-backend-cuda/src/dsv41.rs): S in 1..=8
                "hc_pre_bf16" => k.when(|s| s.n >= 1 && s.n <= 8).tol(BF16_TOL).tol_f32(F32_TOL),
                other => unreachable!("inkspec kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_hip)]
pub(crate) mod launch {
    use crate::device::InkArgs;
    use crate::ffi::IkTensor;
    use std::ffi::{c_int, c_void};
    type T = *const IkTensor;
    type TM = *mut IkTensor;
    extern "C" {
        fn ik_hc_pre_bf16(x: T, mix: T, y: TM, s: *mut c_void) -> c_int;
    }

    /// The launcher status when `kernel` is this module's, `None` otherwise. Called from `HipBackend::launch` like the other
    /// Inkling-2 lane modules' (`a.i(n)` input n, `a.o(n)` output n, `a.s` stream).
    ///
    /// # Safety
    /// As the launchers of `ffi`: the tensors are bounds-checked views of live device buffers held by `a.call`.
    pub(crate) unsafe fn launch(kernel: &str, a: &InkArgs) -> Option<c_int> {
        Some(match kernel {
            "hc_pre_bf16" => ik_hc_pre_bf16(a.i(0), a.i(1), a.o(0), a.s),
            _ => return None,
        })
    }
}
