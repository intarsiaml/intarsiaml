//! Kernel table and device dispatch of the Inkling-2-Mini routed experts: `MoeExperts` with the SiTU activation (NVFP4, `.situ()`) and `MoeExpertsMxfp8` (72) (lane ink-hip-moe; port map
//! docs/results/2026-11-25-inkling2-hip.md). Kernels in kernels/hip/ink_moe.hip, ABI kernels/hip/ik_hip_abi_inkmoe.h; contract
//! `intarsia_core::inkling2::ops`, admitted against the cpu reference (`ink_ref`) by `intarsia kernels test --backend hip --op <op>`.
//!
//! The skeleton: an empty table. A lane adds its `(launcher name, op)` rows to `TABLE`, the registration of each name to `all()`,
//! the `extern "C"` declaration and the dispatch arm to `launch`, the declaration to the ABI header, and nothing outside its own
//! files: `kernels::full_table()` / `all()`, the device dispatch (`device.rs`, the fall-through arm of `launch`) and the host
//! tests already chain this module. An op of `OPS` counts as ported once `TABLE` lists it (until then it stays in
//! `kernels::UNIMPLEMENTED`, which the coverage test reads together with `OPS`).

use intarsia_core::{KernelImpl, Op};

pub const TABLE: &[(&str, Op)] = &[];

/// The ops this module is the home of (a lane's kernels register only these).
pub const OPS: &[Op] = &[Op::MoeExpertsMxfp8];

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, _op)| -> KernelImpl {
            // KernelImpl::new(*_op, *name, "hip").arch([crate::kernels::ARCH]).acts([DType::Bf16]) ... per name, as the CUDA twin
            match *name {
                other => unreachable!("inkmoe kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_hip)]
pub(crate) mod launch {
    use crate::device::InkArgs;
    use std::ffi::c_int;

    extern "C" {}

    /// The launcher status of `kernel` when it is one of this module's, `None` otherwise. Called from `HipBackend::launch` with
    /// the call's tensors already converted (`a.i(n)` input n, `a.o(n)` output n, `a.sc` scratch, `a.pos` the device position or
    /// null, `a.tab*` the row table or null, `a.s` the stream, `a.call.attrs`); the caller checks the status, traces and reports.
    ///
    /// # Safety
    /// As the launchers of `ffi`: the tensors are bounds-checked views of live device buffers held by `a.call`.
    #[allow(clippy::match_single_binding)]
    pub(crate) unsafe fn launch(kernel: &str, a: &InkArgs) -> Option<c_int> {
        let _ = a;
        Some(match kernel {
            _ => return None,
        })
    }
}
