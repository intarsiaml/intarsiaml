//! Kernel table and device dispatch of the Inkling-2-Mini router's top-k, `MoeRoute` (DSv4.1's op; the fused `InkRoute` (76) is optional and not built) (lane ink-hip, the lead; port map
//! docs/results/2026-11-25-inkling2-hip.md). Kernels in kernels/hip/ink_route.hip, ABI kernels/hip/ik_hip_abi_inkroute.h; contract
//! `intarsia_core::inkling2::ops`, admitted against the cpu reference (`ink_ref`) by `intarsia kernels test --backend hip --op <op>`.

use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

pub const TABLE: &[(&str, Op)] = &[("moe_route", Op::MoeRoute)];

const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// The ops this module is the home of (a lane's kernels register only these).
pub const OPS: &[Op] = &[Op::MoeRoute, Op::InkRoute];

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| -> KernelImpl {
            let k = KernelImpl::new(*op, *name, "hip").arch([crate::kernels::ARCH]).exact(Exactness::Deterministic);
            match *name {
                // the router's top-k (lane ink-hip, kernels/hip/ink_route.hip): as CUDA's `moe_route` (dsv41.rs)
                "moe_route" => k.acts([DType::F32]).when(|s| s.experts <= 1024 && (1..=32).contains(&s.top_k)).tol(F32_TOL),
                other => unreachable!("inkroute kernel {other} has no registration"),
            }
        })
        .collect()
}

/// Tests: `moe_route`'s fast tail (lane ink-hip-dec2): 1 = on, 0 = lane 0's loop, `-1` = the env (`INTARSIA_HIP_INK_DEC_FAST` bit 4)
/// or the default (on). Bitwise the same output (tests/ink_mix.rs `moe_route_fast_tail_bitwise`).
pub fn set_route_dec(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_moe_route_set_dec(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_moe_route_set_dec(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}

#[cfg(ik_hip)]
pub(crate) mod launch {
    use crate::device::InkArgs;
    use crate::ffi::IkTensor;
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;

    extern "C" {
        fn ik_moe_route(lg: T, bias: T, top_k: i64, scale: f32, norm: i32, temp: f32, sf: i32, w: TM, ids: TM, s: *mut c_void) -> c_int;
    }

    /// The launcher status of `kernel` when it is one of this module's, `None` otherwise. Called from `HipBackend::launch` with
    /// the call's tensors already converted (`a.i(n)` input n, `a.o(n)` output n, `a.sc` scratch, `a.pos` the device position or
    /// null, `a.tab*` the row table or null, `a.s` the stream, `a.call.attrs`); the caller checks the status, traces and reports.
    ///
    /// # Safety
    /// As the launchers of `ffi`: the tensors are bounds-checked views of live device buffers held by `a.call`.
    #[allow(clippy::match_single_binding)]
    pub(crate) unsafe fn launch(kernel: &str, a: &InkArgs) -> Option<c_int> {
        let at = &a.call.attrs;
        Some(match kernel {
            "moe_route" => {
                ik_moe_route(a.i(0), a.i(1), at.top_k as i64, at.scale, at.norm_topk as i32, at.temperature, at.dsv.score_func as i32, a.o(0), a.o(1), a.s)
            }
            _ => return None,
        })
    }
}
