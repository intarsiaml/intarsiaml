//! Kernel table and device dispatch of the Inkling-2-Mini routed experts: `MoeExperts` with the SiTU activation (NVFP4, `.situ()`
//! and only situ calls, so the 35B's NVFP4 calls keep their kernels) and `MoeExpertsMxfp8` (72; silu / gelu / situ) (lane
//! ink-hip-moe; port map docs/results/2026-11-25-inkling2-hip.md, results docs/results/2026-11-25-inkling2-hip-moe.md). Kernels in
//! kernels/hip/ink_moe.hip, ABI kernels/hip/ik_hip_abi_inkmoe.h; contract `intarsia_core::inkling2::ops`, admitted against the cpu
//! reference (`ink_ref`) by `intarsia kernels test --backend hip --op <op>`.
//!
//! W4A16 only (never `.act_quant()`), the checkpoint's row-major weight layout only (`ink.moe_layout = 0`). Union-grouped decode,
//! a port of CUDA's SIMT kernels: each expert's rows are read once per call for all the pairs routed to it; on those kernels a
//! T-row call is bitwise T one-row calls at every T; the `_rows` twins always take them, and so do the ordinary names by default.
//! The ordinary names are not registered `rows_exact`: by default (`INTARSIA_HIP_INK_MOE_WMMA` unset or `1`; user decision
//! 2026-11-26, `=0` is the revert) they take a bf16 tensor-core prefill above 512 pairs and from 16 pairs per expert (other f32
//! accumulation order, not bitwise the decode kernels).

use intarsia_core::quant::ops::moe_experts_scratch;
use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("moe_experts_situ_bf16", Op::MoeExperts),
    ("moe_experts_situ_bf16_rows", Op::MoeExperts),
    ("moe_experts_mxfp8_bf16", Op::MoeExpertsMxfp8),
    ("moe_experts_mxfp8_bf16_rows", Op::MoeExpertsMxfp8),
];

/// The ops this module is the home of (a lane's kernels register only these).
pub const OPS: &[Op] = &[Op::MoeExpertsMxfp8];

/// bf16 expert outputs: the quantised-MoE budget (quantmoe.rs `TOL`), as CUDA's `inkmoe::MOE_TOL`.
const MOE_TOL: Tol = Tol::new(1.5e-2, 1e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);
/// NVFP4 situ experts: rel_l2 2e-3 (CUDA's `inkmoe::SITU_TOL`: the error is the bf16 rounding of the routing-weighted sum, worst
/// 9.5e-4 over 600 admission-built cases there).
const SITU_TOL: Tol = Tol::new(1.5e-2, 2e-3);

pub const MAX_EXPERTS: usize = 1024;
pub const MAX_K: usize = 64;
/// Most rows of the `_rows` twins (`kernels::ROWS_MAX`).
pub const ROWS_MAX: usize = crate::kernels::ROWS_MAX;

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| -> KernelImpl {
            let k = KernelImpl::new(*op, *name, "hip")
                .arch([crate::kernels::ARCH])
                .acts([DType::Bf16])
                .exact(Exactness::Deterministic)
                .tol(if name.starts_with("moe_experts_situ") { SITU_TOL } else { MOE_TOL })
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
                        s.situ && (1..=ROWS_MAX).contains(&s.m) && s.experts <= MAX_EXPERTS && s.top_k <= MAX_K && s.k % 32 == 0 && s.n % 32 == 0
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

/// Benchmarks only: 1 launches only the gate+up kernel of an experts call, 2 only the down kernel, 3 both with every thread
/// returning at once, 4 both returning after the grouping (the output is garbage); 0 both, in full.
pub fn set_probe(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_probe(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_moe_probe(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}

/// Tests / benchmarks: the load windows of gate_up and of down (lane steps requested ahead: 1, 3, 6 or 9; bitwise the same
/// output); `-1` = the env (`INTARSIA_HIP_INK_MOE_PRE`) or the format's default.
pub fn set_pre(gu: i32, dn: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_set_pre(gu: std::ffi::c_int, dn: std::ffi::c_int);
        }
        // SAFETY: sets two plain ints of the kernel library.
        unsafe { ik_ink_moe_set_pre(gu, dn) }
    }
    #[cfg(not(ik_hip))]
    let _ = (gu, dn);
}

/// Tests / benchmarks: the step form of the expert kernels: 1 lean (the default), 0 the first form (bitwise the same output),
/// `-1` = the env (`INTARSIA_HIP_INK_MOE_LEAN`).
pub fn set_lean(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_set_lean(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_moe_set_lean(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}

/// Tests / benchmarks: the smallest LDS size class of the expert kernels a call may take (0 = sized by the call, the default; 1;
/// 2 = the largest, the tables of the first form; 4 = sized by the call, with up to 32 pairs grouped inside the wave without
/// tables). Bitwise the same output.
pub fn set_sz(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_set_sz(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_moe_set_sz(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}

/// Tests / benchmarks: the one-wave grouping of the expert kernels' size class 0 (at most 16 pairs and 128 experts: decode, T <= 2 at top-8;
/// lane ink-hip-dec2): 1 = on, 0 = the shared-memory grouping of `group_pairs`, `-1` = the env (`INTARSIA_HIP_INK_DEC_FAST` bit 1)
/// or the default (on). Bitwise the same output.
pub fn set_dec(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_set_dec(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_moe_set_dec(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}

/// Tests / benchmarks: expert slices of the decode / rows expert kernels at 2..16 rows (lane ink-hip-verify): gate_up / down,
/// 1 = none (one block per row group, main's grid), >= 2 that many slices, 0 = the rule, `-1` = the env (`INTARSIA_HIP_INK_MOE_SPLIT`)
/// or the default. Bitwise the same output.
pub fn set_split(gu: i32, dn: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_set_split(gu: std::ffi::c_int, dn: std::ffi::c_int);
        }
        // SAFETY: sets two plain ints of the kernel library.
        unsafe { ik_ink_moe_set_split(gu, dn) }
    }
    #[cfg(not(ik_hip))]
    let _ = (gu, dn);
}

/// Tests / benchmarks: pairs per accumulator batch of the lean form below the pair table (4 | 8, bitwise the same output);
/// anything else = the default rule.
pub fn set_nb(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_set_nb(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_moe_set_nb(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}

/// Tests / benchmarks: the tensor-core prefill of the ordinary names (more than 512 pairs): 0 = off (the decode kernels, the
/// `_rows` twins' bits), 1 = the rule (64-pair tiles from 16 pairs per expert), 32 | 64 = every such call with that pair tile,
/// `-1` = the env (`INTARSIA_HIP_INK_MOE_WMMA`) or the default (the rule: on).
pub fn set_wmma(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_set_wmma(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_moe_set_wmma(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}

/// Tests: the pair tile (`0` = the decode kernels, `64` the rule's tile, or the forced `32`) the ordinary names take for a call of
/// `pairs` (token, expert) pairs over `experts` experts, as `set_wmma` / `INTARSIA_HIP_INK_MOE_WMMA` / the default resolve it
/// (the caller's own rule for a call that has no pair table, 512 pairs and fewer, is separate). Without the HIP kernel library: `0`.
pub fn wmma_pairs(pairs: i64, experts: i32) -> i32 {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_moe_wmma_pairs(p: std::ffi::c_longlong, e: std::ffi::c_int) -> std::ffi::c_int;
        }
        // SAFETY: reads plain ints of the kernel library and the environment.
        unsafe { ik_ink_moe_wmma_pairs(pairs, experts) }
    }
    #[cfg(not(ik_hip))]
    {
        let _ = (pairs, experts);
        0
    }
}

#[cfg(ik_hip)]
pub(crate) mod launch {
    use crate::device::InkArgs;
    use crate::ffi::{IkScratch, IkTensor};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type SC = *const IkScratch;
    type S = *mut c_void;

    extern "C" {
        fn ik_moe_experts_situ_bf16(x: T, ids: T, w: T, guw: T, gus: T, dnw: T, dns: T, gs: T, gc: f32, uc: f32, aq: c_int, out: TM, sc: SC, s: S) -> c_int;
        fn ik_moe_experts_situ_bf16_rows(x: T, ids: T, w: T, guw: T, gus: T, dnw: T, dns: T, gs: T, gc: f32, uc: f32, aq: c_int, out: TM, sc: SC, s: S) -> c_int;
        fn ik_moe_experts_mxfp8_bf16(x: T, ids: T, w: T, w13: T, s13: T, w2: T, s2: T, act: c_int, gc: f32, uc: f32, aq: c_int, out: TM, sc: SC, s: S) -> c_int;
        fn ik_moe_experts_mxfp8_bf16_rows(x: T, ids: T, w: T, w13: T, s13: T, w2: T, s2: T, act: c_int, gc: f32, uc: f32, aq: c_int, out: TM, sc: SC, s: S) -> c_int;
    }

    /// IK_ERR_INVALID_ARG / IK_ERR_UNSUPPORTED (kernels/abi/ik_abi.h)
    const INVALID_ARG: c_int = 1;

    /// The launcher status of `kernel` when it is one of this module's, `None` otherwise. Called from `HipBackend::launch` with
    /// the call's tensors already converted (`a.i(n)` input n, `a.o(n)` output n, `a.sc` scratch, `a.pos` the device position or
    /// null, `a.tab*` the row table or null, `a.s` the stream, `a.call.attrs`); the caller checks the status, traces and reports.
    ///
    /// # Safety
    /// As the launchers of `ffi`: the tensors are bounds-checked views of live device buffers held by `a.call`.
    pub(crate) unsafe fn launch(kernel: &str, a: &InkArgs) -> Option<c_int> {
        let (_, op) = super::TABLE.iter().find(|(n, _)| *n == kernel)?;
        let at = &a.call.attrs;
        let ink = &at.ink;
        // the op's own input count (a wrong op for the name would index past the inputs); a situ kernel takes only situ calls
        let n_in = if *op == intarsia_core::Op::MoeExperts { 8 } else { 7 };
        if *op != a.call.op || a.inp.len() < n_in || (*op == intarsia_core::Op::MoeExperts && !ink.situ) {
            return Some(INVALID_ARG);
        }
        let act: c_int = if ink.situ {
            2
        } else if at.gelu {
            1
        } else {
            0
        };
        let (gc, uc) = (ink.gate_cap, ink.up_cap);
        // bit 0: W4A4, bit 1: the decode repack (`ink.moe_layout`); the launchers refuse both (IK_ERR_UNSUPPORTED)
        let aq = ink.act_quant as c_int | ((ink.moe_layout as c_int & 1) << 1);
        Some(match kernel {
            "moe_experts_situ_bf16" => ik_moe_experts_situ_bf16(a.i(0), a.i(1), a.i(2), a.i(3), a.i(4), a.i(5), a.i(6), a.i(7), gc, uc, aq, a.o(0), a.sc, a.s),
            "moe_experts_situ_bf16_rows" => {
                ik_moe_experts_situ_bf16_rows(a.i(0), a.i(1), a.i(2), a.i(3), a.i(4), a.i(5), a.i(6), a.i(7), gc, uc, aq, a.o(0), a.sc, a.s)
            }
            "moe_experts_mxfp8_bf16" => ik_moe_experts_mxfp8_bf16(a.i(0), a.i(1), a.i(2), a.i(3), a.i(4), a.i(5), a.i(6), act, gc, uc, aq, a.o(0), a.sc, a.s),
            "moe_experts_mxfp8_bf16_rows" => {
                ik_moe_experts_mxfp8_bf16_rows(a.i(0), a.i(1), a.i(2), a.i(3), a.i(4), a.i(5), a.i(6), act, gc, uc, aq, a.o(0), a.sc, a.s)
            }
            _ => return None,
        })
    }
}
