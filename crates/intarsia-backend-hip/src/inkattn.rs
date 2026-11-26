//! Kernel table and device dispatch of the Inkling-2-Mini attention ops: `InkKvPrologue` (70), `RelSinkAttn` (71), `InkAttnDecode` (75) (lane ink-hip-attn; port map
//! docs/results/2026-11-25-inkling2-hip.md). Kernels in kernels/hip/ink_attn.hip, ABI kernels/hip/ik_hip_abi_inkattn.h; contract
//! `intarsia_core::inkling2::ops`, admitted against the cpu reference (`ink_ref`) by `intarsia kernels test --backend hip --op <op>`.
//!
//! * `ink_kv_prologue_bf16`: a port of CUDA's `ink_prologue_k` (plus the conv-ring write pass when a long chunk would overwrite
//!   ring rows other rows still read: T > Rr - K + 1).
//! * `rel_sink_attn_bf16`: a port of CUDA's scalar split-KV path (`ink_attn_s1_k` + `ink_attn_comb_k` + `ink_kv_write_k`) at every
//!   T, on slices of at most `ATTN_SLICE` rows in order (the scratch bound). Rows-exact for every T, host or device position:
//!   a row's splits and reduction order are functions of its position and the window only.
//!   With `INTARSIA_HIP_INK_ATTN_FLASH` on ([`INK_ATTN_FLASH_DEFAULT`]: off), calls of more than 16 rows (prefill chunks) run a
//!   WMMA flash kernel instead (a fork of attention_flash.hip's kernel with the rel bias, the SWA window, the sink and the gate),
//!   then the cache-write pass: not bitwise the split-KV form; decode / verify rows (<= 16) never take it.
//! * `ink_attn_decode_bf16` (op 75, 1..=16 rows): bitwise the pair at every T <= 16, host or device position, on one of two paths
//!   the launcher picks (`INTARSIA_INK_DEC_FUSED_KEYS`, default 2048): visits that read at most that many keys (window layers;
//!   full layers at a host position up to that context) take the fused kernel (`ink_attn_dec_k`, one dispatch); others the
//!   split path (prologue writing r, stage 1 forming its split's bias from r, combine). Above 16 rows the model runs the pair.
//! Results: docs/results/2026-11-25-inkling2-hip-attn.md.

use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("ink_kv_prologue_bf16", Op::InkKvPrologue),
    ("rel_sink_attn_bf16", Op::RelSinkAttn),
    ("ink_attn_decode_bf16", Op::InkAttnDecode),
];

/// The ops this module is the home of (a lane's kernels register only these).
pub const OPS: &[Op] = &[Op::InkKvPrologue, Op::RelSinkAttn, Op::InkAttnDecode];

/// bf16-rounded outputs: a one-ulp flip is allowed (the CUDA twin's budget).
const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);

/// Most splits per row (`MAXS` of kernels/hip/ink_attn.hip): the scratch stride.
pub const MAX_SPLIT: usize = 32;

/// Most rows of one attention launch (`SLICE` of kernels/hip/ink_attn.hip): a longer call runs as slices in order.
pub const ATTN_SLICE: usize = 64;

/// Shapes the prologue takes (d_rel <= 32, K <= 4 are checked at launch).
fn prologue_ok(s: &ShapeInfo) -> bool {
    s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0 && s.n_heads / s.n_kv_heads <= 4 && (1..=256).contains(&s.head_dim)
}

/// Shapes the attention takes.
fn attn_ok(s: &ShapeInfo) -> bool {
    s.n_kv_heads > 0
        && s.n_heads % s.n_kv_heads == 0
        && (1..=3).contains(&(s.n_heads / s.n_kv_heads))
        && matches!(s.head_dim, 8 | 16 | 32 | 64 | 128)
        && s.m >= 1
}

/// The split-KV partials of one slice, f32 `acc [T',H,MAX_SPLIT,D]` + `(m, l) [T',H,MAX_SPLIT]`, T' = min(T, ATTN_SLICE). The
/// launcher checks the same expression.
pub fn attn_scratch(s: &ShapeInfo) -> usize {
    s.m.min(ATTN_SLICE) * s.n_heads * MAX_SPLIT * (s.head_dim + 2) * 4
}

/// Whether `rel_sink_attn_bf16` runs prefill chunks (more than 16 rows) on the WMMA flash kernel when `INTARSIA_HIP_INK_ATTN_FLASH`
/// is unset. Off: the flash form is not bitwise the split-KV form (G2 PASS, -12.4 % 4k prefill; results
/// docs/results/2026-11-25-inkling2-hip-attn.md); the default-on flip waits for the user and is this one constant.
pub const INK_ATTN_FLASH_DEFAULT: bool = false;

/// `INTARSIA_HIP_INK_ATTN_FLASH`: `1` = flash prefill, `0` = split-KV, unset = [`INK_ATTN_FLASH_DEFAULT`]. Read at every launch of
/// more than 16 rows (the benches and tests switch it in one process).
pub fn ink_attn_flash() -> bool {
    match std::env::var("INTARSIA_HIP_INK_ATTN_FLASH").ok().as_deref().map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => INK_ATTN_FLASH_DEFAULT,
    }
}

/// Op 75 shapes: the attention's, at most 16 rows (`ATTN_DECODE_MAX_ROWS`), kv heads within the ticket array.
fn decode_ok(s: &ShapeInfo) -> bool {
    attn_ok(s) && s.m <= intarsia_core::inkling2::ops::ATTN_DECODE_MAX_ROWS && s.n_kv_heads <= 256
}

/// Op 75 scratch: the partials of every row (T <= 16), the staged K / V rows `[2][T][Hk][D]` bf16, the split path's q `[T,H,D]`
/// and r `[T,H,32]` bf16. The launcher checks the same expression.
pub fn decode_scratch(s: &ShapeInfo) -> usize {
    s.m * s.n_heads * MAX_SPLIT * (s.head_dim + 2) * 4 + 2 * s.m * s.n_kv_heads * s.head_dim * 2 + s.m * s.n_heads * (s.head_dim + 32) * 2
}

pub fn all() -> Vec<KernelImpl> {
    let bf = [DType::Bf16];
    TABLE
        .iter()
        .map(|(name, op)| -> KernelImpl {
            // f32 math, bf16 stores at the contract's rounding points; rows-exact by construction
            let k = KernelImpl::new(*op, *name, "hip").arch([crate::kernels::ARCH]).acts(bf).exact(Exactness::Deterministic).tol(BF16_TOL);
            match *name {
                "ink_kv_prologue_bf16" => k.when(prologue_ok),
                "rel_sink_attn_bf16" => k.weights(bf).when(attn_ok).scratch(attn_scratch),
                // bitwise the pair (prologue + rel_sink_attn_bf16)
                "ink_attn_decode_bf16" => k.weights(bf).when(decode_ok).scratch(decode_scratch),
                other => unreachable!("inkattn kernel {other} has no registration"),
            }
        })
        .collect()
}

/// Mirror of `ik_ink_attn_attrs` in kernels/hip/ik_hip_abi_inkattn.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IkInkAttnAttrs {
    pub n_heads: i64,
    pub n_kv_heads: i64,
    pub head_dim: i64,
    pub d_rel: i64,
    pub pos0: i64,
    pub window: i64,
    pub tau_floor: i64,
    pub dyn_conv: i32,
    pub kv_ring: i32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub eps: f32,
    pub tau_alpha: f32,
    pub scale: f32,
    pub split_keys: i32,
    pub flash: i32,
    pub diag: i32,
}

/// Keys per split: `INTARSIA_INK_SPLIT_KEYS` (diagnostic for sweeps, as on CUDA; it changes the rounding order of every call
/// alike, so rows stay exact) or the kernel's default (0 = 64).
/// `INTARSIA_INK_DEC_DIAG` (timing diagnostic for op 75, WRONG output): bit 0 skips the prologue jobs, bit 1 the ticket merge.
#[allow(dead_code)]
fn dec_diag() -> i32 {
    static K: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *K.get_or_init(|| std::env::var("INTARSIA_INK_DEC_DIAG").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

#[allow(dead_code)]
fn split_keys() -> i32 {
    static K: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *K.get_or_init(|| std::env::var("INTARSIA_INK_SPLIT_KEYS").ok().and_then(|v| v.trim().parse().ok()).filter(|&k: &i32| k >= 1).unwrap_or(0))
}

#[cfg(ik_hip)]
pub(crate) mod launch {
    use super::IkInkAttnAttrs;
    use crate::device::InkArgs;
    use crate::ffi::{IkScratch, IkTensor};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type SC = *const IkScratch;
    type S = *mut c_void;
    type A = *const IkInkAttnAttrs;

    extern "C" {
        fn ik_ink_kv_prologue_bf16(pq: T, pk: T, pv: T, kt: T, vt: T, qs: T, rs: T, rp: T, a: A, q: TM, rel: TM, k: TM, v: TM, kr: TM, vr: TM, pos: T, s: S) -> c_int;
        fn ik_rel_sink_attn_bf16(q: T, k: T, v: T, kc_in: T, vc_in: T, rel: T, pq: T, a: A, o: TM, kc: TM, vc: TM, sc: SC, pos: T, s: S) -> c_int;
        fn ik_ink_attn_decode_bf16(
            pq: T,
            pk: T,
            pv: T,
            kt: T,
            vt: T,
            qs: T,
            rs: T,
            rp: T,
            kc_in: T,
            vc_in: T,
            a: A,
            o: TM,
            kc: TM,
            vc: TM,
            kr: TM,
            vr: TM,
            sc: SC,
            pos: T,
            s: S,
        ) -> c_int;
    }

    /// The launcher status of `kernel` when it is one of this module's, `None` otherwise. Called from `HipBackend::launch` with
    /// the call's tensors already converted (`a.i(n)` input n, `a.o(n)` output n, `a.sc` scratch, `a.pos` the device position or
    /// null, `a.tab*` the row table or null, `a.s` the stream, `a.call.attrs`); the caller checks the status, traces and reports.
    ///
    /// # Safety
    /// As the launchers of `ffi`: the tensors are bounds-checked views of live device buffers held by `a.call`.
    pub(crate) unsafe fn launch(kernel: &str, a: &InkArgs) -> Option<c_int> {
        if !matches!(kernel, "ink_kv_prologue_bf16" | "rel_sink_attn_bf16" | "ink_attn_decode_bf16") {
            return None;
        }
        let at = &a.call.attrs;
        let attrs = IkInkAttnAttrs {
            n_heads: at.n_heads as i64,
            n_kv_heads: at.n_kv_heads as i64,
            head_dim: at.head_dim as i64,
            d_rel: at.ink.d_rel as i64,
            pos0: at.pos0 as i64,
            window: at.window as i64,
            tau_floor: at.ink.tau_floor as i64,
            dyn_conv: at.ink.dyn_conv as i32,
            kv_ring: at.ink.kv_ring as i32,
            qk_clip: at.ink.qk_clip,
            rel_clip: at.ink.rel_clip,
            eps: at.eps,
            tau_alpha: at.ink.tau_alpha,
            scale: at.scale,
            split_keys: super::split_keys(),
            flash: (kernel == "rel_sink_attn_bf16"
                && a.call.inputs[0].shape.first().copied().unwrap_or(0) > intarsia_core::inkling2::ops::ATTN_DECODE_MAX_ROWS
                && super::ink_attn_flash()) as i32,
            diag: super::dec_diag(),
        };
        let ap = &attrs as *const IkInkAttnAttrs;
        Some(match kernel {
            "ink_kv_prologue_bf16" => ik_ink_kv_prologue_bf16(
                a.i(0),
                a.i(1),
                a.i(2),
                a.i(3),
                a.i(4),
                a.i(5),
                a.i(6),
                a.i(7),
                ap,
                a.o(0),
                a.o(1),
                a.o(2),
                a.o(3),
                a.o(4),
                a.o(5),
                a.pos,
                a.s,
            ),
            "ink_attn_decode_bf16" => ik_ink_attn_decode_bf16(
                a.i(0),
                a.i(1),
                a.i(2),
                a.i(3),
                a.i(4),
                a.i(5),
                a.i(6),
                a.i(7),
                a.i(8),
                a.i(9),
                ap,
                a.o(0),
                a.o(1),
                a.o(2),
                a.o(3),
                a.o(4),
                a.sc,
                a.pos,
                a.s,
            ),
            _ => ik_rel_sink_attn_bf16(a.i(0), a.i(1), a.i(2), a.i(3), a.i(4), a.i(5), a.i(6), ap, a.o(0), a.o(1), a.o(2), a.sc, a.pos, a.s),
        })
    }
}
