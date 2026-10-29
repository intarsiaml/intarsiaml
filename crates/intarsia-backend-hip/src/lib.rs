//! HIP backend for IntarsiaML (ARCHITECTURE 3.1 / 3.3), AMD gfx12 (R9700, gfx1201).
//!
//! The device code lives in `kernels/hip/*.hip` behind the torch-free C ABI (`kernels/abi/ik_abi.h`);
//! `build.rs` compiles it with hipcc into a static library and this crate binds it through `extern "C"`
//! (`ffi.rs`). [`create`] opens a device and returns it as an `Arc<dyn Backend>`; the kernel table
//! (`kernels.rs`) registers every launcher with its shape predicate, exactness class and tolerance so
//! `intarsia kernels test --backend hip` can admit each one against the cpu reference.
//!
//! Without a ROCm toolchain at build time (the Mac, CI) the crate compiles to a stub and `create`
//! returns `Error::Unsupported`.
//!
//! The CUDA and HIP launcher libraries export the same `ik_*` C symbols, so a binary can link only one
//! of them (see the `compile_error!` in intarsia-cli).

pub mod disable;
pub mod kernels;
pub mod quantmoe;

#[cfg(ik_hip)]
mod comm;
#[cfg(ik_hip)]
mod device;
#[cfg(ik_hip)]
mod ffi;

use intarsia_core::Backend;
use std::sync::Arc;

/// True when this build contains the HIP kernels (a ROCm toolchain was found at build time).
pub const AVAILABLE: bool = cfg!(ik_hip);

/// Kernel variants chosen inside one op, as the kernel library resolves them in this process (environment knobs included;
/// host-only, no device): (query heads per block of flash v2, MoE projection variant bits at 64 pairs, at 32 pairs).
/// docs/results/2026-10-19-hip-round6.md; pinned by tests/round6_selection.rs.
#[cfg(ik_hip)]
pub fn inner_variants() -> (i32, i32, i32) {
    // SAFETY: plain host functions of the kernel library without arguments that point anywhere
    unsafe { (ffi::ik_hip_flash_hpb(), ffi::ik_hip_moe_mw(64), ffi::ik_hip_moe_mw(32)) }
}

/// The staging / math variant mask flash v2 runs in this process (INTARSIA_HIP_FLASH_VP, default 5; every value gives the same
/// bits; docs/results/2026-10-22-pf-prefill-ctx.md). Host-only; pinned by tests/pf_flash_vp.rs.
#[cfg(ik_hip)]
pub fn flash_vp() -> i32 {
    // SAFETY: a plain host function of the kernel library without arguments that point anywhere
    unsafe { ffi::ik_hip_flash_vp() }
}

/// The MoE down projection's variant bits (at 64 pairs, at 32), as the kernel library resolves them in this process (round 7
/// selects gate_up and down separately; `inner_variants` has gate_up). docs/results/2026-10-19-hip-round7.md; pinned by
/// tests/round6_selection.rs.
#[cfg(ik_hip)]
pub fn moe_down_variants() -> (i32, i32) {
    // SAFETY: plain host functions of the kernel library without arguments that point anywhere
    unsafe { (ffi::ik_hip_moe_mw_down(64), ffi::ik_hip_moe_mw_down(32)) }
}

/// The number of launch pieces the kernel library gives a launch estimated at `est_us` microseconds with at most `max_pieces`
/// pieces, under the current INTARSIA_HIP_SPLIT_US / INTARSIA_HIP_SPLIT_MIN_US (round 8 split, round 9 trigger;
/// docs/results/2026-10-19-hip-round9.md). Host-only; pinned by tests/split_trigger.rs.
#[cfg(ik_hip)]
pub fn split_pieces(est_us: f64, max_pieces: i64) -> i32 {
    // SAFETY: a plain host function of the kernel library without pointer arguments
    unsafe { ffi::ik_hip_split_pieces(est_us, max_pieces) }
}

/// Lane fuse27 / nq27: W4A8 GEMM calls in this process that read the activations their producer (norm, silu_glu, gated_rmsnorm) had
/// quantised (`INTARSIA_HIP_NORM_QUANT`, `INTARSIA_HIP_ACT_QUANT`) instead of running their own pre-pass (tests, diagnostics).
#[cfg(ik_hip)]
pub fn prequant_hits() -> u64 {
    device::PQ_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Open HIP device `ordinal`.
#[cfg(ik_hip)]
pub fn create(ordinal: usize) -> intarsia_core::Result<Arc<dyn Backend>> {
    disable::check_env()?;
    Ok(Arc::new(device::HipBackend::new(ordinal)?))
}

/// True when split-KV attention stage 1 runs at 8 waves per SIMD (`splitkv_kernel_o8`, round 9; opt-in with
/// INTARSIA_HIP_SPLITKV_OCC=1, otherwise the 7-wave kernel; read at every launch). Host-only; pinned by tests/splitkv_occ.rs.
#[cfg(ik_hip)]
pub fn splitkv_occ_selected() -> bool {
    // SAFETY: a plain host function of the kernel library without arguments
    unsafe { ffi::ik_hip_splitkv_occ() != 0 }
}

/// Test seam (tests/staging.rs, round 9): open HIP device `ordinal`, plus a function that keeps the backend's stream 0 busy
/// for about the given number of microseconds (a 1-thread kernel spinning on the wall clock).
#[cfg(ik_hip)]
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn create_with_spin(
    ordinal: usize,
) -> intarsia_core::Result<(Arc<dyn Backend>, Box<dyn Fn(u32) -> intarsia_core::Result<()>>)> {
    disable::check_env()?;
    let b = Arc::new(device::HipBackend::new(ordinal)?);
    let b2 = b.clone();
    Ok((b, Box::new(move |us| b2.spin(0, us))))
}

/// Open HIP device `ordinal` (stub: this build has no ROCm toolchain).
#[cfg(not(ik_hip))]
pub fn create(ordinal: usize) -> intarsia_core::Result<Arc<dyn Backend>> {
    Err(intarsia_core::Error::Unsupported(format!(
        "intarsia-backend-hip was built without a ROCm toolchain (device {ordinal}); install ROCm (hipcc) and set ROCM_PATH, then rebuild"
    )))
}

/// Open the HIP devices `ordinals` as a tensor-parallel group: one backend per device and the device collective
/// (peer-to-peer) between them. Rank `r` is `ordinals[r]`.
#[cfg(ik_hip)]
pub fn create_group(
    ordinals: &[usize],
) -> intarsia_core::Result<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn intarsia_core::Comm>>)> {
    disable::check_env()?;
    let hips = ordinals
        .iter()
        .map(|&o| device::HipBackend::new(o).map(Arc::new))
        .collect::<intarsia_core::Result<Vec<_>>>()?;
    let backends: Vec<Arc<dyn Backend>> = hips.iter().map(|h| h.clone() as Arc<dyn Backend>).collect();
    let comms = if hips.len() == 2 {
        comm::p2p_group(&hips)?
    } else {
        // the peer-to-peer kernel is a two-rank design; larger groups stage through the host
        intarsia_core::HostCommGroup::new(backends.clone())
    };
    Ok((backends, comms))
}

/// [`create_group`] for exactly two devices with the all-reduce handshake mode given explicitly instead of read from
/// `INTARSIA_HIP_AR_DRAIN` / `INTARSIA_HIP_AR_LL_MAX` (tests that compare modes in one process).
#[cfg(ik_hip)]
#[doc(hidden)]
pub fn create_group_mode(
    ordinals: &[usize],
    drain: i32,
    ll_max: usize,
) -> intarsia_core::Result<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn intarsia_core::Comm>>)> {
    disable::check_env()?;
    let hips = ordinals
        .iter()
        .map(|&o| device::HipBackend::new(o).map(Arc::new))
        .collect::<intarsia_core::Result<Vec<_>>>()?;
    let backends: Vec<Arc<dyn Backend>> = hips.iter().map(|h| h.clone() as Arc<dyn Backend>).collect();
    let comms = comm::p2p_group_mode(&hips, Some(drain), Some(ll_max))?;
    Ok((backends, comms))
}

/// [`create_group_mode`] with the wide fence path given explicitly (`Some((blocks, slot_bytes))`, or `None` for off) instead of read from
/// `INTARSIA_HIP_AR_WIDE` (lane pf27b's tests and rate benchmark).
#[cfg(ik_hip)]
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn create_group_wide(
    ordinals: &[usize],
    drain: i32,
    ll_max: usize,
    wide: Option<(i32, usize)>,
) -> intarsia_core::Result<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn intarsia_core::Comm>>)> {
    disable::check_env()?;
    let hips = ordinals
        .iter()
        .map(|&o| device::HipBackend::new(o).map(Arc::new))
        .collect::<intarsia_core::Result<Vec<_>>>()?;
    let backends: Vec<Arc<dyn Backend>> = hips.iter().map(|h| h.clone() as Arc<dyn Backend>).collect();
    let comms = comm::p2p_group_wide(&hips, Some(drain), Some(ll_max), Some(wide))?;
    Ok((backends, comms))
}

/// [`create_group_wide`] with the GEMM push of lane arx (`INTARSIA_HIP_AR_PUSH`) set explicitly (tests).
#[cfg(ik_hip)]
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn create_group_push(
    ordinals: &[usize],
    drain: i32,
    ll_max: usize,
    wide: Option<(i32, usize)>,
    push: bool,
) -> intarsia_core::Result<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn intarsia_core::Comm>>)> {
    disable::check_env()?;
    let hips = ordinals
        .iter()
        .map(|&o| device::HipBackend::new(o).map(Arc::new))
        .collect::<intarsia_core::Result<Vec<_>>>()?;
    let backends: Vec<Arc<dyn Backend>> = hips.iter().map(|h| h.clone() as Arc<dyn Backend>).collect();
    let comms = comm::p2p_group_push(&hips, Some(drain), Some(ll_max), Some(wide), Some(push))?;
    Ok((backends, comms))
}

/// Lane pf27b's P2P link micro-benchmark on two devices (`comm::p2p_bench`).
#[cfg(ik_hip)]
#[doc(hidden)]
pub fn p2p_bench(ordinals: &[usize], sizes: &[usize], blocks: &[i32], iters: usize) -> intarsia_core::Result<String> {
    let hips = ordinals
        .iter()
        .map(|&o| device::HipBackend::new(o).map(Arc::new))
        .collect::<intarsia_core::Result<Vec<_>>>()?;
    comm::p2p_bench(&hips, sizes, blocks, iters)
}

/// Open HIP devices as a tensor-parallel group (stub: this build has no ROCm toolchain).
#[cfg(not(ik_hip))]
pub fn create_group(
    ordinals: &[usize],
) -> intarsia_core::Result<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn intarsia_core::Comm>>)> {
    Err(intarsia_core::Error::Unsupported(format!(
        "intarsia-backend-hip was built without a ROCm toolchain (devices {ordinals:?}); install ROCm (hipcc) and set ROCM_PATH, then rebuild"
    )))
}
