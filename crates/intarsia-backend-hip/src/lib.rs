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

/// Open HIP device `ordinal`.
#[cfg(ik_hip)]
pub fn create(ordinal: usize) -> intarsia_core::Result<Arc<dyn Backend>> {
    disable::check_env()?;
    Ok(Arc::new(device::HipBackend::new(ordinal)?))
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

/// Open HIP devices as a tensor-parallel group (stub: this build has no ROCm toolchain).
#[cfg(not(ik_hip))]
pub fn create_group(
    ordinals: &[usize],
) -> intarsia_core::Result<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn intarsia_core::Comm>>)> {
    Err(intarsia_core::Error::Unsupported(format!(
        "intarsia-backend-hip was built without a ROCm toolchain (devices {ordinals:?}); install ROCm (hipcc) and set ROCM_PATH, then rebuild"
    )))
}
