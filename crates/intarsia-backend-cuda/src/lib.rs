//! CUDA backend for IntarsiaML (ARCHITECTURE 3.1 / 3.3).
//!
//! The device code lives in `kernels/cuda/*.cu` behind the torch-free C ABI
//! (`kernels/abi/ik_abi.h`); `build.rs` compiles it with nvcc into a static library and this crate
//! binds it through `extern "C"` (`ffi.rs`). [`create`] opens a device and returns it as an
//! `Arc<dyn Backend>`; the kernel table (`kernels.rs`) registers every launcher with its shape
//! predicate, exactness class and tolerance so `intarsia kernels test --backend cuda` can admit
//! each one against the cpu reference.
//!
//! Without a CUDA toolkit at build time (the Mac, CI) the crate compiles to a stub and `create`
//! returns `Error::Unsupported`.

pub mod disable;
pub mod exl3;
pub mod hostmem;
pub mod kernels;
pub mod quantmoe;

#[cfg(ik_cuda)]
mod device;
#[cfg(ik_cuda)]
mod device_quantmoe;
#[cfg(ik_cuda)]
mod ffi;
/// NCCL collectives for tensor parallelism across processes (docs/design/35b-tp2.md).
#[cfg(ik_cuda)]
pub mod nccl;
/// NCCL collectives (stub: this build has no CUDA toolkit).
#[cfg(not(ik_cuda))]
pub mod nccl {
    use intarsia_core::{Backend, Comm, Error, Result};
    use std::sync::Arc;
    fn none() -> Error {
        Error::Unsupported("this binary was built without the CUDA backend (no NCCL)".into())
    }
    pub fn version() -> Result<String> {
        Err(none())
    }
    pub fn unique_id() -> Result<[u8; 128]> {
        Err(none())
    }
    pub fn create(_ordinal: usize, _rank: usize, _world: usize, _id: [u8; 128]) -> Result<(Arc<dyn Backend>, Arc<dyn Comm>)> {
        Err(none())
    }
}

use intarsia_core::Backend;
use std::sync::Arc;

/// True when this build contains the CUDA kernels (a CUDA toolkit was found at build time).
pub const AVAILABLE: bool = cfg!(ik_cuda);

/// Open CUDA device `ordinal`.
#[cfg(ik_cuda)]
pub fn create(ordinal: usize) -> intarsia_core::Result<Arc<dyn Backend>> {
    disable::check_env()?;
    Ok(Arc::new(device::CudaBackend::new(ordinal)?))
}

/// Open CUDA device `ordinal` (stub: this build has no CUDA toolkit).
#[cfg(not(ik_cuda))]
pub fn create(ordinal: usize) -> intarsia_core::Result<Arc<dyn Backend>> {
    Err(intarsia_core::Error::Unsupported(format!(
        "intarsia-backend-cuda was built without a CUDA toolkit (device {ordinal}); install CUDA and set CUDA_PATH, then rebuild"
    )))
}
