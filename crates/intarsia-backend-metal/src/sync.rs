//! Device counters for kernels that hand a stage's result from the threadgroups of a dispatch to the last one to finish it
//! (kernels/metal/inkling2_ctrl.metal, `rt_last`). One untracked shared buffer of [`SYNC_COUNTERS`] u32 per backend, zeroed
//! once at creation. Each launch takes the next run of counters (one per row) round the arena, so two launches in flight at
//! once (overlapping command buffers, hostgap) never share a counter as long as fewer than `SYNC_COUNTERS / rows` launches are
//! in flight; the kernel's last threadgroup resets each counter it used to 0, so no host memset is ever needed.

use crate::dense::Bind;
use crate::MetalBackend;
use intarsia_core::{Error, Result};
use objc2_metal::MTLBuffer;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counters in the arena (256 KiB).
pub(crate) const SYNC_COUNTERS: usize = 1 << 16;
/// Most counters one launch takes.
pub(crate) const SYNC_TAKE_MAX: usize = 4096;

pub(crate) struct SyncArena {
    buf: crate::buffer::MtlBuffer,
    next: AtomicUsize,
}

impl MetalBackend {
    /// `n` fresh counters (all 0 when the GPU reaches the launch: every user resets the ones it took).
    pub(crate) fn sync_counters(&self, n: usize) -> Result<Bind> {
        if n == 0 || n > SYNC_TAKE_MAX {
            return Err(Error::invalid(format!("metal sync arena: {n} counters (1..={SYNC_TAKE_MAX})")));
        }
        let a = match self.sync.get() {
            Some(a) => a,
            None => {
                let buf = self.new_mtl_buffer(SYNC_COUNTERS * 4, true)?;
                // SAFETY: a fresh shared buffer of SYNC_COUNTERS * 4 bytes, not yet visible to the GPU.
                unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, SYNC_COUNTERS * 4) };
                let _ = self.sync.set(SyncArena { buf, next: AtomicUsize::new(0) });
                self.sync.get().expect("set above")
            }
        };
        let start = a.next.fetch_add(n, Ordering::Relaxed) % (SYNC_COUNTERS - SYNC_TAKE_MAX);
        Ok(Bind { mtl: a.buf.clone(), off: start * 4, skip: 0 })
    }
}
