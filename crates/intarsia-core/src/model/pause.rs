//! Lane interleave: a prefill pass that parks between layers, so the batched engine can run another slot's decode step in the
//! middle of a long chunk instead of after it (docs/results/2026-11-13-interleave.md).
//!
//! The thread that runs the pass installs a hook ([`Model::with_layer_pause`]); `run_layers` calls it before some layers of a
//! single-sequence pass of more than `rows` rows (one pause per `rows` rows, spread evenly over the layers). Around the hook the
//! pass's live state is set aside: the residual stream and the pending block output (`ws.h[cur]`, `ws.blk`) are copied to a
//! save buffer on the pass's stream and back after it, and the model's pass-scoped flags are cleared and restored. Whatever
//! the hook runs on the same model in between (a decode step of another sequence state) therefore sees the model as between
//! two passes, and the paused pass resumes with exactly the inputs it had: the chunk, its kernels and their inputs are the
//! same, so its bits are too, wherever and however often it pauses.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use super::Model;
use crate::backend::MemKind;
use crate::error::Result;

/// The hook a pausing pass calls between layers: `Ok` resumes the pass, `Err` ends it with that error.
pub type PauseFn = Box<dyn FnMut() -> Result<()>>;

struct Hook {
    rows: usize,
    f: Option<PauseFn>,
}

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// Removes this thread's hook when [`Model::with_layer_pause`] returns, however it returns.
struct Uninstall;

impl Drop for Uninstall {
    fn drop(&mut self) {
        HOOK.with(|h| h.borrow_mut().take());
    }
}

impl Model {
    /// Run `f` on this thread with `hook` called between layers of every single-sequence pass of more than `rows` rows it runs
    /// (`rows` 0: never). The hook may run passes of OTHER sequence states on this model (on another thread, while this one
    /// waits in it); it must not run anything on the paused pass's state.
    pub fn with_layer_pause<R>(&self, rows: usize, hook: PauseFn, f: impl FnOnce() -> R) -> R {
        HOOK.with(|h| *h.borrow_mut() = Some(Hook { rows, f: Some(hook) }));
        let _un = Uninstall;
        f()
    }

    /// Wait until the device has run everything enqueued on this model's stream: what a parked pass submitted before its park
    /// (lane mtp-store, `BatchModel::park_sync`). Called from the thread that runs the park's decode phase.
    pub(crate) fn park_sync(&self) -> Result<()> {
        self.backend.synchronize(self.stream)
    }

    /// The layers a single-sequence pass of `t` rows pauses in front of (empty: none, the common case). A pass pauses only
    /// with a hook installed on this thread, on one rank, and outside plan recordings.
    pub(super) fn pause_layers(&self, t: usize) -> Vec<usize> {
        let rows = HOOK.with(|h| h.borrow().as_ref().map_or(0, |h| h.rows));
        if rows == 0 || t <= rows || self.tp.world != 1 || !self.peers.is_empty() || self.remote.is_some() {
            return Vec::new();
        }
        if self.recorder.lock().unwrap().is_some() || self.gemma.is_some() {
            return Vec::new();
        }
        let n = self.layers.len();
        let segs = t.div_ceil(rows).min(n.max(1));
        let mut out: Vec<usize> = (1..segs).map(|k| k * n / segs).filter(|&l| l > 0 && l < n).collect();
        out.dedup();
        out
    }

    /// Park the pass in front of a layer: set aside the live residual rows (`ws.h[cur]`, and `ws.blk` when a block output is
    /// pending), clear the pass-scoped flags, call the hook, then put both back. `t`: the pass's rows.
    pub(super) fn layer_pause(&self, t: usize, cur: usize, pending: bool) -> Result<()> {
        let bytes = t * self.dims.hidden * self.act.size_bytes();
        let save = {
            let mut g = self.pause_save.lock().unwrap();
            let want = 2 * self.ws.max_tokens * self.dims.hidden * self.act.size_bytes();
            if g.as_ref().is_none_or(|b| b.bytes() < want) {
                *g = Some(self.backend.alloc(want, MemKind::Device)?);
            }
            g.as_ref().unwrap().clone()
        };
        let b = &*self.backend;
        b.copy(&save, 0, &self.ws.h[cur], 0, bytes, self.stream)?;
        if pending {
            b.copy(&save, bytes, &self.ws.blk, 0, bytes, self.stream)?;
        }
        let flags: [&AtomicBool; 8] = [
            &self.in_layers,
            &self.tail_dead,
            &self.prefill_inv,
            &self.exact_rows,
            &self.head_rows_attn,
            &self.head_whole,
            &self.kv_only_skip_q,
            &self.noncausal_rows,
        ];
        let kept: Vec<bool> = flags.iter().map(|f| f.swap(false, Ordering::Relaxed)).collect();
        let mut f = HOOK.with(|h| h.borrow_mut().as_mut().and_then(|h| h.f.take()));
        let r = f.as_mut().map_or(Ok(()), |f| f());
        HOOK.with(|h| {
            if let Some(h) = h.borrow_mut().as_mut() {
                h.f = f;
            }
        });
        for (f, v) in flags.iter().zip(kept) {
            f.store(v, Ordering::Relaxed);
        }
        r?;
        b.copy(&self.ws.h[cur], 0, &save, 0, bytes, self.stream)?;
        if pending {
            b.copy(&self.ws.blk, 0, &save, bytes, bytes, self.stream)?;
        }
        Ok(())
    }
}
