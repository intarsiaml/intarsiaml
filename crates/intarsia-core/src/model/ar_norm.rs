//! The block-end all-reduce fused with the residual add + rmsnorm that follows it (`INTARSIA_TP_ARNORM`).
//!
//! Today a layer ends `all_reduce(partial -> blk)`, and the next block starts `RmsNormAdd(blk, res, w) -> (n, sum)`
//! (when the norm is not already folded into that block's first GEMV pair, which stays as it is: those blocks never see a
//! standalone `RmsNormAdd`). Two launches and a full round trip of `blk` through memory. With a communicator that has
//! [`Comm::all_reduce_add_rmsnorm`] the pair is one collective whose result is bitwise the pair's.
//!
//! The block does not know what comes next, so the reduction is *deferred*: [`Model::all_reduce`] records it
//! ([`DeferredAr`]) instead of running it, and the next op decides in [`Model::resolve_ar`], called by `exec_call` before
//! anything else is emitted:
//!
//! * an `Op::RmsNormAdd` whose `x` is the reduced tensor, on arguments the communicator accepts (`ar_norm_supported`, a pure
//!   function of dtypes, shapes and offsets): the fused collective runs in place of both;
//! * anything else (a folded norm, a different op, a shape the communicator refuses): the reduction runs now, exactly as
//!   `all_reduce` would have run it, and the op follows.
//!
//! Every rank runs the same model code on the same shapes, so every rank takes the same branch and the collectives stay in
//! lockstep (a captured graph records the same sequence). The reduction is flushed at the end of the layer loop too (the
//! dead tail ends on one). The plan recorder takes the same decisions: it records the collective as usual, marks it
//! `fused_norm`, and records the `RmsNormAdd` call unchanged, so op counts and the batch-consistency rule see no
//! difference; `plan show` prints how many collectives fuse.
//!
//! Off (the default, `Model::tp_arnorm` false) nothing here runs and `all_reduce` is the code it always was.

use super::*;
use crate::comm::ArNorm;

/// An all-reduce `partial -> out` that has not been issued yet.
pub(super) struct DeferredAr {
    partial: TensorRef,
    out: TensorRef,
    /// Index of its [`crate::plan::PlannedComm`] while a plan is recorded.
    plan_idx: Option<usize>,
}

/// Whether a model loaded with `opts` as rank `tp` defers its block-end reductions. Needs row-local fusion on at world 2, a
/// communicator that fuses, fusion not disabled (`INTARSIA_NO_FUSION`) and no collective-skipping test hook.
pub(super) fn enabled(opts: &LoadOptions, tp: &TpRank, no_comm: bool, test_bug: bool) -> bool {
    static NO_FUSION: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    opts.tp_arnorm
        && opts.tp_fusion
        && tp.world == 2
        && !no_comm
        && !test_bug
        && !*NO_FUSION.get_or_init(|| std::env::var_os("INTARSIA_NO_FUSION").is_some_and(|v| v != "0"))
        && tp.comm.as_ref().is_some_and(|c| c.fuses_ar_norm())
}

fn same_tensor(a: &TensorRef, b: &TensorRef) -> bool {
    a.buf.ptr_eq(&b.buf) && a.offset == b.offset && a.dtype == b.dtype && a.shape == b.shape
}

impl Model {
    /// `eager=N captured=M`: the fused collectives this rank ran eagerly, and the ones recorded into a decode graph (each
    /// replay of the graph runs them again). Printed when the model is dropped.
    pub(super) fn ar_fused_counts(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;
        format!("eager={} captured={}", self.ar_fused_eager.load(Relaxed), self.ar_fused_captured.load(Relaxed))
    }

    /// Hold `partial -> out` until the next op (see the module docs). `plan_idx`: its entry in the recorder's collectives.
    pub(super) fn defer_ar(&self, partial: &TensorRef, out: &TensorRef, plan_idx: Option<usize>) {
        let old = self.ar_defer.lock().unwrap().replace(DeferredAr { partial: partial.clone(), out: out.clone(), plan_idx });
        debug_assert!(old.is_none(), "an all-reduce was deferred while another was pending");
    }

    /// Issue the deferred all-reduce, if there is one, the way `all_reduce` runs it.
    pub(super) fn flush_ar(&self, profile: bool) -> Result<()> {
        let d = self.ar_defer.lock().unwrap().take();
        match d {
            Some(d) => self.reduce_deferred(&d, profile),
            None => Ok(()),
        }
    }

    fn reduce_deferred(&self, d: &DeferredAr, profile: bool) -> Result<()> {
        if d.plan_idx.is_some() {
            return Ok(()); // recorded: nothing runs
        }
        let comm = self.tp.comm.as_ref().ok_or_else(|| Error::invalid("a deferred all-reduce without a communicator"))?;
        self.reduce_now(comm, &d.partial, &d.out, profile)
    }

    /// Called by `exec_call` with the call about to be emitted. Returns `true` when the deferred all-reduce (if any) and
    /// `call` have both been run as the fused collective, so the caller must not run `call`. Otherwise the reduction is
    /// out of the way (issued, or only marked fused in a plan recording) and the caller emits `call` as usual.
    pub(super) fn resolve_ar(&self, call: &OpCall, profile: bool) -> Result<bool> {
        let d = self.ar_defer.lock().unwrap().take();
        let Some(d) = d else { return Ok(false) };
        let comm = self.tp.comm.as_ref().ok_or_else(|| Error::invalid("a deferred all-reduce without a communicator"))?;
        if call.op == Op::RmsNormAdd && call.inputs.len() == 3 && call.outputs.len() == 2 && same_tensor(&call.inputs[0], &d.out) {
            let a = ArNorm {
                partial: &d.partial,
                out: &d.out,
                residual: &call.inputs[1],
                weight: &call.inputs[2],
                norm: &call.outputs[0],
                sum: &call.outputs[1],
                eps: call.attrs.eps,
                w_offset: call.attrs.w_offset,
            };
            if comm.ar_norm_supported(&a) {
                if let Some(idx) = d.plan_idx {
                    if let Some(r) = self.recorder.lock().unwrap().as_mut() {
                        r.comm[idx].fused_norm = true;
                    }
                    return Ok(false);
                }
                if profile {
                    // a profiled step times the collective on its own: the pair is one launch, its time is the comm's
                    self.backend.synchronize(self.stream)?;
                }
                let t0 = std::time::Instant::now();
                comm.all_reduce_add_rmsnorm(&a, 0)?;
                let mut captured = false;
                if profile {
                    self.backend.synchronize(self.stream)?;
                    let d = t0.elapsed();
                    comm.note_timed(d);
                    self.telemetry.record_comm(Some(d));
                    self.telemetry.record_op(Op::RmsNormAdd, Some(std::time::Duration::ZERO));
                } else if let Some(ops) = self.capture_ops.lock().unwrap().as_mut() {
                    // being captured: nothing ran, each replay counts the pair
                    ops.push(Op::RmsNormAdd);
                    self.ar_fused_captured.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    captured = true;
                    self.capture_comms.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else {
                    self.telemetry.record_comm(None);
                    self.telemetry.record_op(Op::RmsNormAdd, None);
                }
                if !captured {
                    self.ar_fused_eager.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Ok(true);
            }
        }
        self.reduce_deferred(&d, profile)?;
        Ok(false)
    }
}
