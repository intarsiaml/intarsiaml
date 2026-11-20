//! Lane mtp-sampled: MTP drafts for sampled requests (temperature > 0) in the batching engine, seed-exact.
//!
//! The drafts are the MTP head's greedy chain, as for a greedy request. Every verify row comes back as its logits, which are
//! bitwise the logits of the plain decode step at that position ([`Model::forward_verify_logits`]); the sequence draws its own
//! token from each row with the code and the RNG of a plain step ([`Seq::token_from`], one draw per token) and keeps a draft
//! only where that draw equals it. The first row whose draw differs emits the draw and ends the pass.
//!
//! So the token at every position is the one plain sampling emits there: the request's RNG yields one value per generated
//! token, in order, whether a token came from a plain step or a verify row, and the row it is applied to holds the same bits.
//! The output of a seeded request is therefore identical with drafting on or off, and does not depend on which steps drafted
//! (the scheduler's batch-size gate, the drafter gates, the other sequences). This is TensorFold #2455's rule ("keep a draft
//! only where it equals the target's own keyed draw") and the acceptance of [`crate::spec::SampledSpec::Match`], whose draws
//! go through a different sampling routine and therefore change a seed's text.

use super::{spec_cut, Prep, Seq, SpecStep};
use crate::error::{Error, Result};
use crate::model::{Model, StepOut};
use crate::state::SequenceState;

impl Seq {
    /// Turn seed-exact sampled drafting on for this request if it qualifies: a sampled request under `--spec mtp` with the
    /// exclusions of greedy speculation (no logit trace, no constraint, verify passes enabled). Greedy requests are untouched.
    /// The batching engine calls it right after [`Seq::build`].
    pub(crate) fn enable_keyed(&mut self, mode: crate::spec::SpecMode, verify_rows: usize) {
        if mode == crate::spec::SpecMode::Mtp
            && !self.greedy
            && !self.spec_on
            && self.req.trace_logits.is_none()
            && self.cons.is_none()
            && verify_rows >= 2
        {
            self.spec_on = true;
            self.spec_keyed = true;
            self.ctx = self.req.prompt.clone();
        }
    }

    /// Whether the rows of this sequence's deferred verify pass must come back as logits ([`Seq::accept_verify_rows`]).
    pub(crate) fn keyed(&self) -> bool {
        self.spec_keyed
    }

    /// The draft step of a keyed sequence ([`Seq::prepare_with`], `Next::Draft(k)`): draft `k` tokens after `self.tok` with the
    /// MTP head's greedy chain; with `defer` the verify pass is the batching engine's ([`Prep::Verify`], then
    /// [`Seq::accept_verify_rows`]), otherwise it runs here on the sequence's own state.
    pub(super) fn keyed_draft(&mut self, model: &Model, st: &mut SequenceState, k: usize, defer: bool) -> Result<Prep> {
        self.cost_begin(true);
        if defer && std::mem::take(&mut self.hold_draft) {
            self.held = Some((self.tok, k));
            return Ok(Prep::Draft(k));
        }
        if !defer && model.sampled_verify_fast() && !super::mtp_verify_check() && model.spec_step_rows_ok(st, self.tok, k) {
            // lane samphost (`INTARSIA_SAMPLED_VERIFY_FAST=1`): draft and verify as one replay of the captured step, as a greedy
            // request's pass, with the rows read back for the draws (top-k where `Seq::verify_topk` allows it); the replay is
            // the eager draft and verify pass bit for bit, so the rows and drafts are the eager ones
            let t0 = crate::spec::trace_enabled().then(std::time::Instant::now);
            let len0 = st.len;
            if let Some((pass, rows)) = model.spec_step_rows(st, self.tok, k, self.verify_topk(k + 1))? {
                let t1 = std::time::Instant::now();
                let n = rows.len();
                let topk = rows.iter().any(|r| matches!(r, StepOut::TopK { .. }));
                self.keyed_accept(model, st, len0, &pass.drafts, rows)?;
                if let Some(t0) = t0 {
                    eprintln!(
                        "spec: keyed graph step {n} rows ({}) {:.3} ms, accept {:.3} ms",
                        if topk { "top-k" } else { "logits" },
                        (t1 - t0).as_secs_f64() * 1e3,
                        t1.elapsed().as_secs_f64() * 1e3
                    );
                }
                self.keyed_first()?;
                return Ok(Prep::Ready);
            }
            // capture failed (graph steps are off now): the eager pass below
        }
        let t_draft = (!defer && crate::spec::trace_enabled()).then(std::time::Instant::now);
        let drafts = model.mtp_draft(st, self.tok, k)?;
        if let Some(t) = t_draft {
            // lane samphost: the eager draft ends in its token download, so host time is device time
            eprintln!("spec: keyed draft {:.3} ms, {} drafts", t.elapsed().as_secs_f64() * 1e3, drafts.len());
        }
        self.keyed_finish(model, st, drafts, defer)
    }

    /// [`Seq::keyed_draft`] after the drafts (lane mdraft: they may come from a batched draft).
    pub(super) fn keyed_finish(&mut self, model: &Model, st: &mut SequenceState, drafts: Vec<u32>, defer: bool) -> Result<Prep> {
        if drafts.is_empty() {
            // no room for a draft block near the end of the context window: a plain step
            self.cost_begin(false);
            let (t, m) = self.plain_input();
            return Ok(Prep::Plain(t, m));
        }
        let mut ids = Vec::with_capacity(drafts.len() + 1);
        ids.push(self.tok);
        ids.extend_from_slice(&drafts);
        let len0 = st.len;
        if defer {
            self.deferred = Some((len0, drafts));
            return Ok(Prep::Verify(ids));
        }
        let t_pass = crate::spec::trace_enabled().then(std::time::Instant::now);
        // lane samphost (`INTARSIA_SAMPLED_VERIFY_FAST=1`): the rows as their device-side top-k where every row's plain step
        // would sample from it (`Seq::verify_topk`), so neither the whole rows nor the host's whole-row selection
        let fast = model.sampled_verify_fast().then(|| self.verify_topk(ids.len())).flatten();
        let rows: Vec<StepOut> = match fast {
            Some(k) => model.forward_verify_topk(st, &ids, k)?.1,
            None => {
                if model.sampled_verify_fast() {
                    model.note_verify_topk_fallback();
                }
                model.forward_verify_logits(st, &ids)?.1.into_iter().map(StepOut::Logits).collect()
            }
        };
        let t_acc = t_pass.map(|t| (t, std::time::Instant::now()));
        self.keyed_accept(model, st, len0, &drafts, rows)?;
        if let Some((t0, t1)) = t_acc {
            // lane samphost: the pass (its rows read back included) and the host's acceptance (draws, rollback)
            eprintln!(
                "spec: keyed verify {} rows ({}) {:.3} ms, accept {:.3} ms",
                ids.len(),
                if fast.is_some() { "top-k" } else { "logits" },
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }
        self.keyed_first()?;
        Ok(Prep::Ready)
    }

    /// Lane mtp2s (`INTARSIA_VERIFY_TOPK`): the candidate count `k` when every row of a verify pass of `rows` rows
    /// may come back as the device-side top-k, i.e. the plain step that draws each of those tokens would ask for the
    /// same `k` ([`Seq::topk_for`]).
    pub(crate) fn verify_topk(&self, rows: usize) -> Option<usize> {
        let k = self.topk_for(self.generated + 1)?;
        (1..rows).all(|i| self.topk_for(self.generated + 1 + i) == Some(k)).then_some(k)
    }

    /// The rows (logits, or each row's device-side top-k: lane mtp2s) of this sequence's deferred verify pass,
    /// run batched by the caller and committed to `st`: the keyed acceptance and rollback, then the first token of
    /// the pass becomes `self.tok`.
    pub(crate) fn accept_verify_rows(
        &mut self,
        model: &Model,
        st: &mut SequenceState,
        rows: Vec<StepOut>,
    ) -> Result<()> {
        let (len0, drafts) = self.deferred.take().ok_or_else(|| Error::Other("verify rows without a deferred pass".into()))?;
        self.keyed_accept(model, st, len0, &drafts, rows)?;
        self.keyed_first()
    }

    /// Row `i` of the pass over `[tok, d_1..d_m]` that started at `len0` holds the logits of generated token
    /// `generated + 1 + i` given that `d_1..d_i` were kept. Draw that token as the plain step would; keep going while the draw
    /// is the next draft. Rows after the first mismatch are never looked at (no draw is spent on them) and are rolled back.
    fn keyed_accept(
        &mut self,
        model: &Model,
        st: &mut SequenceState,
        len0: usize,
        drafts: &[u32],
        rows: Vec<StepOut>,
    ) -> Result<()> {
        if rows.len() != drafts.len() + 1 {
            return Err(Error::Other(format!("a sampled verify pass of {} drafts returned {} logits rows", drafts.len(), rows.len())));
        }
        let mut emit = Vec::with_capacity(rows.len());
        let mut accepted = drafts.len();
        let mut rows = rows.into_iter();
        for (i, row) in rows.by_ref().enumerate() {
            let x = self.token_from(row, self.generated + 1 + i)?;
            emit.push(x);
            if drafts.get(i) != Some(&x) {
                accepted = i;
                break;
            }
        }
        // the rows behind the first mismatch go back to the pool unread
        rows.for_each(|r| {
            if let StepOut::Logits(l) = r {
                crate::model::recycle_logits(l)
            }
        });
        spec_cut(model, st, len0 + accepted + 1)?;
        self.account_verify(len0, drafts, SpecStep { emit, accepted, accepted_exact: accepted, plain_next: false });
        Ok(())
    }

    /// Make the first token a keyed pass queued `self.tok` (a pass always emits at least the draw of row 0).
    fn keyed_first(&mut self) -> Result<()> {
        match self.pending.pop_front() {
            Some(t) => {
                self.tok = t;
                Ok(())
            }
            None => Err(Error::Other("a sampled verify pass emitted no token".into())),
        }
    }
}
