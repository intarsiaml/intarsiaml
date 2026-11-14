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
        let drafts = model.mtp_draft(st, self.tok, k)?;
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
        let (_, rows) = model.forward_verify_logits(st, &ids)?;
        self.keyed_accept(model, st, len0, &drafts, rows)?;
        self.keyed_first()?;
        Ok(Prep::Ready)
    }

    /// The logits rows of this sequence's deferred verify pass, run batched by the caller and committed to `st`: the keyed
    /// acceptance and rollback, then the first token of the pass becomes `self.tok`.
    pub(crate) fn accept_verify_rows(&mut self, model: &Model, st: &mut SequenceState, rows: Vec<Vec<f32>>) -> Result<()> {
        let (len0, drafts) = self.deferred.take().ok_or_else(|| Error::Other("verify rows without a deferred pass".into()))?;
        self.keyed_accept(model, st, len0, &drafts, rows)?;
        self.keyed_first()
    }

    /// Row `i` of the pass over `[tok, d_1..d_m]` that started at `len0` holds the logits of generated token
    /// `generated + 1 + i` given that `d_1..d_i` were kept. Draw that token as the plain step would; keep going while the draw
    /// is the next draft. Rows after the first mismatch are never looked at (no draw is spent on them) and are rolled back.
    fn keyed_accept(&mut self, model: &Model, st: &mut SequenceState, len0: usize, drafts: &[u32], rows: Vec<Vec<f32>>) -> Result<()> {
        if rows.len() != drafts.len() + 1 {
            return Err(Error::Other(format!("a sampled verify pass of {} drafts returned {} logits rows", drafts.len(), rows.len())));
        }
        let mut emit = Vec::with_capacity(rows.len());
        let mut accepted = drafts.len();
        for (i, row) in rows.into_iter().enumerate() {
            let x = self.token_from(StepOut::Logits(row), self.generated + 1 + i)?;
            emit.push(x);
            if drafts.get(i) != Some(&x) {
                accepted = i;
                break;
            }
        }
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
