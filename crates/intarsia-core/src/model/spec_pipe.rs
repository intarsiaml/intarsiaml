//! The MTP speculative step without host round trips inside it (lane mmtp, `INTARSIA_METAL_MTP_GPU_ACCEPT`, Metal).
//!
//! The eager step ([`Model::mtp_draft`] + [`Model::forward_verify`]) reads every draft on the host before the next head pass
//! is encoded, and every pass uploads its ids, which drains the queue on Metal (`upload` synchronizes): about six GPU idle
//! gaps of 0.15-0.2 ms per verify pass on the 35B (docs/results/2026-10-29-mmtp-metal-gpu-accept.md). Here one step is:
//!
//! ```text
//!   stage      ws.ids[0] = next, ws.ids[R ..] = the catch-up ids              (one upload, while the GPU is idle)
//!   catch-up   the head over the n queued rows, ids from ws.ids[R ..]         -> argmax into ws.ids[1]
//!   chain      k - 1 one-row head passes, each embedding ws.ids[i]           -> argmax into ws.ids[1 + i]
//!   verify     the main model over ws.ids[0 .. k + 1] on the row-exact kernels -> verify_tok
//!   read       verify_tok and ws.ids (the drafts): the step's only wait
//! ```
//!
//! and the next step's catch-up and chain ([`Model::spec_pipe_predraft`]) are encoded and submitted right after acceptance
//! and rollback, before the engine emits the pass's tokens, so the GPU drafts while the host emits and encodes the next
//! verify pass. The ids of that next draft are staged right after the read, while the queue is still empty. `R` is the
//! model's verify row count (`Model::enable_verify`), so the catch-up ids never overlap the verify input.
//!
//! Same kernels on the same inputs in the same order as the eager step (the head passes as in `mtp_draft_rank`, the verify
//! pass as in `forward_verify_inner`): only the buffer the embeddings read their ids from and where command buffers end
//! change. CUDA and HIP run the same sequence as one captured graph (`spec_graph.rs`), with acceptance and rollback on the
//! host after the one download, as here. Acceptance stays the host's `spec::accept`; rollback is the same
//! `SequenceState::spec_rollback` (GPU copies of the saved recurrent rows, queued without a wait).

use super::feedback::UNREAD;
use super::*;

/// `INTARSIA_METAL_MTP_GPU_ACCEPT` (`0` keeps the eager draft + verify step), read once.
fn env_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_MTP_GPU_ACCEPT").map_or(true, |v| v.trim() != "0"))
}

/// Ids already in `ws.ids` for a next step at `len` tokens fed `next` (`ws.ids[0]`), catch-up ids `catch` (from
/// [`Model::pipe_catch_at`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Staged {
    len: usize,
    next: u32,
    catch: Vec<u32>,
}

/// The catch-up and chain of a step at `len` tokens fed `next`, with `k` drafts over `n` queued rows, submitted ahead of
/// it ([`Model::spec_pipe_predraft`]): the drafts are in `ws.ids[1 ..= k]`. The host mirror is unchanged until the step
/// takes it ([`Model::spec_pipe_step`]); a step that does not leaves only head cache rows past `kv.len`, which the next
/// head pass writes again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Predraft {
    len: usize,
    next: u32,
    k: usize,
    n: usize,
}

/// What one pipelined step returns.
pub(crate) struct PipePass {
    pub(crate) drafts: Vec<u32>,
    pub(crate) out: Vec<u32>,
    /// The ids staged for the next step, assuming the acceptance `spec::accept` gives.
    pub(crate) staged: Option<Staged>,
}

impl Model {
    /// The pipelined step is switched on for this model: by default on Metal (one stream, `fence` commits without waiting)
    /// with the knob on; [`Model::set_spec_pipe`] overrides it.
    pub(crate) fn spec_pipe_on(&self) -> bool {
        match self.spec_pipe.load(std::sync::atomic::Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => env_on() && self.backend.name() == "metal",
        }
    }

    /// Force the pipelined MTP step on (`Some(true)`, on any backend: one whose `fence` waits runs it with that wait) or off
    /// (`Some(false)`), or back to the default (`None`: Metal, `INTARSIA_METAL_MTP_GPU_ACCEPT`). Tests compare both paths.
    pub fn set_spec_pipe(&self, on: Option<bool>) {
        self.spec_pipe.store(on.map_or(0, |b| if b { 1 } else { 2 }), std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether [`Model::spec_pipe_step`] can run `k` drafts after `next` on `st` now (else the eager step).
    pub(crate) fn spec_pipe_ok(&self, st: &SequenceState, next: u32, k: usize) -> bool {
        let Some(m) = st.mtp.as_ref() else { return false };
        let q = k + 1;
        let n = m.pending_rows;
        self.spec_pipe_on()
            && k >= 1
            && self.mtp.is_some()
            // scripted drafts (tests) replace the head's when there are k of them; fewer: the eager step verifies exactly those
            && self.scripted_drafts(st, next, k).map_or(true, |d| d.len() == k)
            && (next as usize) < self.dims.vocab
            && self.recorder.lock().unwrap().is_none()
            && self.tp.world == 1
            && self.peers.is_empty()
            && st.peers.is_empty()
            && st.remote.is_none()
            && st.dflash.is_none()
            && st.id != 0
            && (2..=self.verify_rows).contains(&q)
            && q <= Self::VERIFY_MAX_ROWS
            && self.verify_tok.is_some()
            && self.ws.max_tokens >= 2 * self.verify_rows
            && m.active
            && m.synced
            && n >= 1
            && n <= q
            && m.kv.len + n == st.len
            // the verify rows, and the head's chained rows (positions st.len .. st.len + k - 2), fit their caches
            && st.len + q <= st.cap
            && st.len + k <= m.kv.cap
            && !(self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0)
    }

    /// The catch-up ids of a step at `st` fed `next`: the head row of position `p0 + i` pairs with the token of position
    /// `p0 + i + 1`, the last one is `next`.
    fn pipe_catch(st: &SequenceState, next: u32) -> Result<Vec<u32>> {
        let m = st.mtp.as_ref().ok_or_else(|| Error::invalid("sequence without MTP state"))?;
        let (p0, n) = (m.kv.len, m.pending_rows);
        let mut catch: Vec<u32> = st.tokens.get(p0 + 1..(p0 + n + 1).min(st.len)).unwrap_or(&[]).to_vec();
        if catch.len() < n {
            catch.push(next);
        }
        if catch.len() != n || catch.contains(&UNREAD) {
            return Err(Error::invalid(format!("MTP pipelined step: catch-up of {n} rows has {} tokens", catch.len())));
        }
        Ok(catch)
    }

    /// Write `ws.ids[0] = next` and the catch-up ids (one upload: on Metal it drains the queue, so callers stage while it is empty).
    fn pipe_stage(&self, s: &Staged) -> Result<()> {
        if let Some(&bad) = s.catch.iter().find(|&&i| i as usize >= self.dims.vocab) {
            return Err(Error::BadRequest(format!("token id {bad} >= vocab {}", self.dims.vocab)));
        }
        let at = self.pipe_catch_at();
        let mut v = vec![0i32; at + s.catch.len()];
        v[0] = s.next as i32;
        for (i, &t) in s.catch.iter().enumerate() {
            v[at + i] = t as i32;
        }
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.backend.upload(&self.ws.ids, 0, &bytes, self.stream)
    }

    /// First `ws.ids` entry of the head's catch-up ids: entries `0..q` (`q <= verify_rows`) are the verify input.
    fn pipe_catch_at(&self) -> usize {
        self.verify_rows
    }

    /// `i32 [len]` view of `ws.ids` from entry `i`.
    fn pipe_ids(&self, i: usize, len: usize) -> TensorRef {
        let mut v = self.view(&self.ws.ids, DType::I32, &[len]);
        v.offset = 4 * i;
        v
    }

    /// Encode the head's catch-up over the `n` queued rows (ids at [`Model::pipe_catch_at`]) and its `k - 1` chained passes
    /// for a step fed `next`: the drafts land in `ws.ids[1 ..= k]`. The launches of `mtp_draft_rank` (catch-up
    /// `mtp_run_pending` with `next`, then the chain), ids read on the device. The head cache is back at `kv.len`
    /// afterwards (the host mirror is unchanged). A test's scripted drafts ([`Model::mtp_script`]) then replace the head's.
    fn pipe_draft(&self, st: &mut SequenceState, next: u32, n: usize, k: usize) -> Result<()> {
        let script = self.scripted_drafts(st, next, k);
        let head = self.mtp.as_ref().ok_or_else(|| Error::invalid("no MTP head"))?;
        let mut m = st.mtp.take().ok_or_else(|| Error::invalid("sequence without MTP state"))?;
        let base0 = m.kv.len;
        let h = self.dims.hidden;
        let r = (|| -> Result<()> {
            let pending = TensorRef::new(m.pending.clone(), self.act, &[m.cap_rows, h]);
            let out = self.mtp_layer_at(
                head,
                &m.kv,
                n,
                None,
                &pending.rows(0, n),
                Some(&self.pipe_ids(self.pipe_catch_at(), n)),
                None,
                false,
            )?;
            m.kv.commit(&vec![UNREAD; n]);
            self.draft_head_to_device_into(&out, n - 1, false, self.pipe_ids(1, 1))?;
            self.backend.copy(
                &head.chain,
                0,
                &out.buf,
                out.offset + (n - 1) * m.row_bytes,
                m.row_bytes,
                self.stream,
            )?;
            for i in 1..k {
                let hid = TensorRef::new(head.chain.clone(), self.act, &[1, h]);
                let out = self.mtp_layer_at(head, &m.kv, 1, None, &hid, Some(&self.pipe_ids(i, 1)), None, false)?;
                m.kv.commit(&[UNREAD]);
                self.draft_head_to_device_into(&out, 0, false, self.pipe_ids(1 + i, 1))?;
                self.backend.copy(&head.chain, 0, &out.buf, out.offset, m.row_bytes, self.stream)?;
            }
            if let Some(d) = script.filter(|d| d.len() == k) {
                let b: Vec<u8> = d.iter().flat_map(|&t| (t as i32).to_le_bytes()).collect();
                self.backend.upload(&self.ws.ids, 4, &b, self.stream)?;
            }
            Ok(())
        })();
        let cut = m.kv.truncate(base0, &*self.backend);
        if r.is_err() || cut.is_err() {
            m.synced = false;
        }
        st.mtp = Some(m);
        r?;
        cut
    }

    /// One speculative step ([`Model::spec_pipe_ok`]): draft `k` tokens after `next` (unless `pre` already did, for exactly
    /// this step) and run the verify pass over `[next, drafts]`, with one wait at the end. On return the state is what
    /// [`Model::mtp_draft`] + [`Model::forward_verify`] leave: the verify rows committed (`st.len` + `k + 1`), the head's
    /// cache at the committed positions before them, its queue holding their hidden rows. The caller accepts and rolls back.
    pub(crate) fn spec_pipe_step(
        &self,
        st: &mut SequenceState,
        next: u32,
        k: usize,
        pre: Option<Predraft>,
        staged: Option<&Staged>,
    ) -> Result<PipePass> {
        if !self.spec_pipe_ok(st, next, k) {
            return Err(Error::invalid("spec_pipe_step: not applicable (check Model::spec_pipe_ok)"));
        }
        let q = k + 1;
        let len0 = st.len;
        let n = st.mtp.as_ref().map_or(0, |m| m.pending_rows);
        let catch = Self::pipe_catch(st, next)?;
        Self::check_forward(st, &vec![0u32; q], HeadMode::LastArgmax, false)?;
        let drafted = pre == Some(Predraft { len: len0, next, k, n });
        let run = (|| -> Result<(Vec<u8>, Vec<u8>, TensorRef)> {
            if !drafted {
                let s = Staged { len: len0, next, catch: catch.clone() };
                if staged != Some(&s) {
                    self.pipe_stage(&s)?;
                }
                self.pipe_draft(st, next, n, k)?;
            }
            // the host mirror of the draft: the head took the catch-up rows, its queue is empty
            {
                let m = st.mtp.as_mut().expect("checked by spec_pipe_ok");
                m.kv.commit(&catch);
                m.pending_rows = 0;
            }
            self.det_tap_begin();
            Self::prepare_spec_rows(&*self.backend, st, q)?;
            self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
            let run = (|| -> Result<TensorRef> {
                let hn = self.run_layers(Seqs::One(st), &vec![0u32; q], false, None, false, None)?;
                self.verify_head(&hn, q, false)?;
                Ok(hn)
            })();
            self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
            let hn = run?;
            let tok = self.verify_tok.as_ref().expect("checked by spec_pipe_ok");
            let mut b = vec![0u8; q * 4];
            self.backend.download(tok, 0, &mut b, self.stream)?;
            let mut v = vec![0u8; q * 4];
            self.backend.download(&self.ws.ids, 0, &mut v, self.stream)?;
            Ok((b, v, hn))
        })();
        let (b, v, hn) = match run {
            Ok(x) => x,
            Err(e) => {
                // the recurrent slots, the caches and the head's queue are part-way through the step
                let _ = st.reset(&*self.backend);
                if let Some(m) = st.mtp.as_mut() {
                    m.synced = false;
                }
                return Err(e);
            }
        };
        let word = |b: &[u8], i: usize| i32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]) as u32;
        let out: Vec<u32> = (0..q).map(|i| word(&b, i)).collect();
        let vids: Vec<u32> = (0..q).map(|i| word(&v, i)).collect();
        if vids[0] != next || vids.iter().chain(&out).any(|&t| t as usize >= self.dims.vocab) {
            let _ = st.reset(&*self.backend);
            if let Some(m) = st.mtp.as_mut() {
                m.synced = false;
            }
            return Err(Error::backend(format!("MTP pipelined step: device ids {vids:?} / {out:?} (fed {next})")));
        }
        let drafts = vids[1..].to_vec();
        // the next step's ids while the queue is empty (a later upload would wait for the rollback and the queue copy):
        // after j accepted drafts it is fed out[j] and its head catches up over d_1 .. d_j, out[j]
        let j = crate::spec::accept(&drafts, &out);
        let mut s = Staged { len: len0 + j + 1, next: out[j], catch: Vec::with_capacity(j + 1) };
        s.catch.extend_from_slice(&drafts[..j]);
        s.catch.push(out[j]);
        let staged = match self.pipe_stage(&s) {
            Ok(()) => Some(s),
            Err(_) => None,
        };
        st.commit_verify(&vids);
        if st.has_recurrent() {
            st.spec_rows.saved = q - 1;
        }
        self.telemetry.step();
        // the drafter queues every row; the caller's rollback drops the rejected ones
        self.mtp_after_forward(st, &hn, q, false, false)?;
        self.det_tap_flush(len0, q)?;
        Ok(PipePass { drafts, out, staged })
    }

    /// Submit the catch-up and chain of the next step (at `st`, fed `next`, `k` drafts) now, so the GPU drafts while the
    /// host emits the last pass's tokens. `staged`: the ids [`Model::spec_pipe_step`] left in `ws.ids` (uploaded again if
    /// they are not this step's). `None`: not applicable, nothing ran.
    pub(crate) fn spec_pipe_predraft(
        &self,
        st: &mut SequenceState,
        next: u32,
        k: usize,
        staged: Option<&Staged>,
    ) -> Result<Option<Predraft>> {
        if !self.spec_pipe_ok(st, next, k) {
            return Ok(None);
        }
        let n = st.mtp.as_ref().map_or(0, |m| m.pending_rows);
        let s = Staged { len: st.len, next, catch: Self::pipe_catch(st, next)? };
        if staged != Some(&s) {
            self.pipe_stage(&s)?;
        }
        self.pipe_draft(st, next, n, k)?;
        // commit what is encoded (not waited on): the GPU starts drafting now
        self.backend.fence(self.stream)?;
        Ok(Some(Predraft { len: st.len, next, k, n }))
    }
}
