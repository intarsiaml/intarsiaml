//! Batched decode step (continuous batching, Metal first): one decode token for each of several sequences, each with
//! its own KV cache, in one pass over the weights.
//!
//! Exactness is the contract: the output of row `r` is bitwise what a solo decode step of sequence `r` computes
//! (`Model::forward` with one token), whatever the other rows are. It holds because
//!  * every projection and the head of the pass runs on the backend's row-exact multi-row kernels (the G0 GEMVs, which
//!    are bitwise the one-row serving GEMV per row), selected by marking the ops `OpCall::exact_rows` exactly as a
//!    speculative verify pass does;
//!  * the row-parallel ops (embedding, norms, activation) have no row-count dependence, as the verify passes rely on;
//!  * attention is not batched at all: each row's q-norm / rope / KV append and its split-KV decode attention run as
//!    the ordinary one-row calls of the solo step ([`Model::attn_rows`], the same function), on that row's slice of the
//!    scratch and that row's own cache, at that row's own position. The `exact_rows` mark is off for those calls.
//!
//! The structure of one layer is therefore: norm (B rows) -> q / k / v projections (B rows, row-exact) -> per row:
//! `attn_rows` -> output projection (B rows, row-exact) -> norm -> FFN (B rows, row-exact).

use super::*;

impl Model {
    /// Most sequences any backend takes in one batched step (operator decisions: exact kernels only; Metal 8 rows, the
    /// multi-row GEMVs cost 1.2x a one-row launch at 8 rows and 2.6x at 16, docs/research/continuous-batching-design.md
    /// section 4; CUDA 16, 2026-10-23). A backend's own cap is [`Backend::batch_row_cap`].
    pub const BATCH_MAX_ROWS: usize = 16;

    /// Whether the backend registers row-exact kernels for the projections and the head (`Matmul`, `Logits`), which a
    /// batched step needs. Without them `exact_rows` would silently select the ordinary kernels, whose output depends
    /// on the row count. (The cpu backend registers none: its kernels compute every row with the one-row loop, which
    /// is exact by construction, so batching works there without them, as verify does.)
    pub fn batch_capable(&self) -> bool {
        let by_op = self.registry.by_op();
        [Op::Matmul, Op::Logits].iter().all(|op| by_op.get(op).is_some_and(|ks| ks.iter().any(|k| k.rows_exact)))
    }

    /// Opt in to batched decode of hybrid models: GatedDeltaNet (recurrent-state) blocks, mixture-of-experts blocks and
    /// gated attention (docs/results/2026-10-23-rb-recurrent-batching.md, docs/results/2026-10-23-rb2-cuda-batching.md).
    /// On the cpu backend every multi-row kernel computes each row with the one-row loop, so a batched row is the solo row
    /// by construction; any other backend needs registered row-exact kernels ([`Model::batch_capable`]: CUDA's `_rows`
    /// GEMVs and MoE decode kernels, proven bitwise the one-row kernels by `tests/rows_exact.rs`), and the plan check
    /// ([`crate::plan`], `check_batch_picks`) refuses whatever else would pick a different kernel from the solo step.
    /// Call before [`Model::enable_batch`]; without it `enable_batch` is exactly what it was.
    pub fn allow_recurrent_batch(&mut self) -> Result<()> {
        let ok = match self.backend.name() {
            "cpu" => true,
            "cuda" => self.batch_capable(),
            _ => false,
        };
        if !ok {
            return Err(Error::Unsupported(format!(
                "batched decode of recurrent / mixture-of-experts / gated-attention blocks is implemented for the cuda backend (and the cpu reference); the {} backend has no verified row-exact kernels for it",
                self.backend.name()
            )));
        }
        if self.batch_rows != 0 {
            return Err(Error::invalid("allow_recurrent_batch after enable_batch"));
        }
        self.recurrent_batch = true;
        Ok(())
    }

    /// Whether [`Model::allow_recurrent_batch`] was called.
    pub fn recurrent_batch(&self) -> bool {
        self.recurrent_batch
    }

    /// Enable batched decode steps of up to `max_rows` sequences (2..=[`Model::BATCH_MAX_ROWS`]). Call once, before
    /// [`Model::configure_plan`] and before any [`Model::new_state`]: plan resolution then also dry-runs
    /// [`crate::plan::Phase::Batch`] passes (every row count, at the start and the end of the context), so a batched op without a
    /// row-exact kernel fails the load, admission covers the batch kernels and the scratch workspace is sized for them.
    ///
    /// Memory: `max_rows` sequences each hold a KV cache of `plan.kv_cap_tokens`, so the per-sequence capacity shrinks
    /// to what `max_rows` caches fit in the KV budget the plan measured (`kv_budget_bytes / (max_rows *
    /// kv_bytes_per_token)`, with a plan note). Dense attention-only single-rank models only.
    pub fn enable_batch(&mut self, max_rows: usize) -> Result<()> {
        if !self.peers.is_empty() || self.tp.world > 1 || self.lm_head.is_none() {
            return Err(Error::Unsupported("batched decode needs a single-rank model".into()));
        }
        let hybrid = self.recurrent_batch;
        let dense = self.layers.iter().flatten().all(|b| match b {
            Block::Attn(a) => hybrid || !a.gated,
            Block::Ffn(_) => true,
            Block::Gdn(_) | Block::Moe(_) => hybrid,
            Block::Gemma(_) => false,
        });
        if !dense {
            return Err(Error::Unsupported(
                "batched decode supports dense attention-only models (no recurrent, mixture-of-experts or gated-attention blocks)".into(),
            ));
        }
        // without row-exact kernels `exact_rows` would silently pick the ordinary ones, whose bits depend on the row count
        if !self.batch_capable() && self.backend.name() != "cpu" {
            return Err(Error::Unsupported(format!(
                "batched decode needs row-exact multi-row kernels (Metal); the {} backend registers none",
                self.backend.name()
            )));
        }
        let most = self.backend.batch_row_cap().min(Self::BATCH_MAX_ROWS).min(LOGIT_ROWS).min(self.ws.max_tokens);
        if !(2..=most).contains(&max_rows) {
            return Err(Error::invalid(format!("batch rows {max_rows} outside 2..={most}")));
        }
        // Recurrent state: every slot holds a live copy and up to `RECURRENT_SNAPSHOTS` snapshots, all allocated when the
        // slot is created (`Model::new_state`). The plan's workspace already holds one sequence's worth of them; the other
        // `max_rows - 1` come out of the KV budget.
        let per_state = self.recurrent_bytes_per_state() * (1 + crate::state::RECURRENT_SNAPSHOTS as u64);
        let rec_extra = (max_rows as u64 - 1) * per_state;
        if rec_extra > 0 {
            self.plan.workspace_bytes += rec_extra;
            self.plan.notes.push(format!(
                "{} MiB of the KV budget reserved for the recurrent state of {max_rows} batched sequences (live + {} snapshots each, {} MiB per sequence)",
                rec_extra >> 20,
                crate::state::RECURRENT_SNAPSHOTS,
                per_state >> 20
            ));
        }
        let kv_budget = self.plan.kv_budget_bytes.saturating_sub(rec_extra);
        let bpt = self.kv_bytes_per_token();
        let mut cap = self.plan.kv_cap_tokens;
        if bpt > 0 {
            let by_budget = (kv_budget / (max_rows as u64 * bpt)) as usize;
            if by_budget < cap {
                self.plan.notes.push(format!(
                    "context clamped from {cap} to {by_budget} tokens per sequence: {max_rows} batched sequences share the KV budget"
                ));
                cap = by_budget;
            }
        }
        if cap < 16 {
            return Err(Error::Memory(format!(
                "{max_rows} batched sequences leave {cap} tokens of KV each ({} MiB budget, {bpt} bytes per token)",
                kv_budget >> 20
            )));
        }
        self.plan.kv_cap_tokens = cap;
        self.plan.kv_total_bytes = max_rows as u64 * cap as u64 * bpt;
        if self.batch_tok.is_none() {
            self.batch_tok = Some(self.backend.alloc(Self::BATCH_MAX_ROWS * 4, MemKind::Device)?);
        }
        self.batch_rows = max_rows;
        self.graphs.invalidate();
        Ok(())
    }

    /// Bytes of the recurrent (GatedDeltaNet) state of one sequence: what its live copy and each snapshot hold.
    pub fn recurrent_bytes_per_state(&self) -> u64 {
        self.slot_specs
            .iter()
            .map(|s| match s {
                SlotSpec::Recurrent { bytes } => *bytes as u64,
                _ => 0,
            })
            .sum()
    }

    /// Most sequences a batched step may have (0: batching not enabled).
    pub fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    /// The pre-GPU checks of one `forward_batch` row. A caller that wants a bad row to fail alone (the continuous-batching
    /// adapter) runs this per row first and passes only the rows that pass.
    pub fn check_batch_row(&self, st: &SequenceState, id: u32, mode: HeadMode) -> Result<()> {
        if !matches!(mode, HeadMode::LastLogits | HeadMode::LastArgmax) {
            return Err(Error::invalid("forward_batch rows return logits or a token"));
        }
        Self::check_forward(st, &[id], mode, false)?;
        if !st.peers.is_empty() || (st.has_recurrent() && !self.recurrent_batch) {
            return Err(Error::Unsupported("batched decode of a recurrent or tensor-parallel state".into()));
        }
        if id as usize >= self.dims.vocab {
            return Err(Error::BadRequest(format!("token id {id} >= vocab {}", self.dims.vocab)));
        }
        Ok(())
    }

    /// One decode token for each of several sequences in one batched pass: `ids[r]` is fed to `states[r]` at its
    /// current length, and `modes[r]` (`LastLogits` or `LastArgmax`) says what row `r` returns, in the form
    /// [`Model::forward`] returns for a one-token step (`StepOut::Logits` / `StepOut::Token`). Row `r`'s result is
    /// bitwise the solo step's, and every state ends exactly as after its solo step (one token committed).
    ///
    /// One to [`Model::batch_rows`] rows are allowed (a single row is the solo step itself). A failed pass can leave KV rows written past a state's length,
    /// which attention never reads and the next append overwrites; the engine treats it as fatal anyway.
    pub fn forward_batch(
        &self,
        states: &mut [&mut SequenceState],
        ids: &[u32],
        modes: &[HeadMode],
    ) -> Result<Vec<StepOut>> {
        let b = ids.len();
        if self.batch_rows == 0 {
            return Err(Error::invalid("forward_batch before enable_batch"));
        }
        if b == 0 || b > self.batch_rows || states.len() != b || modes.len() != b {
            return Err(Error::invalid(format!(
                "forward_batch of {b} ids, {} states, {} modes (enabled for 1..={})",
                states.len(),
                modes.len(),
                self.batch_rows
            )));
        }
        for r in 0..b {
            self.check_batch_row(states[r], ids[r], modes[r])?;
        }
        if b == 1 {
            return Ok(vec![self.forward(states[0], ids, modes[0], None)?]);
        }
        let tok = self.batch_tok.as_ref().ok_or_else(|| Error::invalid("forward_batch without its token buffer"))?;
        let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
        self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
        let run = (|| -> Result<()> {
            let refs: Vec<&SequenceState> = states.iter().map(|s| &**s).collect();
            let hn = self.run_layers(Seqs::Batch(&refs), ids, profile, None, true, None)?;
            self.batch_head(&hn, b, profile)
        })();
        self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        run?;

        let v = self.dims.vocab;
        let mut toks = vec![0u8; b * 4];
        if modes.contains(&HeadMode::LastArgmax) {
            self.backend.download(tok, 0, &mut toks, 0)?;
        }
        let mut out = Vec::with_capacity(b);
        for r in 0..b {
            out.push(match modes[r] {
                HeadMode::LastArgmax => {
                    let c = &toks[r * 4..r * 4 + 4];
                    StepOut::Token(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                }
                _ => {
                    let mut bytes = vec![0u8; v * 4];
                    self.backend.download(&self.ws.logits, r * v * 4, &mut bytes, 0)?;
                    StepOut::Logits(
                        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                    )
                }
            });
        }
        for (st, &id) in states.iter_mut().zip(ids) {
            st.commit(&[id]);
        }
        self.telemetry.step();
        Ok(out)
    }

    /// lm_head of rows `[0, b)` into `ws.logits` (`[b, V]` f32) and their argmax into `batch_tok` (`i32 [b]`), on the
    /// row-exact kernels (the `exact_rows` mark is on for the whole pass but the per-row attention).
    fn batch_head(&self, hn: &TensorRef, b: usize, profile: bool) -> Result<()> {
        let v = self.dims.vocab;
        let tok = self.batch_tok.as_ref().ok_or_else(|| Error::invalid("batch head without its buffer"))?;
        let logits = self.view(&self.ws.logits, DType::F32, &[b, v]);
        self.proj(&hn.rows(0, b), self.head()?, &logits, profile)?;
        self.exec(Op::Argmax, vec![logits], vec![self.view(tok, DType::I32, &[b])], OpAttrs::default(), profile)
    }

    /// Plan recording of a batched step ([`crate::plan::Phase::Batch`]): the real pass over `rows` phantom sequences of `pos0`
    /// tokens each, nothing launches. The `exact_rows` mark is set by the caller ([`Model::dry_run`]).
    pub(super) fn batch_probe(&self, rows: usize, pos0: usize) -> Result<()> {
        let sts: Vec<SequenceState> =
            (0..rows).map(|_| crate::plan::phantom_state(&self.slot_specs, self.plan.kv_cap_tokens, pos0)).collect();
        let refs: Vec<&SequenceState> = sts.iter().collect();
        let ids = vec![0u32; rows];
        let hn = self.run_layers(Seqs::Batch(&refs), &ids, false, None, true, None)?;
        self.batch_head(&hn, rows, false)
    }

    /// One attention block of a batched step: the q / k / v projections and the output projection run once on all
    /// rows of `n`, the attention core once per row against that row's own cache at that row's own length.
    pub(super) fn attn_block_batch(
        &self,
        a: &AttnBlock,
        sts: &[&SequenceState],
        n: &TensorRef,
        out: &TensorRef,
        kv_done: bool,
        profile: bool,
    ) -> Result<()> {
        let b = sts.len();
        let bufs = self.attn_bufs(b);
        // The gated block's split folded into the fused prologue, one call per row exactly as the solo step makes it
        // (`attn_block`): decided here, before the q projection the split follows. Evaluated with the row-exact mark off, as
        // the per-row attention calls below run. The calls read row `r` of the q+gate projection and write row `r` of the
        // gate buffer.
        // (only a gated block has one: an ungated block takes exactly the calls it always did)
        let prologues: Vec<Option<OpCall>> = if !a.gated {
            vec![None; b]
        } else {
            let mark = self.exact_rows.swap(false, std::sync::atomic::Ordering::Relaxed);
            let calls = sts
                .iter()
                .enumerate()
                .map(|(r, st)| {
                    let slot = Self::attn_slot(st, a).ok()?;
                    let mut call =
                        self.gate_qkv_call(a, slot, &bufs.row(r), 1, st.len, None).filter(|c| self.fused_ok(c))?;
                    let d = &self.ldims;
                    let (qg, gate) = self.gated_q_buffers(b);
                    call.inputs[0] = qg.reshaped(&[b, d.n_heads, 2 * d.head_dim]).rows(r, 1);
                    call.outputs[1] = gate.rows(r, 1);
                    Some(call)
                })
                .collect();
            self.exact_rows.store(mark, std::sync::atomic::Ordering::Relaxed);
            calls
        };
        // The solo step decides the fold per sequence. The rows share one projection call, so they must all decide alike:
        // a mixed decision would silently run some rows unfused where their solo step fuses (the shapes are the same for
        // every row today, so this is an assertion, not a fallback).
        let folded = a.gated && prologues.iter().all(Option::is_some);
        if a.gated && !folded && prologues.iter().any(Option::is_some) {
            return Err(Error::invalid(
                "batched gated-attention rows disagree on the fused gate-split prologue: a row's bits would depend on its neighbours",
            ));
        }
        let prologues: Vec<Option<OpCall>> = if folded { prologues } else { vec![None; b] };
        let (qc0, gate) = self.attn_project(a, n, b, kv_done, folded, profile)?;
        debug_assert!(
            gate.is_none() || self.recurrent_batch,
            "enable_batch rejects gated attention without the hybrid opt-in"
        );
        // the one-row calls are the solo step's: not row-exact kernels, whose row count would be wrong
        self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        let rows = (|| -> Result<usize> {
            let mut qc_end = None;
            for ((r, st), prologue) in sts.iter().enumerate().zip(prologues) {
                let slot = Self::attn_slot(st, a)?;
                let (_, qc) = self.attn_rows(a, slot, &bufs.row(r), qc0, 1, st.len, None, prologue, profile)?;
                if *qc_end.get_or_insert(qc) != qc {
                    return Err(Error::invalid("batched rows chose different attention paths"));
                }
            }
            Ok(qc_end.unwrap_or(qc0))
        })();
        self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
        let qc = rows?;
        let attn_out = bufs.q[1 - qc].clone();
        // the output gate (gated attention, hybrid opt-in only; `None` otherwise) is applied once for all rows, as in the solo step
        self.attn_out_proj(a, &attn_out, gate, qc, out, b, profile)
    }
}
