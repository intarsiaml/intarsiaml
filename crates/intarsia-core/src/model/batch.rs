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
    /// ([`crate::plan`], `check_batch_ops`) refuses whatever else would pick a different kernel from the solo step.
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
        let per_state = self.recurrent_bytes_per_state() * (1 + self.batch_snapshots() as u64);
        let rec_extra = (max_rows as u64 - 1) * per_state;
        if rec_extra > 0 {
            self.plan.workspace_bytes += rec_extra;
            self.plan.notes.push(format!(
                "{} MiB of the KV budget reserved for the recurrent state of {max_rows} batched sequences (live + {} snapshots each, {} MiB per sequence)",
                rec_extra >> 20,
                self.batch_snapshots(),
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
        // Row tables (R3a): one launch per layer op for all rows instead of one per row, where the backend has the kernels
        // and every op of the step has a registered row-table twin. `INTARSIA_ROW_TABLES=0` keeps the per-row launches.
        if self.row_tab.is_none() && self.row_tables_wanted() {
            self.row_tab = Some(self.backend.alloc(Self::BATCH_MAX_ROWS * 2 * self.slot_specs.len().max(1) * 8, MemKind::Device)?);
            self.batch_pos = Some(self.backend.alloc(Self::BATCH_MAX_ROWS * 4, MemKind::Device)?);
        }
        if self.topk.is_none() && self.backend.supports_topk() && std::env::var("INTARSIA_DEVICE_SAMPLING").is_ok_and(|v| v == "1") {
            let m = Self::BATCH_MAX_ROWS;
            self.topk = Some(TopkBufs {
                ids: self.backend.alloc(m * Self::TOPK_MAX * 4, MemKind::Device)?,
                vals: self.backend.alloc(m * Self::TOPK_MAX * 4, MemKind::Device)?,
                nan: self.backend.alloc(m * 4, MemKind::Device)?,
                scratch: self.backend.alloc(m * 32 * Self::TOPK_MAX * 8, MemKind::Device)?,
            });
        }
        self.batch_rows = max_rows;
        self.graphs.invalidate();
        Ok(())
    }

    /// Whether this model's batched steps use row tables: the backend supports them, every row-table op of the model's blocks has
    /// a registered kernel, and `INTARSIA_ROW_TABLES` is not `0`.
    fn row_tables_wanted(&self) -> bool {
        if !self.backend.supports_row_tables() || std::env::var("INTARSIA_ROW_TABLES").is_ok_and(|v| v == "0") {
            return false;
        }
        let by_op = self.registry.by_op();
        let has = |op: Op| by_op.get(&op).is_some_and(|ks| ks.iter().any(|k| k.row_table));
        let attn = self.layers.iter().flatten().any(|b| matches!(b, Block::Attn(_)));
        let gdn = self.layers.iter().flatten().any(|b| matches!(b, Block::Gdn(_)));
        (!attn || (has(Op::KvAppend) && has(Op::AttnDecode))) && (!gdn || (has(Op::GdnConv) && has(Op::GdnDeltaRule)))
    }

    /// Write the row table and the position vector of a batched step (`states[r]` is row `r`): the device address of every
    /// state buffer of every row, and each row's current length. Stream-ordered before the step's launches.
    pub(super) fn upload_row_table(&self, states: &[&SequenceState]) -> Result<()> {
        let (Some(tab), Some(pos)) = (&self.row_tab, &self.batch_pos) else { return Ok(()) };
        let stride = 2 * self.slot_specs.len().max(1);
        let addr = |b: &Buffer| -> Result<u64> {
            self.backend.device_address(b).ok_or_else(|| Error::invalid("backend gave no device address for a row table"))
        };
        let mut t = vec![0u8; states.len() * stride * 8];
        let mut p = Vec::with_capacity(states.len() * 4);
        for (r, st) in states.iter().enumerate() {
            for (j, slot) in st.slots.iter().enumerate() {
                let (a0, a1) = match slot {
                    StateSlot::AttnKv(kv) => (addr(&kv.k)?, addr(&kv.v)?),
                    StateSlot::Recurrent(rec) => (addr(&rec.buf)?, 0),
                };
                let o = (r * stride + 2 * j) * 8;
                t[o..o + 8].copy_from_slice(&a0.to_le_bytes());
                t[o + 8..o + 16].copy_from_slice(&a1.to_le_bytes());
            }
            p.extend_from_slice(&(st.len as i32).to_le_bytes());
        }
        self.backend.upload(tab, 0, &t, self.stream)?;
        self.backend.upload(pos, 0, &p, self.stream)
    }

    /// Row table of a call over `rows` rows whose state is slot `slot`: entry `2 * slot + sub`.
    pub(super) fn row_tab_for(&self, rows: usize, slot: usize, sub: usize, byte_off: usize) -> Option<crate::ops::RowTab> {
        let tab = self.row_tab.as_ref()?;
        let stride = 2 * self.slot_specs.len().max(1);
        Some(crate::ops::RowTab {
            ptrs: self.view(tab, DType::U8, &[rows, stride, 8]),
            rows,
            stride,
            index: 2 * slot + sub,
            byte_off,
        })
    }

    /// Recurrent-state snapshots each batched slot keeps for prefix reuse. The cpu reference keeps the solo schedule's
    /// `RECURRENT_SNAPSHOTS` (its prefix reuse is part of what its tests prove); on every other backend the prefix store stays
    /// at batch 1 (decision 2026-10-23 06:41): no snapshots, no reuse, so a slot costs one live state (63 MB on the 35B).
    pub fn batch_snapshots(&self) -> usize {
        if self.backend.name() == "cpu" {
            crate::state::RECURRENT_SNAPSHOTS
        } else {
            0
        }
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
        let topk_ok = match mode {
            HeadMode::LastTopK(k) => self.topk.is_some() && (1..=Self::TOPK_MAX).contains(&k) && k < self.dims.vocab,
            _ => false,
        };
        if !matches!(mode, HeadMode::LastLogits | HeadMode::LastArgmax) && !topk_ok {
            return Err(Error::invalid("forward_batch rows return logits, a token or (device sampling) the top-k"));
        }
        Self::check_forward(st, &[id], if topk_ok { HeadMode::LastLogits } else { mode }, false)?;
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
        // one row is the solo step itself (its graph is kept per sequence state, so alternating single rows do not re-record)
        if b == 1 {
            // the solo step has no device top-k: the whole row comes back and the engine samples it on the host, which is the
            // same token (`Seq::token_from` takes `Logits` for a sampled request)
            let mode = if matches!(modes[0], HeadMode::LastTopK(_)) { HeadMode::LastLogits } else { modes[0] };
            return Ok(vec![self.forward(states[0], ids, mode, None)?]);
        }
        let tok = self.batch_tok.as_ref().ok_or_else(|| Error::invalid("forward_batch without its token buffer"))?;
        let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
        self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
        let run = (|| -> Result<()> {
            let refs: Vec<&SequenceState> = states.iter().map(|s| &**s).collect();
            if !profile && self.graph_batch_step(&refs, ids)? {
                return Ok(());
            }
            self.upload_row_table(&refs)?;
            let hn = self.run_layers(Seqs::Batch(&refs), ids, profile, None, true, None)?;
            self.batch_head(&hn, b, profile)
        })();
        self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        run?;

        let v = self.dims.vocab;
        let mut toks = vec![0u8; b * 4];
        if modes.contains(&HeadMode::LastArgmax) {
            self.backend.download(tok, 0, &mut toks, self.stream)?;
        }
        // device-side sampling (R3b): the top-k of every row that asked, one kernel for the step, three small downloads
        let kmax = modes.iter().filter_map(|m| if let HeadMode::LastTopK(k) = m { Some(*k) } else { None }).max().unwrap_or(0);
        let top = if kmax > 0 { Some(self.batch_topk(b, kmax)?) } else { None };
        // the rows that want the whole logits row come back in ONE download of the `[b, V]` block when several do (rows are
        // contiguous in `ws.logits`), not one transfer per row
        let want: Vec<usize> = (0..b).filter(|&r| modes[r] == HeadMode::LastLogits || top.as_ref().is_some_and(|t| t.2[r] != 0 && matches!(modes[r], HeadMode::LastTopK(_)))).collect();
        // one download per run of consecutive wanted rows (rows are contiguous in `ws.logits`)
        let mut runs: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut i = 0;
        while i < want.len() {
            let mut j = i;
            while j + 1 < want.len() && want[j + 1] == want[j] + 1 {
                j += 1;
            }
            let (lo, n) = (want[i], want[j] + 1 - want[i]);
            let mut bytes = vec![0u8; n * v * 4];
            self.backend.download(&self.ws.logits, lo * v * 4, &mut bytes, self.stream)?;
            runs.push((lo, bytes));
            i = j + 1;
        }
        let mut out = Vec::with_capacity(b);
        for r in 0..b {
            out.push(match modes[r] {
                HeadMode::LastArgmax => {
                    let c = &toks[r * 4..r * 4 + 4];
                    StepOut::Token(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                }
                HeadMode::LastTopK(k) if top.as_ref().is_some_and(|t| t.2[r] == 0) => {
                    let (ids, vals, _) = top.as_ref().expect("checked");
                    StepOut::TopK { ids: ids[r * kmax..r * kmax + k].to_vec(), vals: vals[r * kmax..r * kmax + k].to_vec() }
                }
                _ => {
                    // LastLogits, or a top-k row whose logits hold a NaN (the host sampler's comparator path decides then)
                    let (lo, run) = runs.iter().rev().find(|(lo, _)| *lo <= r).expect("row was downloaded");
                    let bytes = &run[(r - lo) * v * 4..(r - lo + 1) * v * 4];
                    StepOut::Logits(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
                }
            });
        }
        for (st, &id) in states.iter_mut().zip(ids) {
            st.commit(&[id]);
        }
        self.telemetry.step();
        Ok(out)
    }

    /// Most candidates device-side sampling returns per row.
    pub const TOPK_MAX: usize = 64;

    /// Top-`kmax` of rows `[0, b)` of `ws.logits` on the device; returns (ids, values, NaN flag) of every row, `kmax` per row.
    fn batch_topk(&self, b: usize, kmax: usize) -> Result<(Vec<u32>, Vec<f32>, Vec<i32>)> {
        let t = self.topk.as_ref().ok_or_else(|| Error::invalid("device-side sampling is off"))?;
        let v = self.dims.vocab;
        let logits = self.view(&self.ws.logits, DType::F32, &[b, v]);
        let (ids, vals, nan) =
            (self.view(&t.ids, DType::I32, &[b, kmax]), self.view(&t.vals, DType::F32, &[b, kmax]), self.view(&t.nan, DType::I32, &[b]));
        let need = b * 32 * kmax * 8;
        let scratch = self.view(&t.scratch, DType::U8, &[need]);
        self.backend.topk_rows(&logits, kmax, &ids, &vals, &nan, &scratch, self.stream)?;
        let (mut i, mut x, mut n) = (vec![0u8; b * kmax * 4], vec![0u8; b * kmax * 4], vec![0u8; b * 4]);
        self.backend.download(&t.ids, 0, &mut i, self.stream)?;
        self.backend.download(&t.vals, 0, &mut x, self.stream)?;
        self.backend.download(&t.nan, 0, &mut n, self.stream)?;
        let f = |c: &[u8]| [c[0], c[1], c[2], c[3]];
        Ok((
            i.chunks_exact(4).map(|c| i32::from_le_bytes(f(c)) as u32).collect(),
            x.chunks_exact(4).map(|c| f32::from_le_bytes(f(c))).collect(),
            n.chunks_exact(4).map(|c| i32::from_le_bytes(f(c))).collect(),
        ))
    }

    /// Whether batched rows may ask for device-side top-k sampling (`HeadMode::LastTopK`).
    pub fn device_sampling(&self) -> bool {
        self.topk.is_some()
    }

    /// lm_head of rows `[0, b)` into `ws.logits` (`[b, V]` f32) and their argmax into `batch_tok` (`i32 [b]`), on the
    /// row-exact kernels (the `exact_rows` mark is on for the whole pass but the per-row attention).
    pub(super) fn batch_head(&self, hn: &TensorRef, b: usize, profile: bool) -> Result<()> {
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
        // Row tables: the whole per-row chain (q / k norm, rope, K and V append, split-KV decode) as one launch per op for all
        // rows, each on its own cache at its own position (R3a). Not for the folded gated prologue or a scaled rope.
        if !folded {
            if let Some(qc) = self.attn_batch_rows(a, sts, &bufs, qc0, profile)? {
                let attn_out = bufs.q[1 - qc].clone();
                return self.attn_out_proj(a, &attn_out, gate, qc, out, b, profile);
            }
        }
        // the one-row calls are the solo step's: not row-exact kernels, whose row count would be wrong
        self.mark_state_bound();
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

    /// The attention chain of [`Model::attn_rows_core`] (unfolded: q-norm, rope, k-norm, rope, K append, V append, decode) over all
    /// rows of a batched step as row-table calls. Every op is the solo kernel's arithmetic: the norms and the rope are row-local
    /// (rope reads one position per row), the append and the decode attention take each row's cache from the table and its
    /// position from `batch_pos`. `None` (nothing launched) where a call has no kernel, e.g. a scaled rope, a sliding window or an
    /// fp8 cache. Returns the final `qc`: the attention output is `bufs.q[1 - qc]`.
    fn attn_batch_rows(
        &self,
        a: &AttnBlock,
        sts: &[&SequenceState],
        bufs: &AttnBufs,
        mut qc: usize,
        profile: bool,
    ) -> Result<Option<usize>> {
        let Some(pos) = self.batch_pos.as_ref() else { return Ok(None) };
        if self.rope_table.is_some() {
            self.row_table_fallback("attention (scaled rope)");
            return Ok(None);
        }
        let b = sts.len();
        let slot = Self::attn_slot(sts[0], a)?;
        let d = &self.ldims;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let pos = self.view(pos, DType::I32, &[b]);
        let cache_shape = [nkv, slot.cap, hd];
        let (kcache, vcache) = (self.view(&slot.k, slot.dtype, &cache_shape), self.view(&slot.v, slot.dtype, &cache_shape));
        let mk = |sub: usize| self.row_tab_for(b, a.slot, sub, 0);
        let eps = OpAttrs { eps: d.rms_eps, w_offset: self.recipe.model.norm_offset, ..Default::default() };
        let rope = OpAttrs { theta: d.rope_theta, head_dim: hd, rotary_dim: a.rotary_dim, ..Default::default() };
        let append = OpAttrs { kv_cap: slot.cap, n_kv_heads: nkv, head_dim: hd, ..Default::default() };
        let attrs = OpAttrs {
            scale: 1.0 / (hd as f32).sqrt(),
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            kv_cap: slot.cap,
            ..Default::default()
        };
        let row_call = |op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, at: OpAttrs, sub: usize| -> Option<OpCall> {
            let mut c = OpCall::new(op, inputs, outputs, at);
            c.pos = Some(pos.clone());
            c.rows = mk(sub);
            c.rows.as_ref()?;
            Some(c)
        };
        // the chain, planned before anything launches so a missing kernel leaves the per-row path untouched; the buffer parities
        // are those of `attn_rows_core` (q alternates through norm and rope, k through norm and rope, v is untouched)
        let (qn, kn) = (a.q_norm.is_some() as usize, a.k_norm.is_some() as usize);
        let qf = (qc + qn + 1) % 2;
        let kf = (kn + 1) % 2;
        let calls = (
            row_call(Op::KvAppend, vec![bufs.k[kf].clone()], vec![kcache.clone()], append.clone(), 0),
            row_call(Op::KvAppend, vec![bufs.v.clone()], vec![vcache.clone()], append, 1),
            row_call(
                Op::AttnDecode,
                vec![bufs.q[qf].clone(), kcache, vcache],
                vec![bufs.q[1 - qf].clone()],
                attrs,
                0,
            ),
        );
        let (Some(ka), Some(va), Some(at)) = calls else { return Ok(None) };
        let mut rope_c = OpCall::new(Op::Rope, vec![bufs.q[0].clone()], vec![bufs.q[1].clone()], rope.clone());
        rope_c.pos = Some(pos.clone());
        for c in [&ka, &va, &at, &rope_c] {
            if let Err(e) = c.check().and_then(|_| self.registry.select(c).map(|_| ())) {
                self.row_table_fallback(&format!("attention {}: {e}", c.op));
                return Ok(None);
            }
        }
        if let Some(w) = &a.q_norm {
            self.exec(Op::QkNorm, vec![bufs.q[qc].clone(), w.clone()], vec![bufs.q[1 - qc].clone()], eps.clone(), profile)?;
            qc = 1 - qc;
        }
        self.exec_pos(Op::Rope, vec![bufs.q[qc].clone()], vec![bufs.q[1 - qc].clone()], rope.clone(), Some(&pos), profile)?;
        qc = 1 - qc;
        let mut kb = 0usize;
        if let Some(w) = &a.k_norm {
            self.exec(Op::QkNorm, vec![bufs.k[kb].clone(), w.clone()], vec![bufs.k[1 - kb].clone()], eps, profile)?;
            kb = 1 - kb;
        }
        self.exec_pos(Op::Rope, vec![bufs.k[kb].clone()], vec![bufs.k[1 - kb].clone()], rope, Some(&pos), profile)?;
        debug_assert_eq!(qc, qf);
        self.exec_call(ka, profile)?;
        self.exec_call(va, profile)?;
        self.exec_call(at, profile)?;
        Ok(Some(qc))
    }

    /// A batched step is about to launch something bound to one sequence's buffers (a per-row fallback): a recording being made
    /// right now cannot serve other sequences.
    pub(super) fn mark_state_bound(&self) {
        if self.capture_ops.lock().unwrap().is_some() {
            self.graphs.state_bound.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// A batched block fell back to its per-row launches although row tables are on (a call without kernel): said once per reason.
    pub(super) fn row_table_fallback(&self, what: &str) {
        static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let mut seen = SEEN.lock().unwrap();
        if !seen.iter().any(|w| w == what) {
            eprintln!("batched step: row-table path unavailable for {what}; per-row launches");
            seen.push(what.to_string());
        }
    }

    /// The decode twin (R3d): a second `Model` over the SAME weights, kernels and plan, with its own small workspace, kernel
    /// scratch, recorded graphs, row tables and sampling buffers, enqueueing on stream 1 (the backend's high-priority lane). A
    /// batched decode step on the twin and a prefill chunk on this model touch disjoint buffers (the twin never reads or writes
    /// this model's workspace, and a sequence is either prefilling or decoding), so the two can be in flight at once on two
    /// threads; each is the very computation it would be alone, bit for bit. `rows`: batched rows the twin runs.
    pub fn decode_twin(&self, rows: usize) -> Result<Model> {
        if self.batch_rows == 0 || rows == 0 || rows > self.batch_rows {
            return Err(Error::invalid("decode_twin needs a batch-enabled model and 1..=batch_rows rows"));
        }
        if !self.peers.is_empty() || self.tp.world > 1 || self.mtp.is_some() || self.gemma.is_some() {
            return Err(Error::Unsupported("decode twin: single-rank models without MTP or Gemma 4 layers".into()));
        }
        let (d, act) = (&self.ldims, self.act);
        let (qdim, kvdim, esz) = (d.n_heads * d.head_dim, d.n_kv_heads * d.head_dim, act.size_bytes());
        let mt = rows.max(Self::BATCH_MAX_ROWS);
        let mut bytes = 0u64;
        let mut wb = |n: usize| -> Result<Buffer> {
            bytes += n as u64;
            self.backend.alloc(n.max(16), MemKind::Device)
        };
        let (has_gdn, has_gated) = self.layers.iter().flatten().fold((false, false), |(g, a), blk| match blk {
            Block::Gdn(_) => (true, a),
            Block::Attn(x) => (g, a || x.gated),
            _ => (g, a),
        });
        let x = gdn_block::alloc_extra(&mut wb, mt, esz, d, has_gdn, has_gated)?;
        let moe = match d.moe.as_ref().filter(|_| self.layers.iter().flatten().any(|bl| matches!(bl, Block::Moe(_)))) {
            Some(mc) => Some(MoeScratch::new(&*self.backend, mc, mt, act)?),
            None => None,
        };
        let ws = Workspace {
            max_tokens: mt,
            ids: wb(mt * 4)?,
            h: [wb(mt * d.hidden * esz)?, wb(mt * d.hidden * esz)?],
            n: wb(mt * d.hidden * esz)?,
            blk: wb(mt * d.hidden * esz)?,
            q: [wb(mt * qdim * esz)?, wb(mt * qdim * esz)?],
            k: [wb(mt * kvdim * esz)?, wb(mt * kvdim * esz)?],
            v: wb(mt * kvdim * esz)?,
            g: wb(mt * d.intermediate * esz)?,
            u: wb(mt * d.intermediate * esz)?,
            a: wb(mt * d.intermediate * esz)?,
            logits: wb(LOGIT_ROWS * d.vocab * 4)?,
            tok: wb(16)?,
            part: None,
            pos: wb(16)?,
            x,
            moe,
        };
        let mut registry = Registry::new(self.backend.name(), &self.backend.info(), self.backend.kernels());
        registry.adopt_status(&self.registry);
        let mut plan = self.plan.clone();
        plan.workspace_bytes = bytes;
        let mut t = Model {
            backend: self.backend.clone(),
            registry,
            dims: self.dims.clone(),
            ldims: self.ldims.clone(),
            recipe: self.recipe.clone(),
            act: self.act,
            kv: self.kv,
            plan,
            weights_bytes: self.weights_bytes,
            embed: self.embed.clone(),
            final_norm: self.final_norm.clone(),
            lm_head: self.lm_head.clone(),
            layers: self.layers.clone(),
            gemma: None,
            slot_specs: self.slot_specs.clone(),
            ws,
            telemetry: self.telemetry.clone(),
            profile_every: 0,
            scratch: None,
            recorder: Mutex::new(None),
            resolved: Default::default(),
            graphs: GraphState::new(self.graphs.configured()),
            capture_ops: Mutex::new(None),
            capture_comms: std::sync::atomic::AtomicU64::new(0),
            rope_table: self.rope_table.clone(),
            tp_fusion: false,
            tp: self.tp.clone(),
            no_comm: self.no_comm,
            test_bug: self.test_bug.clone(),
            peers: Vec::new(),
            remote: None,
            exact_rows: std::sync::atomic::AtomicBool::new(false),
            verify_tok: None,
            verify_rows: 0,
            batch_rows: 0,
            recurrent_batch: self.recurrent_batch,
            batch_tok: None,
            stream: 1,
            row_tab: None,
            batch_pos: None,
            topk: None,
            token_feedback: std::sync::atomic::AtomicBool::new(false),
            feedback_hold: self.feedback_hold,
            tail_merge: 0,
            skip_dead: std::sync::atomic::AtomicBool::new(false),
            tail_dead: std::sync::atomic::AtomicBool::new(false),
            mtp: None,
            mtp_k: self.mtp_k,
        };
        t.enable_batch(rows)?;
        crate::plan::resolve(&mut t)?;
        Ok(t)
    }

    /// A stand-in for a slot whose state is away on another thread (phantom buffers: no device memory).
    pub fn placeholder_state(&self) -> SequenceState {
        crate::plan::phantom_state(&self.slot_specs, self.plan.kv_cap_tokens, 0)
    }
}
