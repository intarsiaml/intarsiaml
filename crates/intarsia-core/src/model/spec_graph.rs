//! The speculative MTP step as one captured graph (`--spec mtp`, and the passes the MTP head drafts inside `--spec auto`, on a
//! backend that records graphs and runs the verify attention one decode call per row: HIP).
//!
//! One step is the draft and the verify together, from the token the main model just emitted to the verify outputs:
//!
//! ```text
//!   catch-up   the head over the n queued hidden rows (tokens known on the host, uploaded)     -> draft 1 (argmax -> ws.ids[1])
//!   chain      k - 1 one-row head passes, each reading the previous draft where its argmax left it -> drafts 2..k
//!   verify     the main model over [next, d_1 .. d_k] (k + 1 rows, every op on its row-exact kernel) -> verify_tok
//!   queue      the verify pass's hidden rows into the head's pending queue (the rejected ones are cut by the rollback)
//! ```
//!
//! The recording is valid at every position because nothing in it names one: the tokens come from `ws.ids`, and every
//! position-dependent launch (rope, KV append, attention) reads its row's position from the table `ws.pos` (a run of
//! consecutive positions per region, rewritten before each replay: [`POS_VERIFY`], [`POS_CATCH_UP`], [`POS_CHAIN`]). The
//! catch-up row count `n` and the verify row count `q = k + 1` fix the shapes, so a recording is kept per
//! (sequence state, n, q). Everything else a step touches has a fixed address: the workspace, the sequence's KV and
//! recurrent buffers, the per-row recurrent copies (`SpecRows`), the head's KV cache and its pending queue.
//!
//! Output is what the eager draft + verify passes produce, bit for bit: the same launches in the same order on the same
//! kernels, only recorded (the head runs on the ordinary kernels, the verify pass on the row-exact ones). Where that needs
//! care is attention: the recording cannot name a position, so it runs the head's catch-up rows and the verify rows as one
//! single-row decode call per row, each at its device position. A backend whose multi-row attention is not bitwise the
//! decode step's per row ([`Backend::verify_attn_per_row`], HIP) therefore runs the eager head's multi-row catch-up
//! (`n >= 2`) the same way (`Model::head_rows_attn`), and the eager verify pass too; on the other backends the multi-row
//! call is bitwise the per-row ones, so both paths agree without it. Acceptance and rollback stay on the host after the one
//! download of the outputs and the drafts.

use super::*;
use std::sync::atomic::Ordering;

use super::graph_step::Captured;

/// First entry of the catch-up token ids in `ws.pos` (the verify ids are entries `0..q` of `ws.ids`, the drafts among them).
const CATCH_IDS: usize = 48;
/// First entry of the scripted drafts in `ws.pos` (tests: [`Model::mtp_script`] replaces the drafts the verify pass takes).
const SCRIPT_IDS: usize = 64;
/// First entry of the id gather's receive buffer in `ws.pos` (tensor parallelism, lane gt2): `world` segments of the verify ids
/// rounded up to 4 entries (16 bytes, the HIP gather's quantum), so at two ranks up to 8 verify rows.
const GATHER_AT: usize = 80;
const GATHER_ENTRIES: usize = POS_TAB_ENTRIES - GATHER_AT;

/// `INTARSIA_SPEC_GRAPH` (`0` turns the captured step off), read once.
fn spec_graph_env_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_SPEC_GRAPH").map_or(true, |v| v != "0"))
}

/// Consecutive capture failures after which captured speculative steps stay off (until [`Model::set_graphs`]`(true)` resets the count):
/// the first failure may be transient (a memory spike during capture), the second is taken as the backend's answer.
const SPEC_CAPTURE_FAILURES_MAX: usize = 2;

/// What one captured speculative step returns.
pub struct SpecPass {
    /// The drafts `d_1..d_k` the head produced.
    pub drafts: Vec<u32>,
    /// The verify pass's greedy token of each of the `k + 1` rows.
    pub out: Vec<u32>,
}

impl Model {
    /// Whether [`Model::spec_step`] can run `k` drafts on `st` now (else the caller drafts and verifies eagerly).
    pub fn spec_step_ok(&self, st: &SequenceState, next: u32, k: usize) -> bool {
        let Some(m) = st.mtp.as_ref() else { return false };
        let q = k + 1;
        let n = m.pending_rows;
        k >= 1
            && self.graphs.usable(self)
            && self.graphs.spec_fail.load(Ordering::Relaxed) < SPEC_CAPTURE_FAILURES_MAX
            && self.backend.verify_attn_per_row()
            && self.mtp.is_some()
            // scripted drafts (tests) the recording cannot take: fewer than k, the eager path verifies exactly those
            && self.scripted_drafts(st, next, k).map_or(true, |d| d.len() == k)
            && self.recorder.lock().unwrap().is_none()
            && self.spec_step_tp_ok(st, n, k)
            && st.id != 0
            && (2..=self.verify_rows).contains(&q)
            && self.verify_tok.is_some()
            && self.ws.max_tokens >= q
            && m.active
            && m.synced
            && n >= 1
            && n <= q
            && m.kv.len + n == st.len
            // the verify rows, and the head's chained rows (positions st.len .. st.len + k - 1), fit their caches
            && st.len + q <= st.cap
            && st.len + k <= m.kv.cap
            && spec_graph_env_on()
            && !(self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0)
    }

    /// The tensor-parallel part of [`Model::spec_step_ok`] (lane gt2): true at one rank; with in-process peers only under
    /// `INTARSIA_SPEC_GRAPH_TP=1`, every peer able to record and replay, its state in step with rank 0's (same length; with a
    /// sharded head, the same queue and head cache), and the verify ids small enough for the in-graph id gather
    /// ([`GATHER_AT`]). A rank in another process (`tp-worker`) keeps the eager step.
    fn spec_step_tp_ok(&self, st: &SequenceState, n: usize, k: usize) -> bool {
        if self.tp.world == 1 {
            return true;
        }
        let q = k + 1;
        let g = q.next_multiple_of(4);
        let whole = self.mtp.as_ref().is_some_and(|h| h.whole());
        spec_graph_tp_env_on()
            && self.remote.is_none()
            && !self.peers.is_empty()
            && self.peers.len() + 1 == self.tp.world
            && st.peers.len() == self.peers.len()
            && self.tp.world * g <= GATHER_ENTRIES
            && self.ws.max_tokens >= g
            && self.peers.iter().zip(&st.peers).all(|(pm, ps)| {
                let p = &ps.st;
                pm.graphs.usable(pm)
                    && pm.recorder.lock().unwrap().is_none()
                    && pm.ws.max_tokens >= g
                    && p.id != 0
                    && p.len == st.len
                    && p.len + q <= p.cap
                    && match (pm.mtp.as_ref(), p.mtp.as_ref()) {
                        // a head whole on rank 0 (lane mt2): the peers hold none
                        (None, None) => whole,
                        // a sharded head: the peer's share of it is where rank 0's is
                        (Some(_), Some(pm_)) => {
                            !whole
                                && pm_.active
                                && pm_.synced
                                && pm_.pending_rows == n
                                && pm_.kv.len + n == p.len
                                && p.len + k <= pm_.kv.cap
                        }
                        _ => false,
                    }
            })
    }

    /// The step's ids from rank 0 to every rank inside a recording (lane gt2): all-gather `ws.ids[0..g]` into the spare
    /// entries of the position table ([`GATHER_AT`]), and a peer copies rank 0's segment over its own `ws.ids[0..g]`. A
    /// capturable collective at a fixed address, so the drafts rank 0's argmax leaves on its device reach the peers' head and
    /// verify passes without a host round trip. Both ranks call it at the same points of the step (the collectives pair up).
    fn spec_gather_ids(&self, g: usize) -> Result<()> {
        let send = self.view(&self.ws.ids, DType::I32, &[g]);
        let mut recv = self.view(&self.ws.pos, DType::I32, &[self.tp.world * g]);
        recv.offset = 4 * GATHER_AT;
        self.all_gather(&send, &recv, false)?;
        if self.tp.rank != 0 {
            self.backend.copy(&self.ws.ids, 0, &self.ws.pos, 4 * GATHER_AT, 4 * g, self.stream)?;
        }
        Ok(())
    }

    /// A peer's share of the captured step (lane gt2), recorded for (`st`, `n` catch-up rows, `k` drafts): with a sharded head
    /// its catch-up and chained passes (their all-reduces pair with rank 0's; each chained pass reads the draft rank 0's argmax
    /// left, gathered), then the gather of the verify ids, the verify pass's layers on the row-exact kernels (and its slice of a
    /// vocab-parallel head) and, with a sharded head, the rows into the head's queue: what [`Model::mtp_draft`] and
    /// [`Model::forward_verify`] run on this rank eagerly. Executes nothing.
    fn capture_spec_peer(&self, st: &SequenceState, n: usize, k: usize) -> Result<Captured> {
        let h = self.dims.hidden;
        let q = k + 1;
        let g = q.next_multiple_of(4);
        let ws = &self.ws;
        let ids_at = |i: usize, len: usize| {
            let mut v = self.view(&ws.ids, DType::I32, &[len]);
            v.offset = 4 * i;
            v
        };
        let pos_at = |i: usize| {
            let mut v = self.view(&ws.pos, DType::I32, &[1]);
            v.offset = 4 * i;
            v
        };
        let head = match (self.mtp.as_ref(), st.mtp.as_ref()) {
            (Some(h), Some(m)) => Some((h, m)),
            _ => None,
        };
        *self.capture_ops.lock().unwrap() = Some(Vec::new());
        self.capture_comms.store(0, Ordering::Relaxed);
        let graph = self.backend.graph_capture(self.stream, &mut || {
            if let Some((head, m)) = head {
                let pending = TensorRef::new(m.pending.clone(), self.act, &[m.cap_rows, h]);
                let mut tab_ids = self.view(&ws.pos, DType::I32, &[n]);
                tab_ids.offset = 4 * CATCH_IDS;
                let out = self.mtp_layer_at(head, &m.kv, n, None, &pending.rows(0, n), Some(&tab_ids), Some(&pos_at(POS_CATCH_UP)), false)?;
                self.backend.copy(&head.chain, 0, &out.buf, out.offset + (n - 1) * m.row_bytes, m.row_bytes, self.stream)?;
                for i in 1..k {
                    self.spec_gather_ids(g)?;
                    let hid = TensorRef::new(head.chain.clone(), self.act, &[1, h]);
                    let out = self.mtp_layer_at(head, &m.kv, 1, None, &hid, Some(&ids_at(i, 1)), Some(&pos_at(POS_CHAIN + i - 1)), false)?;
                    self.backend.copy(&head.chain, 0, &out.buf, out.offset, m.row_bytes, self.stream)?;
                }
            }
            self.spec_gather_ids(g)?;
            self.exact_rows.store(true, Ordering::Relaxed);
            let run = (|| -> Result<TensorRef> {
                let hn = self.run_layers(Seqs::One(st), &vec![0u32; q], false, Some(&pos_at(POS_VERIFY)), false, None)?;
                if self.vocab_par {
                    self.verify_head(&hn, q, false)?;
                }
                Ok(hn)
            })();
            self.exact_rows.store(false, Ordering::Relaxed);
            let hn = run?;
            if let Some((_, m)) = head {
                self.backend.copy(&m.pending, 0, &hn.buf, hn.offset, q * m.row_bytes, self.stream)?;
            }
            Ok(())
        });
        let ops = self.capture_ops.lock().unwrap().take().unwrap_or_default();
        let comms = self.capture_comms.swap(0, Ordering::Relaxed) as usize;
        Ok(Captured { state_id: st.id, graph: graph?, ops, comms })
    }

    /// A peer's part of [`Model::spec_step`] on its own thread: the per-row recurrent copies and the position table, the
    /// recording for this step's shape (captured now if missing; `ok` falls to false on a failure), then, once every rank has
    /// passed `gate` and all recordings exist, the replay. `Ok(false)`: nothing ran (some rank has no recording).
    fn spec_step_peer(
        &self,
        st: &mut SequenceState,
        key: (u64, usize, usize, bool),
        tab: &[u8],
        gate: &std::sync::Barrier,
        ok: &std::sync::atomic::AtomicBool,
    ) -> Result<bool> {
        let (_, n, q, _) = key;
        let prep = (|| -> Result<()> {
            Self::prepare_spec_rows(&*self.backend, st, q)?;
            self.backend.upload(&self.ws.pos, 0, tab, self.stream)?;
            let mut cache = self.graphs.spec.lock().unwrap();
            if !cache.contains_key(&key) {
                if cache.len() >= 96 {
                    cache.clear();
                }
                let c = self.capture_spec_peer(st, n, q - 1)?;
                cache.insert(key, c);
                self.graphs.spec_captures.fetch_add(1, Ordering::Relaxed);
                self.telemetry.graph_event(false);
            }
            Ok(())
        })();
        if let Err(e) = &prep {
            eprintln!("tensor-parallel rank {}: speculative step graph capture failed ({e})", self.tp.rank);
            ok.store(false, Ordering::Relaxed);
        }
        gate.wait();
        if !ok.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let cache = self.graphs.spec.lock().unwrap();
        let c = cache.get(&key).expect("captured before the gate");
        self.backend.graph_replay(&c.graph, self.stream)?;
        self.graphs.spec_replays.fetch_add(1, Ordering::Relaxed);
        self.telemetry.graph_event(true);
        self.telemetry.record_ops(&c.ops);
        for _ in 0..c.comms {
            self.telemetry.record_comm(None);
        }
        Ok(true)
    }

    /// One speculative step: draft `k` tokens after `next` (the token the main model emits at position `st.len`) with the
    /// MTP head and run the verify pass over `[next, drafts]`, as one replay of a captured graph. On return the verify
    /// rows are committed to `st` (`st.len` advanced by `k + 1`), the head's cache holds the committed positions before
    /// them and its queue the `k + 1` new hidden rows, exactly as after [`Model::mtp_draft`] and
    /// [`Model::forward_verify`]; the caller accepts and rolls back (`spec_verify`). `Ok(None)`: capture failed, graph
    /// steps are off from here on and nothing ran.
    pub fn spec_step(&self, st: &mut SequenceState, next: u32, k: usize) -> Result<Option<SpecPass>> {
        if !self.spec_step_ok(st, next, k) {
            return Err(Error::invalid("spec_step: not applicable (check Model::spec_step_ok)"));
        }
        if next as usize >= self.dims.vocab {
            return Err(Error::BadRequest(format!("token id {next} >= vocab {}", self.dims.vocab)));
        }
        let q = k + 1;
        let p = st.len;
        let (n, p0) = {
            let m = st.mtp.as_ref().expect("checked by spec_step_ok");
            (m.pending_rows, m.kv.len)
        };
        // the catch-up tokens: the head row of position p0 + i pairs with the token of position p0 + i + 1; the last one is `next`
        let mut catch: Vec<u32> = st.tokens.get(p0 + 1..(p0 + n + 1).min(st.len)).unwrap_or(&[]).to_vec();
        if catch.len() < n {
            catch.push(next);
        }
        if catch.len() != n {
            return Err(Error::invalid(format!("spec_step: catch-up of {n} rows has {} tokens", catch.len())));
        }
        if let Some(&bad) = catch.iter().find(|&&i| i as usize >= self.dims.vocab) {
            return Err(Error::BadRequest(format!("token id {bad} >= vocab {}", self.dims.vocab)));
        }
        // the per-row recurrent copies (before the capture: a recording keeps their addresses)
        let recurrent = st.has_recurrent();
        if recurrent {
            let need = st.recurrent_bytes();
            st.spec_rows.saved = 0;
            st.spec_rows.base = st.len;
            st.spec_rows.bufs.retain(|b| b.bytes() >= need);
            while st.spec_rows.bufs.len() < q - 1 {
                let b = self.backend.alloc(need.max(16), MemKind::Device)?;
                st.spec_rows.bufs.push(b);
            }
        }
        // tests: scripted drafts replace the head's for the verify pass (the head still runs, so its cache and queue behave
        // as in real use), as `Model::mtp_draft` does
        let scripted = self.scripted_drafts(st, next, k);
        // the step's device inputs: the verify input row 0 (`ws.ids`), then the position table and the catch-up / scripted
        // ids (`ws.pos`)
        self.backend.upload(&self.ws.ids, 0, &(next as i32).to_le_bytes(), self.stream)?;
        let mut tab = vec![0i32; POS_TAB_ENTRIES];
        for r in 0..q {
            tab[POS_VERIFY + r] = (p + r) as i32;
        }
        for i in 0..n {
            tab[POS_CATCH_UP + i] = (p - n + i) as i32;
            tab[CATCH_IDS + i] = catch[i] as i32;
        }
        for i in 0..k.saturating_sub(1) {
            tab[POS_CHAIN + i] = (p + i) as i32;
        }
        for (i, &t) in scripted.iter().flatten().enumerate() {
            tab[SCRIPT_IDS + i] = t as i32;
        }
        let tab_bytes: Vec<u8> = tab.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.backend.upload(&self.ws.pos, 0, &tab_bytes, self.stream)?;

        let key = (st.id, n, q, scripted.is_some());
        let tp = !self.peers.is_empty();
        let run = (|| -> Result<Option<Vec<u8>>> {
            let mut cache = self.graphs.spec.lock().unwrap();
            if !cache.contains_key(&key) {
                if cache.len() >= 96 {
                    cache.clear(); // recordings of states that are gone
                }
                match self.capture_spec_step(st, n, k, scripted.is_some()) {
                    Ok(c) => {
                        cache.insert(key, c);
                        self.graphs.spec_captures.fetch_add(1, Ordering::Relaxed);
                    self.graphs.spec_fail.store(0, Ordering::Relaxed); // consecutive failures only
                        self.telemetry.graph_event(false);
                    }
                    Err(e) => {
                        let n = self.graphs.spec_fail.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "speculative step graph capture failed ({e}); {}",
                        if n < SPEC_CAPTURE_FAILURES_MAX { "this step runs eagerly, the next one retries" } else { "speculative steps run eagerly from here on" }
                    );
                    return Ok(None);
                    }
                }
            }
            let c = cache.get(&key).expect("captured above");
            if tp {
                // the peers replay their share in lockstep; a peer without a recording (its capture failed) means nobody replays
                if !self.spec_replay_tp(st, c, key, &tab_bytes)? {
                    let n = self.graphs.spec_fail.fetch_add(1, Ordering::Relaxed) + 1;
                    if n >= SPEC_CAPTURE_FAILURES_MAX {
                        eprintln!("speculative steps run eagerly from here on (a tensor-parallel peer cannot record them)");
                    }
                    return Ok(None);
                }
            } else {
                self.backend.graph_replay(&c.graph, self.stream)?;
            }
            self.graphs.spec_replays.fetch_add(1, Ordering::Relaxed);
            self.telemetry.graph_event(true);
            for &op in &c.ops {
                self.telemetry.record_op(op, None);
            }
            for _ in 0..c.comms {
                self.telemetry.record_comm(None);
            }
            drop(cache);
            // outputs [argmax of each row | the pass's input ids]
            let tok = self.verify_tok.as_ref().expect("checked by spec_step_ok");
            let mut b = vec![0u8; (Self::VERIFY_MAX_ROWS + q) * 4];
            self.backend.download(tok, 0, &mut b, self.stream)?;
            Ok(Some(b))
        })();
        let b = match run {
            Ok(Some(b)) => b,
            Ok(None) => return Ok(None),
            Err(e) => {
                // the recurrent slots, the caches and the head's queue are part-way through the step
                let _ = st.reset(&*self.backend);
                if let Some(m) = st.mtp.as_mut() {
                    m.synced = false;
                }
                return Err(e);
            }
        };
        let word = |i: usize| i32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]) as u32;
        let out: Vec<u32> = (0..q).map(word).collect();
        let vids: Vec<u32> = (0..q).map(|i| word(Self::VERIFY_MAX_ROWS + i)).collect();
        debug_assert_eq!(vids[0], next);
        let drafts = vids[1..].to_vec();
        // the host mirrors of what the pass did: the head cache took the catch-up rows (its chained rows are speculative and
        // not committed: the next pass writes those positions again), the verify rows are committed, the head's queue holds
        // their hidden rows
        {
            let m = st.mtp.as_mut().expect("checked by spec_step_ok");
            m.kv.commit(&catch);
            m.pending_rows = q;
        }
        st.commit_verify(&vids);
        if recurrent {
            st.spec_rows.saved = q - 1;
        }
        // the peers' host mirrors (lane gt2): what their eager draft and verify passes leave (`mtp_draft_tp`, `verify_peer`)
        for (pm, p) in self.peers.iter().zip(st.peers.iter_mut()) {
            if let Some(m) = p.st.mtp.as_mut() {
                m.kv.commit(&catch);
                m.pending_rows = q;
            }
            p.st.commit_verify(&vids);
            if p.st.has_recurrent() {
                p.st.spec_rows.saved = q - 1;
            }
            pm.dflash_after_forward(&mut p.st, q, false)?;
        }
        self.telemetry.step();
        // `--spec auto` (lane ag1): the replayed verify pass left the DFlash taps' features in the staging buffers (the
        // recording holds the tap launches), so the draft's context takes the rows as after an eager verify pass. Nothing
        // without the draft.
        self.dflash_after_forward(st, q, false)?;
        Ok(Some(SpecPass { drafts, out }))
    }

    /// Record the step for (`st`, `n` catch-up rows, `k` drafts). Executes nothing.
    fn capture_spec_step(&self, st: &SequenceState, n: usize, k: usize, scripted: bool) -> Result<Captured> {
        let head = self.mtp.as_ref().ok_or_else(|| Error::invalid("no MTP head"))?;
        let m = st.mtp.as_ref().ok_or_else(|| Error::invalid("sequence without MTP state"))?;
        let h = self.dims.hidden;
        let q = k + 1;
        let ws = &self.ws;
        // tensor parallelism (lane gt2): the ids go to the peers through the in-graph gather; with a sharded head after every
        // draft a chained pass embeds, else once (the verify ids)
        let tp = !self.peers.is_empty();
        let split_tp = tp && !head.whole();
        let g = q.next_multiple_of(4);
        let ids_at = |i: usize, len: usize| {
            let mut v = self.view(&ws.ids, DType::I32, &[len]);
            v.offset = 4 * i;
            v
        };
        let tab_ids_at = |i: usize, len: usize| {
            let mut v = self.view(&ws.pos, DType::I32, &[len]);
            v.offset = 4 * i;
            v
        };
        let pos_at = |i: usize| {
            let mut v = self.view(&ws.pos, DType::I32, &[1]);
            v.offset = 4 * i;
            v
        };
        *self.capture_ops.lock().unwrap() = Some(Vec::new());
        self.capture_comms.store(0, Ordering::Relaxed);
        let graph = self.backend.graph_capture(self.stream, &mut || {
            // catch-up: the head over the n queued rows -> draft 1
            let pending = TensorRef::new(m.pending.clone(), self.act, &[m.cap_rows, h]);
            let out = self.mtp_layer_at(
                head,
                &m.kv,
                n,
                None,
                &pending.rows(0, n),
                Some(&tab_ids_at(CATCH_IDS, n)),
                Some(&pos_at(POS_CATCH_UP)),
                false,
            )?;
            self.draft_head_to_device_into(&out, n - 1, false, ids_at(1, 1))?;
            self.backend.copy(&head.chain, 0, &out.buf, out.offset + (n - 1) * m.row_bytes, m.row_bytes, self.stream)?;
            // chain: drafts 2..k, each head row reading the previous draft (ws.ids[i]) and the previous head output
            for i in 1..k {
                if split_tp {
                    // the peers' shares of the chained pass embed this draft too (lane gt2)
                    self.spec_gather_ids(g)?;
                }
                let hid = TensorRef::new(head.chain.clone(), self.act, &[1, h]);
                let out = self.mtp_layer_at(
                    head,
                    &m.kv,
                    1,
                    None,
                    &hid,
                    Some(&ids_at(i, 1)),
                    Some(&pos_at(POS_CHAIN + i - 1)),
                    false,
                )?;
                self.draft_head_to_device_into(&out, 0, false, ids_at(1 + i, 1))?;
                self.backend.copy(&head.chain, 0, &out.buf, out.offset, m.row_bytes, self.stream)?;
            }
            if scripted {
                self.backend.copy(&ws.ids, 4, &ws.pos, 4 * SCRIPT_IDS, 4 * k, self.stream)?;
            }
            if tp {
                // the verify ids to the peers (lane gt2)
                self.spec_gather_ids(g)?;
            }
            // verify: [next, d_1 .. d_k] from ws.ids, every op on its row-exact kernel
            self.exact_rows.store(true, Ordering::Relaxed);
            let run = (|| -> Result<TensorRef> {
                let hn = self.run_layers(Seqs::One(st), &vec![0u32; q], false, Some(&pos_at(POS_VERIFY)), false, None)?;
                self.verify_head(&hn, q, false)?;
                Ok(hn)
            })();
            self.exact_rows.store(false, Ordering::Relaxed);
            let hn = run?;
            // the verify rows' hidden states into the head's queue (the catch-up rows were consumed above), and the pass's
            // input ids next to its outputs
            self.backend.copy(&m.pending, 0, &hn.buf, hn.offset, q * m.row_bytes, self.stream)?;
            let tok = self.verify_tok.as_ref().ok_or_else(|| Error::invalid("spec_step before enable_verify"))?;
            self.backend.copy(tok, Self::VERIFY_MAX_ROWS * 4, &ws.ids, 0, q * 4, self.stream)?;
            Ok(())
        });
        let ops = self.capture_ops.lock().unwrap().take().unwrap_or_default();
        let comms = self.capture_comms.swap(0, Ordering::Relaxed) as usize;
        Ok(Captured { state_id: st.id, graph: graph?, ops, comms })
    }

    /// Rank 0's replay of the captured step with in-process peers (lane gt2): each peer, on its own thread, prepares and
    /// records its share if missing ([`Model::spec_step_peer`]); once every rank has passed the gate, all replay (or, when a
    /// peer could not record, none: nothing ran anywhere and the step runs eagerly, `Ok(false)`). A failure after the gate
    /// aborts the group, as an eager pass's does.
    fn spec_replay_tp(&self, st: &mut SequenceState, c: &Captured, key: (u64, usize, usize, bool), tab: &[u8]) -> Result<bool> {
        if st.peers.len() != self.peers.len() {
            return Err(Error::invalid("this sequence state does not belong to a tensor-parallel model (use Model::new_state)"));
        }
        let comm0 = self.tp.comm.clone().ok_or_else(|| Error::invalid("tensor-parallel model without a communicator"))?;
        let gate = std::sync::Barrier::new(self.peers.len() + 1);
        let ok = std::sync::atomic::AtomicBool::new(true);
        let mut peers = std::mem::take(&mut st.peers);
        let out = std::thread::scope(|sc| {
            let handles: Vec<_> = self
                .peers
                .iter()
                .zip(peers.iter_mut())
                .map(|(m, ps)| {
                    let (gate, ok) = (&gate, &ok);
                    sc.spawn(move || {
                        let r = m.spec_step_peer(&mut ps.st, key, tab, gate, ok);
                        if r.is_err() {
                            if let Some(c) = &m.tp.comm {
                                c.abort();
                            }
                        }
                        r
                    })
                })
                .collect();
            gate.wait();
            let mut res = if ok.load(Ordering::Relaxed) { self.backend.graph_replay(&c.graph, self.stream).map(|_| true) } else { Ok(false) };
            if res.is_err() {
                comm0.abort();
            }
            for h in handles {
                let e = match h.join() {
                    Ok(Ok(_)) => continue,
                    Ok(Err(e)) => e,
                    Err(_) => Error::backend("a tensor-parallel rank panicked"),
                };
                if res.is_ok() {
                    res = Err(e);
                }
            }
            res
        });
        st.peers = peers;
        self.note_tp_pass(out)
    }

    /// Whether [`Model::forward_verify`] can run its `q` rows on `st` as the replay of a captured verify-only pass: graphs
    /// usable, a backend whose verify attention is one decode call per row at a device position (so the recording names no
    /// position; HIP), one rank, and `INTARSIA_VERIFY_GRAPH=1` (default off). The eager pass and the replay launch the same ops on
    /// the same kernels in the same order, so their outputs are bitwise equal (`tests/verify_rows_real.rs` checks it on the
    /// real model).
    pub(super) fn verify_graph_ok(&self, st: &SequenceState, q: usize, profile: bool) -> bool {
        !profile
            && verify_graph_env_on()
            && self.graphs.usable(self)
            && self.graphs.spec_fail.load(Ordering::Relaxed) < SPEC_CAPTURE_FAILURES_MAX
            && self.backend.verify_attn_per_row()
            // HIP only so far: there it is measured (lane dfh window 2); CUDA and Metal keep the eager pass until someone measures them
            && self.backend.name() == "hip"
            && self.recorder.lock().unwrap().is_none()
            && self.tp.world == 1
            && self.peers.is_empty()
            && st.id != 0
            && (2..=self.verify_rows).contains(&q)
            && q <= POS_TAB_ENTRIES
            && self.verify_tok.is_some()
            && self.ws.max_tokens >= q
            && st.len + q <= st.cap
    }

    /// The layers and the head of a verify pass over `ids` (`[last token, drafts...]`) as one graph replay, recorded on first
    /// use per (sequence state, row count). Inputs: the ids in `ws.ids`, row r's position `st.len + r` in `ws.pos`
    /// ([`POS_VERIFY`]). Leaves the greedy token of every row in `verify_tok` and returns the pass's hidden rows (the
    /// output [`Model::run_chunk`] would have returned); commits nothing. `Ok(None)`: the capture failed and nothing ran
    /// (the caller runs the pass eagerly).
    pub(super) fn verify_replay(&self, st: &SequenceState, ids: &[u32]) -> Result<Option<TensorRef>> {
        let q = ids.len();
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
        self.backend.upload(&self.ws.ids, 0, &bytes, self.stream)?;
        let mut tab = vec![0i32; POS_TAB_ENTRIES];
        for (r, v) in tab.iter_mut().enumerate().skip(POS_VERIFY).take(q) {
            *v = (st.len + r - POS_VERIFY) as i32;
        }
        let tab_bytes: Vec<u8> = tab.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.backend.upload(&self.ws.pos, 0, &tab_bytes, self.stream)?;
        let key = (st.id, q);
        let mut cache = self.graphs.verify.lock().unwrap();
        if !cache.contains_key(&key) {
            if cache.len() >= 96 {
                cache.clear(); // recordings of states that are gone
            }
            match self.capture_verify(st, q) {
                Ok(c) => {
                    cache.insert(key, c);
                    self.graphs.spec_captures.fetch_add(1, Ordering::Relaxed);
                    self.graphs.spec_fail.store(0, Ordering::Relaxed);
                    self.telemetry.graph_event(false);
                }
                Err(e) => {
                    let n = self.graphs.spec_fail.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "verify pass graph capture failed ({e}); {}",
                        if n < SPEC_CAPTURE_FAILURES_MAX { "this pass runs eagerly, the next one retries" } else { "speculative passes run eagerly from here on" }
                    );
                    return Ok(None);
                }
            }
        }
        let (c, hn) = cache.get(&key).expect("captured above");
        self.backend.graph_replay(&c.graph, self.stream)?;
        self.graphs.spec_replays.fetch_add(1, Ordering::Relaxed);
        self.telemetry.graph_event(true);
        for &op in &c.ops {
            self.telemetry.record_op(op, None);
        }
        Ok(Some(hn.clone()))
    }

    /// Record the verify-only pass of `q` rows for `st` (the eager [`Model::forward_verify`]'s launches, positions from the
    /// table). Executes nothing.
    fn capture_verify(&self, st: &SequenceState, q: usize) -> Result<(Captured, TensorRef)> {
        let pos = {
            let mut v = self.view(&self.ws.pos, DType::I32, &[1]);
            v.offset = 4 * POS_VERIFY;
            v
        };
        *self.capture_ops.lock().unwrap() = Some(Vec::new());
        self.capture_comms.store(0, Ordering::Relaxed);
        let mut out = None;
        let graph = self.backend.graph_capture(self.stream, &mut || {
            self.exact_rows.store(true, Ordering::Relaxed);
            let run = (|| -> Result<TensorRef> {
                let hn = self.run_layers(Seqs::One(st), &vec![0u32; q], false, Some(&pos), false, None)?;
                self.verify_head(&hn, q, false)?;
                Ok(hn)
            })();
            self.exact_rows.store(false, Ordering::Relaxed);
            out = Some(run?);
            Ok(())
        });
        let ops = self.capture_ops.lock().unwrap().take().unwrap_or_default();
        self.capture_comms.swap(0, Ordering::Relaxed);
        let graph = graph?;
        let hn = out.ok_or_else(|| Error::invalid("verify capture produced no output"))?;
        Ok((Captured { state_id: st.id, graph, ops, comms: 0 }, hn))
    }
}

/// `INTARSIA_SPEC_GRAPH_TP=1` (lane gt2) lets the captured step run with in-process tensor-parallel peers (default off: the
/// step drafts and verifies eagerly at tp > 1, as before), read once.
fn spec_graph_tp_env_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_SPEC_GRAPH_TP").is_ok_and(|v| v == "1"))
}

/// `INTARSIA_VERIFY_GRAPH=1` turns the captured verify-only pass on (default off: at tp=1 on HIP it gains 0.2-2% over the eager
/// pass, lane dfh window 3; default-on is decided at tp=2 from an interleaved measurement), read once.
fn verify_graph_env_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_VERIFY_GRAPH").is_ok_and(|v| v == "1"))
}
