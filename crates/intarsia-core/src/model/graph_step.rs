//! The single-token decode step as a captured graph (CUDA graph): record the whole step once per sequence
//! state (token embedding, every block, final norm, lm_head, argmax) and replay it for every later token.
//!
//! What makes one recording valid at every position:
//!  * token id and position are device scalars (`ws.ids`, `ws.pos`) the host rewrites before each replay; the
//!    position-dependent ops (rope, kv_append, attn_decode) read `OpCall::pos` when they run;
//!  * everything else a step touches has a fixed address: workspace buffers, kernel scratch, weights, and the
//!    sequence state's KV / recurrent buffers (which is why the recording is bound to one state, by `SequenceState::id`);
//!  * the decode kernels are the same at every position (`ResolvedPlan::decode_position_stable`).
//!
//! The recording ends in both heads (logits `[1, V]` and the argmax token), so one graph serves greedy decoding and
//! sampling. Capture executes nothing: the step that triggers it is replayed once, never run eagerly first (the
//! recurrent state and the KV append are in place and would apply twice).
//!
//! Profiled steps (per-op timing synchronises after every op) run on the eager path; so does anything the backend
//! cannot capture. A failed capture switches graphs off for the model with a note on stderr.
//!
//! Tensor parallelism (world > 1): only with a communicator whose collectives are capturable (`Comm::capture_safe`,
//! NCCL); the all-reduces are recorded into each rank's graph on the same stream as the kernels. Rank 0 decides per
//! step whether the step replays ([`Model::decode_graph_now`]) and the other ranks follow that decision (the remote
//! rank gets it with the `Forward` command), so every rank issues the same collectives in the same order. A rank other
//! than 0 records the layers only (rank 0 owns the head), or, with a vocab-parallel head (`LoadOptions::tp_vocab`), the
//! layers and its slice of the head with the logits all-gather. Under TP a failed capture fails the step, which takes the
//! group down (ADR-0006): falling back to eager on one rank only would leave the ranks deciding differently.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Graph decode profiles every `profile_every * PROFILE_STRETCH`-th step instead of every `profile_every`-th: a
/// profiled step is an eager one (844 synchronised launches on the 35B model).
pub(super) const PROFILE_STRETCH: u64 = 8;

/// Solo decode recordings kept at once (one per sequence slot of a batched server).
const SOLO_GRAPHS_MAX: usize = 32;

pub(super) struct Captured {
    pub(super) state_id: u64,
    pub(super) graph: GraphHandle,
    /// Ops of one step, in order (counted into telemetry per replay).
    pub(super) ops: Vec<Op>,
    /// Collectives recorded in the step (counted into telemetry per replay).
    pub(super) comms: usize,
}

pub(super) struct GraphState {
    enabled: AtomicBool,
    /// Solo decode steps: one recording per (sequence state, head) because a solo recording names that state's own buffers.
    /// Several states alternating (a batched server with one running row at a time) keep one recording each instead of
    /// re-recording at every switch (Fable review m3); the table is cleared at `SOLO_GRAPHS_MAX` entries.
    captured: Mutex<std::collections::HashMap<(u64, bool), Captured>>,
    /// Batched decode steps (R3c): one recording per row count, valid for any set of sequences because every state buffer a
    /// batched step touches is named by the row table the host rewrites before each replay (`Model::row_tab`).
    batched: Mutex<std::collections::HashMap<usize, Captured>>,
    /// A recording of a batched step found a launch bound to one sequence's buffers (a per-row fallback): batched graphs are
    /// off for this model.
    batched_off: AtomicBool,
    /// Speculative MTP steps (`model/spec_graph.rs`): one recording per (sequence state, catch-up rows, verify rows, scripted drafts).
    pub(super) spec: Mutex<std::collections::HashMap<(u64, usize, usize, bool), Captured>>,
    /// A speculative step failed to capture: they run eagerly from here on.
    pub(super) spec_off: AtomicBool,
    pub(super) spec_captures: AtomicU64,
    pub(super) spec_replays: AtomicU64,
    /// Set by the per-row fallbacks of a batched step while a recording is being made.
    pub(super) state_bound: AtomicBool,
    batch_captures: AtomicU64,
    batch_replays: AtomicU64,
    captures: AtomicU64,
    replays: AtomicU64,
}

/// How often the decode graph was captured and replayed (`Model::graph_stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    pub enabled: bool,
    pub captures: u64,
    pub replays: u64,
    /// Batched-step recordings (one per row count) and replays (`INTARSIA_BATCH_GRAPHS`, R3c).
    pub batch_captures: u64,
    pub batch_replays: u64,
    /// Speculative-step recordings and replays (`Model::spec_step`).
    pub spec_captures: u64,
    pub spec_replays: u64,
}

impl GraphState {
    pub(super) fn new(enabled: bool) -> GraphState {
        GraphState {
            enabled: AtomicBool::new(enabled),
            captured: Mutex::new(Default::default()),
            batched: Mutex::new(Default::default()),
            batched_off: AtomicBool::new(false),
            spec: Mutex::new(Default::default()),
            spec_off: AtomicBool::new(false),
            spec_captures: AtomicU64::new(0),
            spec_replays: AtomicU64::new(0),
            state_bound: AtomicBool::new(false),
            batch_captures: AtomicU64::new(0),
            batch_replays: AtomicU64::new(0),
            captures: AtomicU64::new(0),
            replays: AtomicU64::new(0),
        }
    }

    /// Whether graphs were requested for this model (before the backend / plan conditions of `usable`).
    pub(super) fn configured(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Drop the recording (scratch moved, kernels re-resolved, state gone).
    pub(super) fn invalidate(&self) {
        self.captured.lock().unwrap().clear();
        self.batched.lock().unwrap().clear();
        self.spec.lock().unwrap().clear();
    }

    /// Graphs requested, supported by the backend, and valid at every position for this plan. Under tensor
    /// parallelism also: the communicator is capturable and no host-side test hook (`INTARSIA_TP_TEST_BUG`) runs.
    pub(super) fn usable(&self, m: &Model) -> bool {
        self.enabled.load(Ordering::Relaxed)
            && m.backend.supports_graphs()
            && m.resolved.decode_position_stable
            && !m.registry.reference_only
            && (m.tp.world == 1 || (m.tp.comm.as_ref().is_some_and(|c| c.capture_safe()) && m.test_bug.is_none()))
    }
}

/// In-process tensor parallelism (one process, a thread per rank): the peers keep their own telemetry, so rank 0 prints
/// each peer's graph counters when the group goes away, as `tp-worker` does for a remote rank. That line is the evidence
/// that the other ranks replayed in lockstep (a rank that ran eagerly while rank 0 replayed gives the same bits).
impl Drop for Model {
    fn drop(&mut self) {
        if self.tp_arnorm {
            // evidence that the fused all-reduce + rmsnorm ran (eager calls, and calls recorded into a graph: each replay runs those)
            for m in std::iter::once(&*self).chain(self.peers.iter()) {
                eprintln!("tensor-parallel rank {}: fused all-reduce+rmsnorm {}", m.tp.rank, m.ar_fused_counts());
            }
        }
        for p in &self.peers {
            let g = p.graph_stats();
            eprintln!(
                "tensor-parallel rank {}: decode graph enabled={} captures={} replays={}",
                p.tp.rank, g.enabled, g.captures, g.replays
            );
        }
    }
}

impl Model {
    /// Turn graph replay of the decode step on or off (`--no-graphs`). Off drops the recording.
    pub fn set_graphs(&self, on: bool) {
        self.graphs.enabled.store(on, Ordering::Relaxed);
        if !on {
            self.graphs.invalidate();
        }
    }

    pub fn graph_stats(&self) -> GraphStats {
        GraphStats {
            enabled: self.graphs.usable(self),
            captures: self.graphs.captures.load(Ordering::Relaxed),
            replays: self.graphs.replays.load(Ordering::Relaxed),
            batch_captures: self.graphs.batch_captures.load(Ordering::Relaxed),
            batch_replays: self.graphs.batch_replays.load(Ordering::Relaxed),
            spec_captures: self.graphs.spec_captures.load(Ordering::Relaxed),
            spec_replays: self.graphs.spec_replays.load(Ordering::Relaxed),
        }
    }

    /// Will rank 0's next forward of `ids` in `mode` try the decode graph? The test `forward_inner` applies; under TP
    /// the other ranks are told the answer so they replay (or not) in lockstep.
    pub(super) fn decode_graph_now(&self, st: &SequenceState, ids: &[u32], mode: HeadMode) -> bool {
        ids.len() == 1
            && matches!(mode, HeadMode::LastLogits | HeadMode::LastArgmax)
            && st.id != 0
            && self.graphs.usable(self)
            && !(self.profile_every > 0
                && self.telemetry.steps_total() % (self.profile_every as u64 * PROFILE_STRETCH) == 0)
    }

    /// One decode step through the captured graph. `Ok(None)`: not applicable now (capture failed and graphs
    /// were switched off); the caller runs the step eagerly. `st.len + 1 <= st.cap` was checked by the caller.
    pub(super) fn graph_decode_step(&self, st: &SequenceState, id: u32, want_logits: bool) -> Result<Option<StepOut>> {
        if !self.graph_step(st, id, true)? {
            return Ok(None);
        }
        let v = self.dims.vocab;
        if want_logits {
            let mut bytes = vec![0u8; v * 4];
            self.backend.download(&self.ws.logits, 0, &mut bytes, self.stream)?;
            Ok(Some(StepOut::Logits(
                bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            )))
        } else {
            let mut b = [0u8; 4];
            self.backend.download(&self.ws.tok, 0, &mut b, self.stream)?;
            Ok(Some(StepOut::Token(i32::from_le_bytes(b) as u32)))
        }
    }

    /// A tensor-parallel rank other than 0: one decode step's layers through its captured graph, because rank 0
    /// replays its own in this step. Enqueued only (the next collective or download orders it). Fails when this rank
    /// cannot replay.
    pub(super) fn graph_peer_step(&self, st: &SequenceState, id: u32) -> Result<()> {
        if !self.graphs.usable(self) {
            return Err(Error::invalid(format!(
                "tensor-parallel rank {}: rank 0 replays the decode graph but this rank cannot (--no-graphs, INTARSIA_TP_TEST_BUG, or no capturable communicator)",
                self.tp.rank
            )));
        }
        if self.graph_step(st, id, false)? {
            Ok(())
        } else {
            Err(Error::backend(format!("tensor-parallel rank {}: decode graph unavailable", self.tp.rank)))
        }
    }

    /// Upload the step's inputs, capture on first use for this state, replay. `head`: record the head (rank 0) or the
    /// layers only. `Ok(false)`: not applicable (a state without an id, or a tp=1 capture failure: graphs are now off).
    fn graph_step(&self, st: &SequenceState, id: u32, head: bool) -> Result<bool> {
        if st.id == 0 || self.recorder.lock().unwrap().is_some() {
            if self.tp.world > 1 {
                return Err(Error::invalid("tensor-parallel decode graph: no replay possible for this state"));
            }
            return Ok(false);
        }
        if id as usize >= self.dims.vocab {
            return Err(Error::BadRequest(format!("token id {id} >= vocab {}", self.dims.vocab)));
        }
        // the step's device inputs
        self.backend.upload(&self.ws.ids, 0, &(id as i32).to_le_bytes(), self.stream)?;
        self.backend.upload(&self.ws.pos, 0, &(st.len as i32).to_le_bytes(), self.stream)?;
        let mut cache = self.graphs.captured.lock().unwrap();
        let key = (st.id, head);
        if !cache.contains_key(&key) {
            if cache.len() >= SOLO_GRAPHS_MAX {
                cache.clear(); // recordings of states that are gone, or too many alternating ones
            }
            match self.capture_decode(st, head) {
                Ok(c) => {
                    cache.insert(key, c);
                    self.graphs.captures.fetch_add(1, Ordering::Relaxed);
                    self.telemetry.graph_event(false);
                }
                Err(e) if self.tp.world > 1 => {
                    // the other ranks replay this step: fail it (the group goes down) rather than diverge
                    self.graphs.enabled.store(false, Ordering::Relaxed);
                    return Err(Error::backend(format!(
                        "tensor-parallel rank {}: decode graph capture failed: {e}",
                        self.tp.rank
                    )));
                }
                Err(e) => {
                    eprintln!("decode graph capture failed ({e}); decoding eagerly from here on");
                    self.graphs.enabled.store(false, Ordering::Relaxed);
                    return Ok(false);
                }
            }
        }
        let c = cache.get(&key).expect("captured above");
        self.backend.graph_replay(&c.graph, self.stream)?;
        self.graphs.replays.fetch_add(1, Ordering::Relaxed);
        self.telemetry.graph_event(true);
        self.telemetry.record_ops(&c.ops);
        for _ in 0..c.comms {
            self.telemetry.record_comm(None);
        }
        Ok(true)
    }

    /// A batched decode step (`states[r]` fed `ids[r]`, head included: logits `[b, V]` and the argmax tokens) through the graph
    /// recorded for this row count. `Ok(false)`: not applicable (graphs off, a per-row fallback in the step, capture failed);
    /// nothing ran and the caller runs the step eagerly. Like the solo step, the recording is made by the step that needs it
    /// and replayed once; capture executes nothing.
    pub(super) fn graph_batch_step(&self, states: &[&SequenceState], ids: &[u32]) -> Result<bool> {
        let b = ids.len();
        if !self.graphs.usable(self)
            || self.graphs.batched_off.load(Ordering::Relaxed)
            || self.row_tab.is_none()
            || self.recorder.lock().unwrap().is_some()
            || self.tp.world > 1
            || std::env::var("INTARSIA_BATCH_GRAPHS").is_ok_and(|v| v == "0")
        {
            return Ok(false);
        }
        for &i in ids {
            if i as usize >= self.dims.vocab {
                return Err(Error::BadRequest(format!("token id {i} >= vocab {}", self.dims.vocab)));
            }
        }
        // the step's device inputs: tokens, then the row table and positions (stream-ordered before the replay)
        let prof_upload = crate::hostprof::span(crate::hostprof::Cat::Upload);
        let ids_i32: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
        self.backend.upload(&self.ws.ids, 0, &ids_i32, self.stream)?;
        self.upload_row_table(states)?;
        drop(prof_upload);
        let mut cache = self.graphs.batched.lock().unwrap();
        if !cache.contains_key(&b) {
            *self.capture_ops.lock().unwrap() = Some(Vec::new());
            self.graphs.state_bound.store(false, Ordering::Relaxed);
            let zeros = vec![0u32; b];
            let graph = self.backend.graph_capture(self.stream, &mut || {
                // only the row count matters: tokens are read from ws.ids, state buffers and positions from the tables
                let refs: Vec<&SequenceState> = states.to_vec();
                let hn = self.run_layers(Seqs::Batch(&refs), &zeros, false, None, false, None)?;
                self.batch_head(&hn, b, false)
            });
            let ops = self.capture_ops.lock().unwrap().take().unwrap_or_default();
            if self.graphs.state_bound.swap(false, Ordering::Relaxed) {
                // a launch of this step names one sequence's buffers: the recording would replay on that sequence only
                eprintln!("batched decode graph: a per-row launch is bound to one sequence's state; batched steps run eagerly");
                self.graphs.batched_off.store(true, Ordering::Relaxed);
                return Ok(false);
            }
            match graph {
                Ok(graph) => {
                    cache.insert(b, Captured { state_id: 0, graph, ops, comms: 0 });
                    self.graphs.batch_captures.fetch_add(1, Ordering::Relaxed);
                    self.telemetry.graph_event(false);
                }
                Err(e) => {
                    eprintln!("batched decode graph capture failed ({e}); batched steps run eagerly from here on");
                    self.graphs.batched_off.store(true, Ordering::Relaxed);
                    return Ok(false);
                }
            }
        }
        let c = cache.get(&b).expect("captured above");
        self.backend.graph_replay(&c.graph, self.stream)?;
        self.graphs.batch_replays.fetch_add(1, Ordering::Relaxed);
        self.telemetry.graph_event(true);
        self.telemetry.record_ops(&c.ops);
        Ok(true)
    }

    fn capture_decode(&self, st: &SequenceState, head: bool) -> Result<Captured> {
        let pos = self.view(&self.ws.pos, DType::I32, &[1]);
        *self.capture_ops.lock().unwrap() = Some(Vec::new());
        self.capture_comms.store(0, Ordering::Relaxed);
        let graph = self.backend.graph_capture(self.stream, &mut || {
            // only the length of `ids` matters: the token is read from ws.ids, the position from ws.pos
            let hn = self.run_chunk_at(st, &[0], false, Some(&pos), false)?;
            // a vocab-parallel peer records its slice of the head and the gather (no argmax): rank 0 replays the same collective
            if head || self.vocab_par {
                self.head_to_device(&hn, 0, false)
            } else {
                Ok(())
            }
        });
        let ops = self.capture_ops.lock().unwrap().take().unwrap_or_default();
        let comms = self.capture_comms.swap(0, Ordering::Relaxed) as usize;
        Ok(Captured { state_id: st.id, graph: graph?, ops, comms })
    }
}
