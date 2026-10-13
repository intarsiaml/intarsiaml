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

use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Graph decode profiles every `profile_every * PROFILE_STRETCH`-th step instead of every `profile_every`-th: a
/// profiled step is an eager one (844 synchronised launches on the 35B model).
pub(super) const PROFILE_STRETCH: u64 = 8;

struct Captured {
    state_id: u64,
    graph: GraphHandle,
    /// Ops of one step, in order (counted into telemetry per replay).
    ops: Vec<Op>,
}

pub(super) struct GraphState {
    enabled: AtomicBool,
    captured: Mutex<Option<Captured>>,
    captures: AtomicU64,
    replays: AtomicU64,
}

/// How often the decode graph was captured and replayed (`Model::graph_stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    pub enabled: bool,
    pub captures: u64,
    pub replays: u64,
}

impl GraphState {
    pub(super) fn new(enabled: bool) -> GraphState {
        GraphState {
            enabled: AtomicBool::new(enabled),
            captured: Mutex::new(None),
            captures: AtomicU64::new(0),
            replays: AtomicU64::new(0),
        }
    }

    /// Drop the recording (scratch moved, kernels re-resolved, state gone).
    pub(super) fn invalidate(&self) {
        *self.captured.lock().unwrap() = None;
    }

    /// Graphs requested, supported by the backend, and valid at every position for this plan.
    pub(super) fn usable(&self, m: &Model) -> bool {
        self.enabled.load(Ordering::Relaxed)
            && m.backend.supports_graphs()
            && m.resolved.decode_position_stable
            && !m.registry.reference_only
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
        }
    }

    /// One decode step through the captured graph. `Ok(None)`: not applicable now (capture failed and graphs
    /// were switched off); the caller runs the step eagerly. `st.len + 1 <= st.cap` was checked by the caller.
    pub(super) fn graph_decode_step(&self, st: &SequenceState, id: u32, want_logits: bool) -> Result<Option<StepOut>> {
        if st.id == 0 || self.recorder.lock().unwrap().is_some() {
            return Ok(None);
        }
        if id as usize >= self.dims.vocab {
            return Err(Error::BadRequest(format!("token id {id} >= vocab {}", self.dims.vocab)));
        }
        // the step's device inputs
        self.backend.upload(&self.ws.ids, 0, &(id as i32).to_le_bytes(), 0)?;
        self.backend.upload(&self.ws.pos, 0, &(st.len as i32).to_le_bytes(), 0)?;
        let mut cache = self.graphs.captured.lock().unwrap();
        if cache.as_ref().map(|c| c.state_id) != Some(st.id) {
            *cache = None; // a recording of another state's buffers is of no use
            match self.capture_decode(st) {
                Ok(c) => {
                    *cache = Some(c);
                    self.graphs.captures.fetch_add(1, Ordering::Relaxed);
                    self.telemetry.graph_event(false);
                }
                Err(e) => {
                    eprintln!("decode graph capture failed ({e}); decoding eagerly from here on");
                    self.graphs.enabled.store(false, Ordering::Relaxed);
                    return Ok(None);
                }
            }
        }
        let c = cache.as_ref().expect("captured above");
        self.backend.graph_replay(&c.graph, 0)?;
        self.graphs.replays.fetch_add(1, Ordering::Relaxed);
        self.telemetry.graph_event(true);
        for &op in &c.ops {
            self.telemetry.record_op(op, None);
        }
        drop(cache);
        let v = self.dims.vocab;
        if want_logits {
            let mut bytes = vec![0u8; v * 4];
            self.backend.download(&self.ws.logits, 0, &mut bytes, 0)?;
            Ok(Some(StepOut::Logits(
                bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            )))
        } else {
            let mut b = [0u8; 4];
            self.backend.download(&self.ws.tok, 0, &mut b, 0)?;
            Ok(Some(StepOut::Token(i32::from_le_bytes(b) as u32)))
        }
    }

    fn capture_decode(&self, st: &SequenceState) -> Result<Captured> {
        let pos = self.view(&self.ws.pos, DType::I32, &[1]);
        *self.capture_ops.lock().unwrap() = Some(Vec::new());
        let graph = self.backend.graph_capture(0, &mut || {
            // only the length of `ids` matters: the token is read from ws.ids, the position from ws.pos
            let hn = self.run_chunk_at(st, &[0], false, Some(&pos), false)?;
            self.head_to_device(&hn, 0, false)
        });
        let ops = self.capture_ops.lock().unwrap().take().unwrap_or_default();
        Ok(Captured { state_id: st.id, graph: graph?, ops })
    }
}
