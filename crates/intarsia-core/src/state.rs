//! Sequence state (ARCHITECTURE section 4): per-layer state slots.
//!
//! `AttnKv` is contiguous, single sequence. `Recurrent` (GatedDeltaNet / Mamba state: conv state
//! and the delta-rule matrix, f32) is a fixed-size slot that cannot be rolled back by truncation,
//! so prefix reuse for hybrid models goes through snapshots: [`SequenceState::snapshot`] copies
//! every recurrent slot at the current token count, [`SequenceState::restorable_len`] says which
//! prefix lengths [`SequenceState::truncate`] can return to.

use crate::backend::{Backend, Buffer, MemKind};
use crate::dtype::DType;
use crate::error::{Error, Result};
use std::sync::Arc;

/// Recurrent-state snapshots a sequence keeps for prefix reuse (see [`SequenceState::snapshot`]).
pub const RECURRENT_SNAPSHOTS: usize = 4;

#[derive(Clone, Debug)]
pub enum SlotSpec {
    AttnKv { n_kv_heads: usize, head_dim: usize, dtype: DType },
    Recurrent { bytes: usize },
}

pub struct AttnKvState {
    /// `[n_kv_heads, cap, head_dim]`
    pub k: Buffer,
    pub v: Buffer,
    /// Rows filled so far.
    pub len: usize,
    pub cap: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
}

pub struct RecurrentState {
    pub buf: Buffer,
    pub bytes: usize,
}

pub enum StateSlot {
    AttnKv(AttnKvState),
    Recurrent(RecurrentState),
}

impl StateSlot {
    pub fn bytes(&self) -> usize {
        match self {
            StateSlot::AttnKv(s) => s.k.bytes() + s.v.bytes(),
            StateSlot::Recurrent(r) => r.bytes,
        }
    }
}

/// Copy of every recurrent slot (concatenated, slot order) at token count `len`.
pub struct RecurrentSnapshot {
    pub len: usize,
    buf: Buffer,
}

static NEXT_STATE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A tensor-parallel peer rank's share of a sequence: its own slots on its own device.
pub struct PeerState {
    pub backend: Arc<dyn Backend>,
    pub st: SequenceState,
}

/// All state of one sequence.
pub struct SequenceState {
    /// Unique per allocated state (never reused): a captured CUDA graph is bound to one state's buffers
    /// and identifies it by this id. 0 for recording-only phantom states.
    pub id: u64,
    pub slots: Vec<StateSlot>,
    /// Tokens processed so far (KV rows filled in every attention slot).
    pub len: usize,
    /// Capacity in tokens of the attention slots.
    pub cap: usize,
    /// Token ids whose state is held (prefix-reuse key); `tokens.len() == len`.
    pub tokens: Vec<u32>,
    /// Recurrent-state snapshots, oldest first; each is valid for `tokens[..len]` as long as that
    /// prefix is unchanged (dropped when the state is truncated below `len`).
    pub snapshots: Vec<RecurrentSnapshot>,
    /// Tensor parallelism: the state of ranks 1.. (this struct is rank 0's). Created, truncated and snapshotted
    /// together with the primary; advanced by the rank's own forward pass.
    pub peers: Vec<PeerState>,
    /// Tensor parallelism across processes (`crate::tp_remote`): the remote rank's copy of this sequence. Truncated and
    /// snapshotted after the primary, advanced by the remote rank's forward pass.
    pub remote: Option<crate::tp_remote::RemoteSeq>,
}

impl SequenceState {
    pub fn new(backend: &dyn Backend, specs: &[SlotSpec], cap: usize) -> Result<SequenceState> {
        let mut slots = Vec::new();
        for s in specs {
            slots.push(match s {
                SlotSpec::AttnKv { n_kv_heads, head_dim, dtype } => {
                    let bytes = n_kv_heads * cap * head_dim * dtype.size_bytes();
                    let k = backend.alloc(bytes, MemKind::Device)?;
                    let v = backend.alloc(bytes, MemKind::Device)?;
                    StateSlot::AttnKv(AttnKvState {
                        k,
                        v,
                        len: 0,
                        cap,
                        n_kv_heads: *n_kv_heads,
                        head_dim: *head_dim,
                        dtype: *dtype,
                    })
                }
                SlotSpec::Recurrent { bytes } => {
                    let buf = backend.alloc((*bytes).max(16), MemKind::Device)?;
                    backend.zero(&buf, 0)?;
                    StateSlot::Recurrent(RecurrentState { buf, bytes: *bytes })
                }
            });
        }
        let id = NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(SequenceState { id, slots, len: 0, cap, tokens: Vec::new(), snapshots: Vec::new(), peers: Vec::new(), remote: None })
    }

    pub fn total_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.bytes()).sum::<usize>() + self.peers.iter().map(|p| p.st.total_bytes()).sum::<usize>()
    }

    /// Bytes of the filled part of the attention caches.
    pub fn used_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| match s {
                StateSlot::AttnKv(a) => 2 * a.n_kv_heads * a.len * a.head_dim * a.dtype.size_bytes(),
                StateSlot::Recurrent(r) => r.bytes,
            })
            .sum::<usize>()
            + self.peers.iter().map(|p| p.st.used_bytes()).sum::<usize>()
    }

    pub fn has_recurrent(&self) -> bool {
        self.slots.iter().any(|s| matches!(s, StateSlot::Recurrent(_)))
    }

    /// Bytes of all recurrent slots (what one snapshot holds).
    pub fn recurrent_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| match s {
                StateSlot::Recurrent(r) => r.bytes,
                _ => 0,
            })
            .sum()
    }

    /// Copy the recurrent state as of the current token count. At most `keep` snapshots are held
    /// (the oldest is dropped, its buffer reused); a snapshot at the same length is replaced.
    /// No-op without recurrent slots or at length 0 (the zero state needs no copy).
    pub fn snapshot(&mut self, backend: &dyn Backend, keep: usize) -> Result<()> {
        for p in &mut self.peers {
            p.st.snapshot(&*p.backend, keep)?;
        }
        if let Some(r) = &self.remote {
            r.snapshot(keep)?;
        }
        let total = self.recurrent_bytes();
        if total == 0 || self.len == 0 || keep == 0 {
            return Ok(());
        }
        if let Some(i) = self.snapshots.iter().position(|s| s.len == self.len) {
            self.snapshots.remove(i);
        }
        let reuse = if self.snapshots.len() >= keep { Some(self.snapshots.remove(0).buf) } else { None };
        let buf = match reuse {
            Some(b) if b.bytes() >= total => b,
            _ => backend.alloc(total, MemKind::Device)?,
        };
        let mut off = 0;
        for s in &self.slots {
            if let StateSlot::Recurrent(r) = s {
                backend.copy(&buf, off, &r.buf, 0, r.bytes, 0)?;
                off += r.bytes;
            }
        }
        self.snapshots.push(RecurrentSnapshot { len: self.len, buf });
        Ok(())
    }

    /// Largest prefix length `<= want` that [`SequenceState::truncate`] can return to: the current
    /// length, a snapshot, or 0 (the zero state). Without recurrent slots every length qualifies.
    pub fn restorable_len(&self, want: usize) -> usize {
        let want = want.min(self.len);
        if !self.has_recurrent() || want == self.len {
            return want;
        }
        self.snapshots.iter().map(|s| s.len).filter(|&l| l <= want).max().unwrap_or(0)
    }

    /// Roll back to the first `len` tokens (prefix reuse). Attention caches just shorten;
    /// recurrent state is restored from the snapshot taken at exactly `len`, zeroed for 0, and
    /// otherwise cannot be rolled back (`Unsupported`). Truncating to 0 always zeroes recurrent
    /// state, even when the state is already empty: a forward that failed halfway through the first
    /// chunk leaves written-but-uncommitted state behind.
    pub fn truncate(&mut self, len: usize, backend: &dyn Backend) -> Result<()> {
        if len > self.len {
            return Err(Error::invalid("truncate beyond current length"));
        }
        for p in &mut self.peers {
            if len > p.st.len {
                return Err(Error::invalid("truncate beyond a peer rank's length"));
            }
            p.st.truncate(len, &*p.backend)?;
        }
        if self.has_recurrent() {
            if len == 0 {
                for s in &self.slots {
                    if let StateSlot::Recurrent(r) = s {
                        backend.zero(&r.buf, 0)?;
                    }
                }
            } else if len != self.len {
                let Some(snap) = self.snapshots.iter().find(|s| s.len == len) else {
                    return Err(Error::Unsupported(format!(
                        "cannot roll recurrent state back to {len} tokens (no snapshot there; length {})",
                        self.len
                    )));
                };
                let mut off = 0;
                for s in &self.slots {
                    if let StateSlot::Recurrent(r) = s {
                        backend.copy(&r.buf, 0, &snap.buf, off, r.bytes, 0)?;
                        off += r.bytes;
                    }
                }
            }
        }
        self.snapshots.retain(|s| s.len <= len);
        for s in self.slots.iter_mut() {
            if let StateSlot::AttnKv(a) = s {
                a.len = len;
            }
        }
        self.len = len;
        self.tokens.truncate(len);
        // the remote rank follows only a rollback the primary could make (same snapshots on both, so it can too)
        if let Some(r) = &mut self.remote {
            r.truncate(len)?;
        }
        Ok(())
    }

    pub fn reset(&mut self, backend: &dyn Backend) -> Result<()> {
        self.truncate(0, backend)
    }

    /// Advance after a forward pass appended `ids`.
    pub fn commit(&mut self, ids: &[u32]) {
        self.len += ids.len();
        self.tokens.extend_from_slice(ids);
        for s in self.slots.iter_mut() {
            if let StateSlot::AttnKv(a) = s {
                a.len = self.len;
            }
        }
    }
}
