//! Sequence state (ARCHITECTURE section 4): per-layer state slots.
//!
//! Phase 0 implements `AttnKv` (contiguous, single sequence). `Recurrent`
//! (GDN / Mamba state) is defined and allocated but unused by Qwen3.

use crate::backend::{Backend, Buffer, MemKind};
use crate::dtype::DType;
use crate::error::{Error, Result};

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

/// All state of one sequence.
pub struct SequenceState {
    pub slots: Vec<StateSlot>,
    /// Tokens processed so far (KV rows filled in every attention slot).
    pub len: usize,
    /// Capacity in tokens of the attention slots.
    pub cap: usize,
    /// Token ids whose state is held (prefix-reuse key); `tokens.len() == len`.
    pub tokens: Vec<u32>,
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
        Ok(SequenceState { slots, len: 0, cap, tokens: Vec::new() })
    }

    pub fn total_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.bytes()).sum()
    }

    /// Bytes of the filled part of the attention caches.
    pub fn used_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| match s {
                StateSlot::AttnKv(a) => 2 * a.n_kv_heads * a.len * a.head_dim * a.dtype.size_bytes(),
                StateSlot::Recurrent(r) => r.bytes,
            })
            .sum()
    }

    pub fn has_recurrent(&self) -> bool {
        self.slots.iter().any(|s| matches!(s, StateSlot::Recurrent(_)))
    }

    /// Roll back to the first `len` tokens (prefix reuse). Recurrent state cannot
    /// be rolled back; truncating to anything but the current length or 0 fails.
    pub fn truncate(&mut self, len: usize, backend: &dyn Backend) -> Result<()> {
        if len > self.len {
            return Err(Error::invalid("truncate beyond current length"));
        }
        if self.has_recurrent() && len != self.len {
            if len != 0 {
                return Err(Error::Unsupported("cannot roll back recurrent state to a non-zero prefix".into()));
            }
            for s in &self.slots {
                if let StateSlot::Recurrent(r) = s {
                    backend.zero(&r.buf, 0)?;
                }
            }
        }
        for s in self.slots.iter_mut() {
            if let StateSlot::AttnKv(a) = s {
                a.len = len;
            }
        }
        self.len = len;
        self.tokens.truncate(len);
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
