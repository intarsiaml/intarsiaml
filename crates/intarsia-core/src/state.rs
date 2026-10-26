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
    /// The MTP drafter's position at `len` (`--spec mtp`): its KV length and pending hidden rows ([`MtpState`]).
    mtp: Option<MtpSnap>,
}

struct MtpSnap {
    kv_len: usize,
    rows: usize,
    buf: Option<Buffer>,
}

/// State of the MTP drafter (`--spec mtp`, `crate::model::mtp`) beside a sequence. The MTP layer's position `p` reads
/// the main model's final normed hidden state of position `p` and the token of position `p + 1` (vLLM's convention), so
/// it trails the main model: `kv.len` positions are in its KV cache, the hidden rows of positions `kv.len .. st.len` wait
/// in `pending` until the token after them is known. `kv.len + pending_rows == st.len` between passes.
pub struct MtpState {
    /// The MTP layer's attention cache (one `AttnKv` slot); `kv.len` = positions it has processed.
    pub kv: SequenceState,
    /// `[cap_rows, hidden]` (activation dtype): pending main-model hidden rows, oldest first.
    pub pending: Buffer,
    pub pending_rows: usize,
    pub cap_rows: usize,
    /// Bytes of one hidden row.
    pub row_bytes: usize,
    /// The current request drafts (`Seq::spec_on`): only then does each pass queue rows and run the head. A request
    /// that never drafts (sampled, constrained, traced) skips the head entirely and leaves the state unsynced.
    pub active: bool,
    /// `kv.len + pending_rows == st.len` with valid pending rows. False once a pass ran without the head (inactive) or
    /// a head pass failed; only an empty sequence (`truncate(0)`) syncs it again, so a drafting request on an unsynced
    /// state starts from an empty cache (`Seq::new`).
    pub synced: bool,
}

/// Recurrent state after each row of the last speculative verify pass (`Model::forward_verify` on a model with
/// recurrent slots): `bufs[r]` holds every recurrent slot (concatenated, slot order, like a snapshot) as it was after
/// row `r` of the pass that started at token count `base`; `saved` rows are valid (the pass's last row is the live
/// state). [`SequenceState::spec_rollback`] restores one of them.
#[derive(Default)]
pub struct SpecRows {
    pub bufs: Vec<Buffer>,
    pub base: usize,
    pub saved: usize,
}

/// State of the DFlash drafter (`--spec dflash`, `crate::model::dflash`) beside a sequence: the draft layers' context K/V.
/// Position `p`'s row is `K/V(hidden_norm(fc(target layer outputs at p)))` and is written right after the target pass that
/// processed position `p` (every pass, so a back-off step leaves no hole); `kv.len` rows are valid. Drafting needs
/// `kv.len == st.len`. A rollback of the target (`truncate`, `spec_rollback`) shortens `kv` to the same length.
pub struct DflashState {
    /// The draft layers' attention caches (one `AttnKv` slot per draft layer); `kv.len` = context rows ingested.
    pub kv: SequenceState,
    /// The current request drafts: only then does each pass ingest its rows. A request that never drafts skips the draft
    /// entirely and leaves the state unsynced.
    pub active: bool,
    /// `kv.len` covers the target's rows (`st.len` when the pass ends). False once a pass ran without ingest or an
    /// ingest failed or rows were restored from a prefix store; only `truncate(0)` syncs it again, so a drafting request
    /// on an unsynced state starts from an empty cache (`Seq::new`).
    pub synced: bool,
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
    /// Free snapshot buffers kept for reuse (`SequenceState::reserve_snapshots`): a batching engine allocates them once
    /// when it creates its slots, so serving never allocates device memory for a snapshot. Dropped snapshots return here.
    pub snap_pool: Vec<Buffer>,
    /// Tensor parallelism: the state of ranks 1.. (this struct is rank 0's). Created, truncated and snapshotted
    /// together with the primary; advanced by the rank's own forward pass.
    pub peers: Vec<PeerState>,
    /// Tensor parallelism across processes (`crate::tp_remote`): the remote rank's copy of this sequence. Truncated and
    /// snapshotted after the primary, advanced by the remote rank's forward pass.
    pub remote: Option<crate::tp_remote::RemoteSeq>,
    /// Prefill-clean watermark (prefix store, `crate::prefix_store`): rows `0..clean` were written by prefill passes of
    /// the engine's chunk schedule, none by a decode step, so their state is what a cold prefill of `tokens[..clean]`
    /// computes. Lowered by [`SequenceState::truncate`], set by [`SequenceState::load_prefix`]; the engine advances it.
    pub clean: usize,
    /// The MTP drafter's state (`--spec mtp` only).
    pub mtp: Option<Box<MtpState>>,
    /// The DFlash drafter's state (`--spec dflash` only).
    pub dflash: Option<Box<DflashState>>,
    /// Per-row recurrent state of the last verify pass (speculative decoding on a model with recurrent slots).
    pub spec_rows: SpecRows,
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
        Ok(SequenceState {
            id,
            slots,
            len: 0,
            cap,
            tokens: Vec::new(),
            snapshots: Vec::new(),
            snap_pool: Vec::new(),
            peers: Vec::new(),
            remote: None,
            clean: 0,
            mtp: None,
            dflash: None,
            spec_rows: SpecRows::default(),
        })
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
    ///
    /// Other ranks first, the primary last, so that a failure leaves no rank with a snapshot the primary lacks (a rank
    /// holding extra snapshots is harmless: [`SequenceState::truncate`] checks the primary's before it asks any other
    /// rank). The remote rank's failure is returned from here, as the remote rank answers.
    pub fn snapshot(&mut self, backend: &dyn Backend, keep: usize) -> Result<()> {
        if let Some(r) = &self.remote {
            r.snapshot(keep)?;
        }
        for p in &mut self.peers {
            p.st.snapshot(&*p.backend, keep)?;
        }
        let total = self.recurrent_bytes();
        if total == 0 || self.len == 0 || keep == 0 {
            return Ok(());
        }
        if let Some(i) = self.snapshots.iter().position(|s| s.len == self.len) {
            let old = self.snapshots.remove(i);
            self.snap_pool.push(old.buf);
        }
        // at most `keep` snapshots: the oldest ones' buffers become free ...
        while self.snapshots.len() >= keep {
            let old = self.snapshots.remove(0);
            self.snap_pool.push(old.buf);
        }
        // ... and a free buffer is reused before any device allocation (none while serving, once reserved)
        let reuse = self.snap_pool.pop();
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
        let mtp = match &self.mtp {
            Some(m) => {
                let bytes = m.pending_rows * m.row_bytes;
                let buf = if bytes > 0 {
                    let b = backend.alloc(bytes, MemKind::Device)?;
                    backend.copy(&b, 0, &m.pending, 0, bytes, 0)?;
                    Some(b)
                } else {
                    None
                };
                Some(MtpSnap { kv_len: m.kv.len, rows: m.pending_rows, buf })
            }
            None => None,
        };
        self.snapshots.push(RecurrentSnapshot { len: self.len, buf, mtp });
        Ok(())
    }

    /// Drop the snapshots taken past `len` tokens; their buffers go back to the free pool.
    fn drop_snapshots_above(&mut self, len: usize) {
        let mut i = 0;
        while i < self.snapshots.len() {
            if self.snapshots[i].len > len {
                let s = self.snapshots.remove(i);
                self.snap_pool.push(s.buf);
            } else {
                i += 1;
            }
        }
    }

    /// Allocate the snapshot buffers up front so that `keep` snapshots (plus the live state) exist without any device
    /// allocation while serving (the batching engine's slots; no-op without recurrent slots). Idempotent.
    pub fn reserve_snapshots(&mut self, backend: &dyn Backend, keep: usize) -> Result<()> {
        let total = self.recurrent_bytes();
        if total == 0 {
            return Ok(());
        }
        while self.snap_pool.len() + self.snapshots.len() < keep {
            self.snap_pool.push(backend.alloc(total, MemKind::Device)?);
        }
        Ok(())
    }

    /// Byte offset of recurrent slot `slot` in a snapshot / [`SpecRows`] buffer (recurrent slots concatenated in slot order).
    pub fn recurrent_offset(&self, slot: usize) -> usize {
        self.slots[..slot]
            .iter()
            .map(|s| match s {
                StateSlot::Recurrent(r) => r.bytes,
                _ => 0,
            })
            .sum()
    }

    /// Roll back to `keep` tokens inside the last speculative verify pass (`base < keep <= len`): attention caches and
    /// the MTP drafter shorten, recurrent slots take the state saved after row `keep - base - 1` of that pass
    /// ([`SpecRows`]). Single-rank states only.
    pub fn spec_rollback(&mut self, keep: usize, backend: &dyn Backend) -> Result<()> {
        if keep == self.len {
            return Ok(());
        }
        if keep > self.len {
            return Err(Error::invalid("spec rollback beyond the current length"));
        }
        if !self.peers.is_empty() || self.remote.is_some() {
            return Err(Error::Unsupported("speculative rollback of a tensor-parallel state".into()));
        }
        if self.has_recurrent() {
            let sr = &self.spec_rows;
            let m = keep.checked_sub(sr.base).filter(|m| (1..=sr.saved).contains(m)).ok_or_else(|| {
                Error::invalid(format!(
                    "cannot roll recurrent state back to {keep} tokens (last verify pass from {} saved {} rows)",
                    sr.base, sr.saved
                ))
            })?;
            let src = &sr.bufs[m - 1];
            let mut off = 0;
            for s in &self.slots {
                if let StateSlot::Recurrent(r) = s {
                    backend.copy(&r.buf, 0, src, off, r.bytes, 0)?;
                    off += r.bytes;
                }
            }
            // the live state is now row m - 1's; the rows before it can still be returned to (a stop inside the run)
            self.spec_rows.saved = m - 1;
        }
        self.drop_snapshots_above(keep);
        for s in self.slots.iter_mut() {
            if let StateSlot::AttnKv(a) = s {
                a.len = keep;
            }
        }
        self.len = keep;
        self.tokens.truncate(keep);
        if let Some(m) = &mut self.mtp {
            if m.kv.len > keep {
                m.kv.truncate(keep, backend)?;
            }
            m.pending_rows = keep - m.kv.len;
        }
        if let Some(d) = &mut self.dflash {
            if d.kv.len > keep {
                d.kv.truncate(keep, backend)?;
            }
        }
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
    ///
    /// All-or-nothing across ranks as far as a rank can refuse: the primary and the in-process peers check that they can
    /// make the rollback ([`SequenceState::check_truncate`]) before any of them changes, and the remote rank, which
    /// checks for itself, is asked next and confirms before the primary changes. An error from a check or from the
    /// remote rank's refusal leaves every rank as it was; a device error part-way is fatal for the group.
    pub fn truncate(&mut self, len: usize, backend: &dyn Backend) -> Result<()> {
        self.check_truncate(len)?;
        if let Some(r) = &mut self.remote {
            r.truncate(len)?;
        }
        for p in &mut self.peers {
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
        if len != self.len || len == 0 {
            let snap = self.snapshots.iter().find(|s| s.len == len).and_then(|s| s.mtp.as_ref());
            if let Some(m) = &mut self.mtp {
                if len == 0 {
                    m.kv.truncate(0, backend)?;
                    m.pending_rows = 0;
                    m.synced = true;
                } else if !m.synced {
                    // nothing to keep consistent: the next drafting request starts from an empty sequence
                } else if let Some(ms) = snap.filter(|ms| ms.kv_len + ms.rows == len && ms.rows <= m.cap_rows) {
                    m.kv.truncate(ms.kv_len.min(m.kv.len), backend)?;
                    if m.kv.len != ms.kv_len {
                        return Err(Error::invalid("MTP snapshot ahead of the MTP cache"));
                    }
                    if let Some(b) = &ms.buf {
                        backend.copy(&m.pending, 0, b, 0, ms.rows * m.row_bytes, 0)?;
                    }
                    m.pending_rows = ms.rows;
                } else if len >= m.kv.len {
                    m.pending_rows = len - m.kv.len;
                } else {
                    // the hidden row of position len - 1 is gone: the drafter reads a zero row there once (drafts only,
                    // never the output)
                    m.kv.truncate(len - 1, backend)?;
                    backend.zero(&m.pending, 0)?;
                    m.pending_rows = 1;
                }
            }
        }
        if let Some(d) = &mut self.dflash {
            if len == 0 {
                d.kv.truncate(0, backend)?;
                d.synced = true;
            } else if d.kv.len > len {
                d.kv.truncate(len, backend)?;
            }
        }
        self.spec_rows.saved = 0;
        self.drop_snapshots_above(len);
        for s in self.slots.iter_mut() {
            if let StateSlot::AttnKv(a) = s {
                a.len = len;
            }
        }
        self.len = len;
        self.tokens.truncate(len);
        self.clean = self.clean.min(len);
        Ok(())
    }

    /// After the prefix store wrote every slot for `tokens` (`crate::prefix_store::PrefixStore::restore`): take that
    /// length, drop the snapshots (they belonged to the state that was overwritten) and mark the rows prefill-clean.
    pub fn load_prefix(&mut self, tokens: &[u32]) {
        let len = tokens.len();
        self.len = len;
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        self.snapshots.clear();
        for s in self.slots.iter_mut() {
            if let StateSlot::AttnKv(a) = s {
                a.len = len;
            }
        }
        self.clean = len;
        // the restored rows never went through the MTP head: its cache no longer covers the sequence
        if let Some(m) = &mut self.mtp {
            m.synced = false;
        }
        if let Some(d) = &mut self.dflash {
            d.synced = false;
        }
    }

    /// Advance a draft KV state by `n` ingested rows (every attention slot): the DFlash context cache holds no tokens.
    pub fn advance_rows(&mut self, n: usize) {
        self.len += n;
        for s in self.slots.iter_mut() {
            if let StateSlot::AttnKv(a) = s {
                a.len = self.len;
            }
        }
    }

    /// Whether [`SequenceState::truncate`] to `len` would pass its own checks here and on the in-process peers: not
    /// beyond a length, and a recurrent snapshot exists where one is needed. Changes nothing.
    fn check_truncate(&self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(Error::invalid("truncate beyond current length"));
        }
        for p in &self.peers {
            if len > p.st.len {
                return Err(Error::invalid("truncate beyond a peer rank's length"));
            }
            p.st.check_truncate(len)?;
        }
        if self.has_recurrent() && len != 0 && len != self.len && !self.snapshots.iter().any(|s| s.len == len) {
            return Err(Error::Unsupported(format!(
                "cannot roll recurrent state back to {len} tokens (no snapshot there; length {})",
                self.len
            )));
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
