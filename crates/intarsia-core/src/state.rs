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
    /// Rows of single-token decode steps whose features sit in [`DflashState::store`] and are not yet in `kv`
    /// (positions `kv.len .. kv.len + pending`): a plain step stashes its row (a copy) instead of running the draft's context
    /// projection, which a long fallback stretch of the cost gate would pay for nothing. Flushed in one batch before the next draft.
    pub pending: usize,
    /// Per-tap feature rows `[PEND_MAX, H]` of the pending rows (allocated at the first stash).
    pub store: Vec<crate::backend::Buffer>,
    /// `--spec auto` (lane asp): every pass stashes its feature rows instead of ingesting them, so a pass the MTP head
    /// drafted costs the draft only a copy; the rows (a prefix of the stash survives a rollback) go in as one batch before the
    /// next DFlash draft.
    pub lazy: bool,
}

impl DflashState {
    /// The target rolled back to `len` tokens: the context cache and the pending rows follow it.
    pub fn cut(&mut self, len: usize, backend: &dyn crate::backend::Backend) -> crate::error::Result<()> {
        if len < self.kv.len {
            self.kv.truncate(len, backend)?;
            self.pending = 0;
        } else {
            self.pending = self.pending.min(len - self.kv.len);
        }
        Ok(())
    }
}

static NEXT_STATE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A tensor-parallel peer rank's share of a sequence: its own slots on its own device.
pub struct PeerState {
    pub backend: Arc<dyn Backend>,
    pub st: SequenceState,
}

/// What kind of forward pass wrote a [`Pass`]'s rows: which kernels' bits they carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    /// An ordinary pass (a prefill chunk, a packed prefill segment, or a pass whose caller tagged nothing): its rows carry the
    /// bits of a `rows`-row pass on this backend.
    Plain,
    /// A row-invariant prefill pass (lane rip, `Model::forward_prefill` with `Model::row_invariant_prefill` on): each row has
    /// the bits a one-pass prefill of the prompt gives it, whatever the pass's row count or first row.
    RowInvariant,
    /// A decode step (`rows` 1): decode kernels, not a prefill pass's, so a cold prefill would give these rows other bits.
    Decode,
    /// A row-exact speculative-verify pass (`rows` 1: each row is what a decode step writes, whatever the pass's row count).
    Verify,
}

/// One forward pass that wrote the KV rows `start..end` ([`SequenceState::passes`]). `rows` is the row count whose kernels'
/// bits the rows carry: the pass's own row count for an ordinary or row-invariant pass, 1 for a decode step or a row-exact
/// speculative-verify pass (their kernels give a row the bits of a 1-row pass whatever the row count); `kind` says which.
/// A truncate (or a speculative rollback) that cuts a pass clips `end` and keeps `rows` and `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub start: usize,
    pub end: usize,
    pub rows: usize,
    pub kind: PassKind,
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
    /// The forward passes that wrote rows `0..len`, in order: one entry per committed pass ([`SequenceState::commit`]),
    /// with its row count and kind. A truncate or a speculative rollback drops the passes past the new length and clips the
    /// one it cuts through, a prefix-store restore leaves none. Exact prefix reuse on a chunk-sensitive backend reads it
    /// ([`SequenceState::ladder_reuse_len`], [`SequenceState::wide_reuse_len`], [`SequenceState::inv_len`]).
    pub passes: Vec<Pass>,
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
            passes: Vec::new(),
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
    /// ([`SpecRows`]). Every in-process tensor-parallel rank rolls back with it (each from its own saved rows).
    pub fn spec_rollback(&mut self, keep: usize, backend: &dyn Backend) -> Result<()> {
        if keep == self.len {
            return Ok(());
        }
        if keep > self.len {
            return Err(Error::invalid("spec rollback beyond the current length"));
        }
        if self.remote.is_some() {
            return Err(Error::Unsupported("speculative rollback of a state with a rank in another process".into()));
        }
        // tensor parallelism (in-process peers, lane dfh): every rank ran the same verify pass and saved its own slice of the
        // recurrent state per row; all check before any changes, so a refusal leaves every rank as it was
        self.check_spec_rollback(keep)?;
        for p in &self.peers {
            if p.st.len != self.len {
                return Err(Error::backend(format!("tensor-parallel ranks diverged ({} vs {} tokens)", p.st.len, self.len)));
            }
            p.st.check_spec_rollback(keep)?;
        }
        for p in &mut self.peers {
            p.st.spec_rollback_rank(keep, &*p.backend)?;
        }
        self.spec_rollback_rank(keep, backend)
    }

    /// Whether [`SequenceState::spec_rollback`] to `keep` can be made on this rank's state.
    fn check_spec_rollback(&self, keep: usize) -> Result<()> {
        if self.has_recurrent() {
            let sr = &self.spec_rows;
            keep.checked_sub(sr.base).filter(|m| (1..=sr.saved).contains(m)).ok_or_else(|| {
                Error::invalid(format!(
                    "cannot roll recurrent state back to {keep} tokens (last verify pass from {} saved {} rows)",
                    sr.base, sr.saved
                ))
            })?;
        }
        Ok(())
    }

    /// [`SequenceState::spec_rollback`] on this rank's state alone (checked by [`SequenceState::check_spec_rollback`]).
    fn spec_rollback_rank(&mut self, keep: usize, backend: &dyn Backend) -> Result<()> {
        if self.has_recurrent() {
            let sr = &self.spec_rows;
            let m = keep - sr.base;
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
        self.clip_passes(keep);
        self.tokens.truncate(keep);
        if let Some(m) = &mut self.mtp {
            if m.kv.len > keep {
                m.kv.truncate(keep, backend)?;
            }
            m.pending_rows = keep - m.kv.len;
        }
        if let Some(d) = &mut self.dflash {
            d.cut(keep, backend)?;
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

    /// Exact prefix reuse on a backend whose prefill bits depend on a pass's row count (`Backend::prefix_reuse_any_cut`
    /// false: CUDA, Metal): the longest prefix `<= want` a prompt may resume from and still compute its cold prefill's bits. That is
    /// 0, or one of `points` (the ends of the prompt's non-final cold chunks) whose rows `0..L` this state wrote with exactly
    /// the cold schedule's passes, the full `chunk`-row passes `(0, chunk), (chunk, 2 chunk), .., (L - chunk, L)`. Rows a
    /// decode step, a busy-size chunk or a resumed prompt wrote never qualify, nor does a prompt's own last token.
    pub fn ladder_reuse_len(&self, want: usize, chunk: usize, points: &[usize]) -> usize {
        let want = want.min(self.len);
        let mut best = 0;
        for (i, p) in self.passes.iter().enumerate() {
            if (p.start, p.end) != (i * chunk, (i + 1) * chunk) || p.rows != chunk || p.end > want {
                break;
            }
            if points.contains(&p.end) {
                best = p.end;
            }
        }
        best
    }

    /// Exact prefix reuse of a recurrent state on a backend whose prefill bits depend on a pass's row count
    /// (`Backend::prefix_reuse_any_cut` false): the longest `L <= want` the state can return to
    /// ([`SequenceState::restorable_len`]) that is one of `cuts` (where the prompt's own cold prefill leaves its state,
    /// `scheduler::recurrent_schedule_cuts`) and whose rows `0..L` prefill passes wrote in an unbroken run from row 0, each
    /// ending at one of `cuts`. The rows then came from the cold schedule's own passes, so the state at `L` and the remainder's
    /// passes are the cold run's. Anywhere else the cold run has no state at `L` and its passes cut elsewhere: lane hrr
    /// (docs/results/2026-10-29-hrr-hybrid-reuse.md) measured a resume at the previous prompt's `plen - 1` changing the
    /// 35B's greedy text on Metal. Rows a decode step, a verify pass or a prefix-store restore (no pass) wrote end the run.
    pub fn recurrent_reuse_len(&self, want: usize, cuts: &[usize]) -> usize {
        let want = want.min(self.len);
        let mut best = 0;
        let mut end = 0;
        for p in &self.passes {
            if p.start != end || p.end > want || !matches!(p.kind, PassKind::Plain | PassKind::RowInvariant) || !cuts.contains(&p.end) {
                break;
            }
            end = p.end;
            if self.restorable_len(end) == end {
                best = end;
            }
        }
        best
    }

    /// Exact prefix reuse on a backend whose passes of `min_rows` rows or more write rows bitwise equal to each other's and
    /// to a one-pass prefill's, whatever the split (`Backend::prefix_reuse_min_pass_rows`; Metal: only a 1-row pass
    /// differs): the longest prefix `<= want` whose rows this state wrote with such passes only, in an unbroken run from
    /// row 0. A decode step or a row-exact verify pass (`rows` 1), the rows of a prefix-store restore (no pass) and a gap
    /// end it. The caller still has to prefill the rest in passes of `min_rows` rows or more.
    pub fn wide_reuse_len(&self, want: usize, min_rows: usize) -> usize {
        let want = want.min(self.len);
        let mut end = 0;
        for p in &self.passes {
            if end >= want || p.start != end || p.rows < min_rows {
                break;
            }
            end = p.end;
        }
        end.min(want)
    }

    /// Row-invariant prefix (lane rip): the rows `0..n` written by row-invariant prefill passes ([`PassKind::RowInvariant`])
    /// only, in an unbroken run from row 0, so their state is bitwise what a cold prefill of `tokens[..n]` in any chunks
    /// writes and a prompt sharing them may resume at any row below `n` (`Model::reuse_len`). A decode step, a verify pass,
    /// an ordinary pass, a gap and a prefix-store restore (no pass) end it.
    pub fn inv_len(&self) -> usize {
        let mut end = 0;
        for p in &self.passes {
            if p.start != end || p.kind != PassKind::RowInvariant {
                break;
            }
            end = p.end;
        }
        end.min(self.len)
    }

    /// Drop the passes at or past row `len` and clip the one that crosses it (truncate, speculative rollback).
    fn clip_passes(&mut self, len: usize) {
        self.passes.retain(|p| p.start < len);
        if let Some(p) = self.passes.last_mut() {
            p.end = p.end.min(len);
        }
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
                d.pending = 0;
                d.synced = true;
            } else {
                d.cut(len, backend)?;
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
        self.clip_passes(len);
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
        // no pass of this state wrote the restored rows: exact reuse never resumes inside them
        self.passes.clear();
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
        self.passes.push(Pass { start: self.len, end: self.len + n, rows: 1, kind: PassKind::Plain });
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

    /// Advance after a forward pass appended `ids` (an ordinary pass: [`PassKind::Plain`] of `ids.len()` rows).
    pub fn commit(&mut self, ids: &[u32]) {
        self.commit_pass(ids, ids.len(), PassKind::Plain);
    }

    /// [`SequenceState::commit`] for a decode step ([`PassKind::Decode`], 1 row per token).
    pub fn commit_decode(&mut self, ids: &[u32]) {
        self.commit_pass(ids, 1, PassKind::Decode);
    }

    /// [`SequenceState::commit`] for a row-exact speculative-verify pass: its rows carry the bits of 1-row kernels, not of
    /// `ids.len()`-row ones ([`PassKind::Verify`]; each row is what a decode step writes).
    pub fn commit_verify(&mut self, ids: &[u32]) {
        self.commit_pass(ids, 1, PassKind::Verify);
    }

    /// Re-tag the passes this state committed from row `from` on (the ones a single `Model::forward` call wrote, which may be
    /// several chunks) as `kind`; their row counts stay.
    pub fn tag_passes(&mut self, from: usize, kind: PassKind) {
        for p in self.passes.iter_mut().rev().take_while(|p| p.start >= from) {
            p.kind = kind;
        }
    }

    fn commit_pass(&mut self, ids: &[u32], rows: usize, kind: PassKind) {
        self.passes.push(Pass { start: self.len, end: self.len + ids.len(), rows, kind });
        self.len += ids.len();
        self.tokens.extend_from_slice(ids);
        for s in self.slots.iter_mut() {
            if let StateSlot::AttnKv(a) = s {
                a.len = self.len;
            }
        }
    }
}
