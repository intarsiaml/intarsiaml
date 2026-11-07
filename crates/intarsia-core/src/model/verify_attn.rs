//! One attention launch per layer for the rows of a speculative verify pass (lane vat).
//!
//! The verify pass of a backend whose multi-row attention is not bitwise the decode step's (HIP) runs `AttnDecode` once per row
//! (row `r` at position `pos0 + r`). The row-table decode kernel (`OpCall::rows`, the batched-decode kernel, bitwise the solo
//! one-row launch) does the same arithmetic for all rows in one launch: here every row of the table names the SAME sequence's K / V
//! cache and row `r` reads position `pos0 + r`, so row `r` sees the prefix plus the rows `<= r` exactly as its own launch did.
//!
//! The tables are small and almost constant: each attention layer owns a 16-row region of device addresses (rewritten only when the
//! layer's cache buffers change), and one `i32` position vector is shared by all layers (rewritten when `(pos0, t)` changes, i.e.
//! once per pass, by the pass's first attention layer). Eager passes only: a captured pass (device positions) keeps the per-row
//! launches. `INTARSIA_HIP_VERIFY_ATTN_ROWS=0` keeps them everywhere.

use super::*;
use crate::state::AttnKvState;

/// Most rows one verify launch covers (the row-table kernels' `ROWS_MAX`).
pub(super) const VERIFY_ATTN_ROWS_MAX: usize = 16;
/// Bytes of the position vector at the start of the table buffer.
const POS_BYTES: usize = 256;
/// Bytes of one layer's region: `VERIFY_ATTN_ROWS_MAX` rows of (K address, V address).
const REGION_BYTES: usize = VERIFY_ATTN_ROWS_MAX * 2 * 8;

/// The verify tables' device buffer and what it currently holds.
#[derive(Default)]
pub(super) struct VerifyAttnTab {
    buf: Option<Buffer>,
    /// Per state slot: the (K, V) cache addresses its region holds.
    filled: Vec<Option<(u64, u64)>>,
    /// `(pos0, t)` the position vector holds.
    pos_key: Option<(usize, usize)>,
}

/// Lane c27: the K / V address tables of the captured speculative step (`INTARSIA_VERIFY_ATTN_ROWS_CAPTURED=1`). The shared region of [`VerifyAttnTab`]
/// is rewritten by every eager verify pass of any sequence, so a graph that read it would replay another sequence's addresses; each capture instead owns
/// a table buffer, written eagerly before the capture starts (one 16-row region per attention cache of the sequence and of its MTP head, found by
/// the cache's addresses) and kept alive here for as long as the graph can replay (graphs are bound to their sequence; the oldest tables go first).
#[derive(Default)]
pub(super) struct CaptTabs {
    /// The table the capture in progress reads: its buffer and the (K, V) addresses of each region.
    cur: Option<(Buffer, Vec<(u64, u64)>)>,
    /// Tables of finished captures, by sequence id, oldest first.
    keep: std::collections::VecDeque<(u64, Buffer)>,
}

const CAPT_TABS_KEEP: usize = 256;

/// Planted faults for the byte-equality test (`INTARSIA_HIP_VAT_MUT`): 1 = every row reads `pos0` (no `+ r`); 2 = the last row's
/// position is one too small; 3 = the region's rows past the first hold the K address as their V address; 4 = the position vector is not rewritten
/// when only `t` changes. Unset (every real run): none.
fn mutant() -> u32 {
    std::env::var("INTARSIA_HIP_VAT_MUT").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// The `t` positions of a verify pass starting at `pos0`: `i32` little-endian, row `r` at `pos0 + r`.
pub fn verify_rows_pos_bytes(pos0: usize, t: usize) -> Vec<u8> {
    let m = mutant();
    (0..t)
        .flat_map(|r| {
            let p = match m {
                1 => pos0,
                2 if r + 1 == t && r > 0 => pos0 + r - 1,
                _ => pos0 + r,
            };
            (p as i32).to_le_bytes()
        })
        .collect()
}

/// One layer's region of the table for `t` rows: every row holds the same (K, V) cache addresses (`[t, 2]` little-endian u64).
pub fn verify_rows_tab_bytes(k: u64, v: u64, t: usize) -> Vec<u8> {
    let m = mutant();
    let mut out = Vec::with_capacity(t * 16);
    for r in 0..t {
        out.extend_from_slice(&k.to_le_bytes());
        out.extend_from_slice(&(if m == 3 && r > 0 { k } else { v }).to_le_bytes());
    }
    out
}

/// Whether a position vector for `(pos0, t)` must be written when the buffer holds `held` (mutant 4: only `pos0` is compared).
pub fn verify_rows_pos_stale(held: Option<(usize, usize)>, pos0: usize, t: usize) -> bool {
    if mutant() == 4 {
        return held.is_none_or(|h| h.0 != pos0);
    }
    held != Some((pos0, t))
}

/// Said once per reason on stderr (`grep 'verify attention rows' ` in a run's log tells whether the path engaged or fell back).
fn note_once(what: &str) {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if !seen.iter().any(|w| w == what) {
        eprintln!("verify attention rows: {what}");
        seen.push(what.to_string());
    }
}

impl Model {
    /// Before a capture of the speculative step of `st` (and its MTP head cache `head`): write the K / V address regions the capture's row-table
    /// attention launches will read (`INTARSIA_VERIFY_ATTN_ROWS_CAPTURED=1`; nothing otherwise). Eager host writes, so they cannot happen inside the capture.
    pub(super) fn capt_tabs_begin(&self, st: &SequenceState, head: Option<&SequenceState>) -> Result<()> {
        if !self.verify_attn_rows_captured || !self.backend.verify_attn_per_row() || !self.backend.supports_row_tables() {
            return Ok(());
        }
        let mut addrs: Vec<(u64, u64)> = Vec::new();
        for s in st.slots.iter().chain(head.into_iter().flat_map(|h| h.slots.iter())) {
            if let StateSlot::AttnKv(a) = s {
                match (self.backend.device_address(&a.k), self.backend.device_address(&a.v)) {
                    (Some(k), Some(v)) => addrs.push((k, v)),
                    _ => {
                        note_once("no device address for a KV cache buffer");
                        return Ok(());
                    }
                }
            }
        }
        if addrs.is_empty() {
            return Ok(());
        }
        let buf = self.backend.alloc(addrs.len() * REGION_BYTES, MemKind::Device)?;
        for (i, &(k, v)) in addrs.iter().enumerate() {
            self.backend.upload(&buf, i * REGION_BYTES, &verify_rows_tab_bytes(k, v, VERIFY_ATTN_ROWS_MAX), self.stream)?;
        }
        self.capt_tabs.lock().unwrap().cur = Some((buf, addrs));
        Ok(())
    }

    /// After the capture: the table now belongs to the graph (kept for `st_id`).
    pub(super) fn capt_tabs_end(&self, st_id: u64) {
        let mut ct = self.capt_tabs.lock().unwrap();
        if let Some((buf, _)) = ct.cur.take() {
            ct.keep.push_back((st_id, buf));
            while ct.keep.len() > CAPT_TABS_KEEP {
                ct.keep.pop_front();
            }
        }
    }

    /// The verify pass's `t` attention rows of layer `a` as one row-table launch. `Ok(false)` (nothing launched): the knob is off, the
    /// backend has no such kernel, or the pass is captured or exceeds the table; the caller launches per row.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_attn_rows(
        &self,
        a: &AttnBlock,
        slot: &AttnKvState,
        q: &TensorRef,
        kcache: &TensorRef,
        vcache: &TensorRef,
        out: &TensorRef,
        attrs: &OpAttrs,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
        profile: bool,
    ) -> Result<bool> {
        let capturing = self.capture_ops.lock().unwrap().is_some();
        if !self.verify_attn_rows
            || t < 2
            || t > VERIFY_ATTN_ROWS_MAX
            || !self.backend.verify_attn_per_row()
            || !self.backend.supports_row_tables()
            // a captured pass only with the opt-in, and only with the step's device positions (row r reads entry r)
            || (capturing && !(self.verify_attn_rows_captured && dpos.is_some()))
        {
            return Ok(false);
        }
        let recording = self.recorder.lock().unwrap().is_some();
        let (tab, pos) = if capturing {
            // the capture's own table (see `CaptTabs`) and the step's device positions (entry r is row r's position, the value the per-row launch reads)
            let (Some(ka), Some(va)) = (self.backend.device_address(&slot.k), self.backend.device_address(&slot.v)) else {
                return Ok(false);
            };
            let found = {
                let ct = self.capt_tabs.lock().unwrap();
                ct.cur.as_ref().and_then(|(buf, addrs)| addrs.iter().position(|&x| x == (ka, va)).map(|i| (buf.clone(), i)))
            };
            let Some((buf, i)) = found else {
                note_once("captured row-table attention: no table for this cache, per-row launches");
                return Ok(false);
            };
            let mut tab = self.view(&buf, DType::U8, &[t, 2, 8]);
            tab.offset = i * REGION_BYTES;
            let mut pos = dpos.expect("checked above").clone();
            pos.shape = vec![t];
            (tab, pos)
        } else {
            let n_slots = self.slot_specs.len().max(1);
            if a.slot >= n_slots {
                return Ok(false);
            }
            let mut st = self.verify_tab.lock().unwrap();
            if st.buf.is_none() {
                st.buf = Some(self.backend.alloc(POS_BYTES + n_slots * REGION_BYTES, MemKind::Device)?);
                st.filled = vec![None; n_slots];
                st.pos_key = None;
            }
            let buf = st.buf.clone().expect("allocated above");
            if !recording {
                let addr = |b: &Buffer| self.backend.device_address(b);
                let (Some(ka), Some(va)) = (addr(&slot.k), addr(&slot.v)) else {
                    note_once("no device address for a KV cache buffer");
                    return Ok(false);
                };
                if st.filled[a.slot] != Some((ka, va)) {
                    self.backend.upload(&buf, POS_BYTES + a.slot * REGION_BYTES, &verify_rows_tab_bytes(ka, va, VERIFY_ATTN_ROWS_MAX), self.stream)?;
                    st.filled[a.slot] = Some((ka, va));
                }
                if verify_rows_pos_stale(st.pos_key, pos0, t) {
                    self.backend.upload(&buf, 0, &verify_rows_pos_bytes(pos0, t), self.stream)?;
                    st.pos_key = Some((pos0, t));
                }
            }
            let mut tab = self.view(&buf, DType::U8, &[t, 2, 8]);
            tab.offset = POS_BYTES + a.slot * REGION_BYTES;
            (tab, self.view(&buf, DType::I32, &[t]))
        };
        // the positions are the call's `pos` tensor; `attrs.pos0` stays 0 as in the batched row-table calls (`check` would otherwise
        // bound `pos0 + t` by the cache capacity, which the last rows of a full cache exceed in the call's own terms)
        let mut at = attrs.clone();
        at.pos0 = 0;
        let mut call = OpCall::new(Op::AttnDecode, vec![q.clone(), kcache.clone(), vcache.clone()], vec![out.clone()], at);
        call.pos = Some(pos);
        call.rows = Some(crate::ops::RowTab { ptrs: tab, rows: t, stride: 2, index: 0, byte_off: 0 });
        call.exact_rows = self.verifying();
        let need = match call.check().and_then(|_| self.registry.select(&call).map(|k| k.scratch_bytes(&ShapeInfo::from_call(&call)))) {
            Ok(n) => n,
            Err(e) => {
                note_once(&format!("no kernel for the row-table call: {e}"));
                return Ok(false);
            }
        };
        // the plan sizes the scratch from the verify probes it recorded with this path; a pass it did not see falls back
        if !recording && need > self.scratch.as_ref().map_or(0, |b| b.bytes()) {
            note_once("scratch smaller than the row-table call needs");
            return Ok(false);
        }
        note_once(if capturing { "engaged in the captured step (one row-table attention launch per layer)" } else { "engaged (one row-table attention launch per layer)" });
        self.exec_call(call, profile)?;
        Ok(true)
    }
}
