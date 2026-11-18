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
//! launches (CUDA: `INTARSIA_CUDA_VERIFY_ATTN_ROWS=1` also takes them, see [`VerifyCapTab`]). `INTARSIA_HIP_VERIFY_ATTN_ROWS=0` keeps them everywhere.

use super::*;
use crate::state::AttnKvState;
use std::collections::HashMap;

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

/// Lane cvat (CUDA, `INTARSIA_CUDA_VERIFY_ATTN_ROWS=1`): the tables of CAPTURED passes. A recording bakes the address of its K / V table
/// region into the graph, so a region must never be rewritten while a graph that names it can still be replayed: regions are keyed by
/// the (K, V) cache addresses they hold and allocated once, in a buffer of its own, before the capture starts
/// ([`Model::register_verify_cap_tabs`]); a capture that meets a cache pair without a region keeps the per-row launches. Positions are
/// the capture's own device position table (row `r` reads entry `r`), nothing is uploaded in a capture.
#[derive(Default)]
pub(super) struct VerifyCapTab {
    buf: Option<Buffer>,
    regions: HashMap<(u64, u64), usize>,
}

/// Regions of [`VerifyCapTab`] (16 rows of (K, V) addresses each): 256 KiB of device memory at most.
const CAP_REGIONS_MAX: usize = 1024;

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
    /// Give every attention cache of `st` (and of the MTP head `head_kv`) its K / V table region, before a graph capture starts
    /// (`INTARSIA_CUDA_VERIFY_ATTN_ROWS=1` only). A pair already registered keeps its region; at most [`CAP_REGIONS_MAX`] exist.
    pub(super) fn register_verify_cap_tabs(&self, st: &SequenceState, head_kv: Option<&SequenceState>) -> Result<()> {
        if !self.cuda_verify_attn_rows {
            return Ok(());
        }
        let mut tab = self.verify_cap_tab.lock().unwrap();
        for s in st.slots.iter().chain(head_kv.into_iter().flat_map(|h| h.slots.iter())) {
            let StateSlot::AttnKv(kv) = s else { continue };
            let (Some(ka), Some(va)) = (self.backend.device_address(&kv.k), self.backend.device_address(&kv.v)) else { continue };
            if tab.regions.contains_key(&(ka, va)) {
                continue;
            }
            if tab.regions.len() >= CAP_REGIONS_MAX {
                note_once("captured: all table regions taken, per-row launches for caches registered later");
                continue;
            }
            if tab.buf.is_none() {
                tab.buf = Some(self.backend.alloc(CAP_REGIONS_MAX * REGION_BYTES, MemKind::Device)?);
            }
            let buf = tab.buf.clone().expect("allocated above");
            let idx = tab.regions.len();
            self.backend.upload(&buf, idx * REGION_BYTES, &verify_rows_tab_bytes(ka, va, VERIFY_ATTN_ROWS_MAX), self.stream)?;
            tab.regions.insert((ka, va), idx);
        }
        Ok(())
    }

    /// A captured verify pass's `t` attention rows of one layer as one row-table launch (`INTARSIA_CUDA_VERIFY_ATTN_ROWS=1`): row `r`
    /// reads entry `r` of the capture's device position table `dpos` and the cache pair's region of [`VerifyCapTab`]. `Ok(false)`
    /// (nothing recorded): the knob is off, the pass is outside the table's reach or the pair has no region; the caller records per-row launches.
    #[allow(clippy::too_many_arguments)]
    fn verify_attn_rows_captured(
        &self,
        slot: &AttnKvState,
        q: &TensorRef,
        kcache: &TensorRef,
        vcache: &TensorRef,
        out: &TensorRef,
        attrs: &OpAttrs,
        t: usize,
        dpos: &TensorRef,
        profile: bool,
    ) -> Result<bool> {
        if !self.cuda_verify_attn_rows || t < 2 || t > VERIFY_ATTN_ROWS_MAX || dpos.dtype != DType::I32 || self.tp.world != 1 || !self.peers.is_empty() {
            return Ok(false);
        }
        let (Some(ka), Some(va)) = (self.backend.device_address(&slot.k), self.backend.device_address(&slot.v)) else {
            return Ok(false);
        };
        let (buf, idx) = {
            let tab = self.verify_cap_tab.lock().unwrap();
            match (tab.buf.clone(), tab.regions.get(&(ka, va))) {
                (Some(b), Some(&i)) => (b, i),
                _ => {
                    note_once("captured: a cache without a table region, per-row launches");
                    return Ok(false);
                }
            }
        };
        let mut tab = self.view(&buf, DType::U8, &[t, 2, 8]);
        tab.offset = idx * REGION_BYTES;
        let mut pos = dpos.clone();
        pos.shape = vec![t];
        let mut at = attrs.clone();
        at.pos0 = 0;
        let mut call = OpCall::new(Op::AttnDecode, vec![q.clone(), kcache.clone(), vcache.clone()], vec![out.clone()], at);
        call.pos = Some(pos);
        call.rows = Some(crate::ops::RowTab { ptrs: tab, rows: t, stride: 2, index: 0, byte_off: 0 });
        call.exact_rows = self.verifying();
        let need = match call.check().and_then(|_| self.registry.select(&call).map(|k| k.scratch_bytes(&ShapeInfo::from_call(&call)))) {
            Ok(n) => n,
            Err(e) => {
                note_once(&format!("captured: no kernel for row-table call: {e}"));
                return Ok(false);
            }
        };
        if need > self.scratch.as_ref().map_or(0, |b| b.bytes()) {
            note_once("captured: scratch smaller than the row-table call needs");
            return Ok(false);
        }
        note_once("captured: engaged (one row-table attention launch per layer in the graph)");
        self.exec_call(call, profile)?;
        Ok(true)
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
        if let (Some(d), true) = (dpos, capturing) {
            return self.verify_attn_rows_captured(slot, q, kcache, vcache, out, attrs, t, d, profile);
        }
        if !self.verify_attn_rows
            || dpos.is_some()
            || t < 2
            || t > VERIFY_ATTN_ROWS_MAX
            || !self.backend.verify_attn_per_row()
            || !self.backend.supports_row_tables()
            || self.capture_ops.lock().unwrap().is_some()
        {
            return Ok(false);
        }
        let n_slots = self.slot_specs.len().max(1);
        if a.slot >= n_slots {
            return Ok(false);
        }
        let recording = self.recorder.lock().unwrap().is_some();
        let (tab, pos) = {
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
        note_once("engaged (one row-table attention launch per layer)");
        self.exec_call(call, profile)?;
        Ok(true)
    }
}
