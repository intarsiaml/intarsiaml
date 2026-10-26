//! A cheap per-rank ledger of the passes a rank ran and the collectives it issued (`INTARSIA_TP_COLL_TRACE=1`, lane tp2): when the
//! ranks of a group go out of step (one waits in a collective its peer never launches, then the bounded wait traps 0x1016), the
//! first pass where the two ledgers differ names the step that decided differently. `issued` counts the collective launches a rank
//! made eagerly plus the collectives each graph replay runs (a capture runs nothing), so equal ledgers mean equal collective counts.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::comm::Comm;

const RING: usize = 4096;

pub(super) struct CollTrace {
    pub(super) on: bool,
    issued: AtomicU64,
    passes: AtomicU64,
    ring: Mutex<VecDeque<String>>,
    /// When the group exposes its device launch counters on the host (INTARSIA_HIP_CTRS_HOST=1): every ledger line carries them.
    comm: Option<Arc<dyn Comm>>,
}

impl CollTrace {
    pub(super) fn new(on: bool) -> CollTrace {
        CollTrace { on, issued: AtomicU64::new(0), passes: AtomicU64::new(0), ring: Mutex::new(VecDeque::new()), comm: None }
    }

    pub(super) fn set_comm(&mut self, c: Option<Arc<dyn Comm>>) {
        self.comm = c;
    }

    /// One collective launched eagerly on this rank's stream.
    pub(super) fn issue(&self) {
        if self.on {
            self.issued.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn push(&self, mut line: String) {
        if let Some([a, b]) = self.comm.as_ref().and_then(|c| c.debug_counters()) {
            line.push_str(&format!(" dev={a}/{b}"));
        }
        let mut r = self.ring.lock().unwrap();
        if r.len() >= RING {
            r.pop_front();
        }
        r.push_back(line);
    }

    /// A graph replay that runs `comms` collectives.
    pub(super) fn replay(&self, kind: &str, rows: usize, comms: usize) {
        if !self.on {
            return;
        }
        let before = self.issued.fetch_add(comms as u64, Ordering::Relaxed);
        let n = self.passes.fetch_add(1, Ordering::Relaxed);
        self.push(format!("#{n} {kind} rows={rows} issued {before}->{}", before + comms as u64));
    }

    /// A pass over the layers (prefill chunk, solo eager step, batched eager step, or a capture); recorded when the guard drops.
    pub(super) fn pass(&self, kind: &'static str, rows: usize, capture: bool) -> Option<PassGuard<'_>> {
        self.on.then(|| PassGuard { t: self, kind, rows, capture, before: self.issued.load(Ordering::Relaxed) })
    }

    pub(super) fn dump(&self) -> Vec<String> {
        self.ring.lock().unwrap().iter().cloned().collect()
    }
}

pub(super) struct PassGuard<'a> {
    t: &'a CollTrace,
    kind: &'static str,
    rows: usize,
    capture: bool,
    before: u64,
}

impl Drop for PassGuard<'_> {
    fn drop(&mut self) {
        let n = self.t.passes.fetch_add(1, Ordering::Relaxed);
        let after = self.t.issued.load(Ordering::Relaxed);
        self.t.push(format!(
            "#{n} {}{} rows={} issued {}->{}",
            self.kind,
            if self.capture { "+capture" } else { "" },
            self.rows,
            self.before,
            after
        ));
    }
}
