//! Run-to-run determinism tap (lane det, debugging, off by default): `INTARSIA_DET_TAP=<file>` copies the last (up to
//! [`ROWS`]) rows of every layer's output (residual stream and the pending block output, when there is one) of a
//! single-sequence pass into a per-rank device buffer, and after the pass appends one line per rank to `<file>`: `pos=<first position> t=<rows>
//! rank=<r> <hashes per layer>`, a layer's hashes (FNV-1a 64 of each row's residual and block bytes, oldest row first) joined by commas. Both ranks of a tensor-parallel group hold the same residual stream after every
//! all-reduce, so their lines must be equal; across runs the first layer whose hash differs is where the runs part.
//! The copies are device-to-device copies on the rank's stream (captured into a decode graph like the rest of the step); the
//! download after the pass waits for each rank's stream, which a peer rank's step otherwise does not.

use super::*;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

/// rows per layer the tap keeps (the last ones of a longer pass)
const ROWS: usize = 64;

struct Tap {
    file: Mutex<std::fs::File>,
    /// per rank: `[n_layers, 2 * ROWS, hidden]` in the activation dtype (residual rows, then block output rows)
    bufs: Mutex<Vec<Option<Buffer>>>,
    /// per rank: rows of the current pass already in the tap (a prefill may run as several chunks)
    rows: Mutex<Vec<usize>>,
}

fn tap() -> Option<&'static Tap> {
    static T: OnceLock<Option<Tap>> = OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var("INTARSIA_DET_TAP").ok().filter(|p| !p.is_empty())?;
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).expect("INTARSIA_DET_TAP: cannot open the file");
        eprintln!("det tap: layer output hashes -> {path}");
        Some(Tap { file: Mutex::new(file), bufs: Mutex::new(vec![None; 8]), rows: Mutex::new(vec![0; 8]) })
    })
    .as_ref()
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0100_0000_01b3))
}

impl Model {
    /// This rank's tap buffer, allocated (and zeroed) on first use: the first pass is a prefill, never a graph capture.
    fn det_tap_buf(&self, t: &Tap) -> Result<Buffer> {
        let mut g = t.bufs.lock().unwrap();
        if let Some(b) = &g[self.tp.rank] {
            return Ok(b.clone());
        }
        let b = self.backend.alloc(self.layers.len() * 2 * ROWS * self.dims.hidden * self.act.size_bytes(), MemKind::Device)?;
        self.backend.zero(&b, self.stream)?;
        g[self.tp.rank] = Some(b.clone());
        Ok(b)
    }

    /// After layer `li` of a chunk of `t` rows of a single-sequence pass: its output rows into the tap, after the rows of the
    /// pass's earlier chunks (the first [`ROWS`] rows of a pass are kept; no-op when off).
    pub(super) fn det_tap_layer(&self, li: usize, t: usize, res: &TensorRef, blk: Option<&TensorRef>) -> Result<()> {
        let Some(tp) = tap() else { return Ok(()) };
        let h = self.dims.hidden;
        let buf = self.det_tap_buf(tp)?;
        let c = tp.rows.lock().unwrap()[self.tp.rank];
        let n = t.min(ROWS - c);
        if li + 1 == self.layers.len() {
            tp.rows.lock().unwrap()[self.tp.rank] += n;
        }
        if n == 0 {
            return Ok(());
        }
        let row = h * self.act.size_bytes();
        // a plain device copy per tensor: residual rows in the layer's first ROWS rows, the pending block output's in the next
        let dst = (li * 2 * ROWS + c) * row;
        self.backend.copy(&buf, dst, &res.buf, res.offset, n * row, self.stream)?;
        if let Some(b) = blk {
            self.backend.copy(&buf, dst + ROWS * row, &b.buf, b.offset, n * row, self.stream)?;
        }
        Ok(())
    }

    /// Before a pass: no rows of it in the tap yet (no-op when off).
    pub(super) fn det_tap_begin(&self) {
        if let Some(tp) = tap() {
            tp.rows.lock().unwrap().iter_mut().for_each(|r| *r = 0);
        }
    }

    /// After a single-sequence pass that started at `pos` with `t` rows: every rank's tap line (no-op when off).
    pub(super) fn det_tap_flush(&self, pos: usize, t: usize) -> Result<()> {
        // `t`: the pass's rows (the line's `t=`); the tap holds the first min(t, ROWS)
        let Some(tp) = tap() else { return Ok(()) };
        let mut lines = String::new();
        for r in 0..self.tp.world {
            let Some(m) = self.rank_model(r) else { continue };
            let Some(buf) = tp.bufs.lock().unwrap().get(r).cloned().flatten() else { continue };
            let mut host = vec![0u8; buf.bytes()];
            m.backend.download(&buf, 0, &mut host, m.stream)?;
            let (layer, row) = (host.len() / m.layers.len(), host.len() / m.layers.len() / (2 * ROWS));
            let n = std::mem::take(&mut tp.rows.lock().unwrap()[r]);
            // a row's hash covers its residual and its block output bytes
            let hash = |l: &[u8], i: usize| fnv(&[&l[i * row..(i + 1) * row], &l[(ROWS + i) * row..(ROWS + i + 1) * row]].concat());
            let hs: Vec<String> = host
                .chunks(layer)
                .map(|l| (0..n).map(|i| format!("{:016x}", hash(l, i))).collect::<Vec<_>>().join(","))
                .collect();
            lines += &format!("pos={pos} t={t} rank={r} {}\n", hs.join(" "));
        }
        tp.file.lock().unwrap().write_all(lines.as_bytes()).map_err(|e| Error::Other(format!("INTARSIA_DET_TAP: {e}")))
    }
}
