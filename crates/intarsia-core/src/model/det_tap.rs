//! Run-to-run determinism tap (lane det, debugging, off by default): `INTARSIA_DET_TAP=<file>` copies the last row of every
//! layer's output (residual stream plus the pending block output) of a single-sequence pass into a per-rank device buffer,
//! and after the pass appends one line per rank to `<file>`: `pos=<first position> t=<rows> rank=<r> <hash per layer>`
//! (FNV-1a 64 of the row's bytes). Both ranks of a tensor-parallel group hold the same residual stream after every
//! all-reduce, so their lines must be equal; across runs the first layer whose hash differs is where the runs part.
//! The copies are ordinary launches on the rank's stream (captured into a decode graph like the rest of the step); the
//! download after the pass waits for each rank's stream, which a peer rank's step otherwise does not.

use super::*;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

struct Tap {
    file: Mutex<std::fs::File>,
    /// per rank: `[n_layers, hidden]` in the activation dtype
    bufs: Mutex<Vec<Option<Buffer>>>,
}

fn tap() -> Option<&'static Tap> {
    static T: OnceLock<Option<Tap>> = OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var("INTARSIA_DET_TAP").ok().filter(|p| !p.is_empty())?;
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).expect("INTARSIA_DET_TAP: cannot open the file");
        eprintln!("det tap: layer output hashes -> {path}");
        Some(Tap { file: Mutex::new(file), bufs: Mutex::new(vec![None; 8]) })
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
        let b = self.backend.alloc(self.layers.len() * self.dims.hidden * self.act.size_bytes(), MemKind::Device)?;
        self.backend.zero(&b, self.stream)?;
        g[self.tp.rank] = Some(b.clone());
        Ok(b)
    }

    /// After layer `li` of a single-sequence pass of `t` rows: its output's last row into the tap (no-op when off).
    pub(super) fn det_tap_layer(&self, li: usize, t: usize, res: &TensorRef, blk: Option<&TensorRef>) -> Result<()> {
        let Some(tp) = tap() else { return Ok(()) };
        let h = self.dims.hidden;
        let buf = self.det_tap_buf(tp)?;
        let out = self.view(&buf, self.act, &[self.layers.len(), h]).rows(li, 1);
        // no block output pending: the row twice (a function of the residual alone, which is all the hash needs)
        let b = blk.unwrap_or(res).rows(t - 1, 1);
        self.exec(Op::Add, vec![res.rows(t - 1, 1), b], vec![out], OpAttrs::default(), false)
    }

    /// After a single-sequence pass that started at `pos` with `t` rows: every rank's tap line (no-op when off).
    pub(super) fn det_tap_flush(&self, pos: usize, t: usize) -> Result<()> {
        let Some(tp) = tap() else { return Ok(()) };
        let mut lines = String::new();
        for r in 0..self.tp.world {
            let Some(m) = self.rank_model(r) else { continue };
            let Some(buf) = tp.bufs.lock().unwrap().get(r).cloned().flatten() else { continue };
            let mut host = vec![0u8; buf.bytes()];
            m.backend.download(&buf, 0, &mut host, m.stream)?;
            let row = host.len() / m.layers.len();
            let hs: Vec<String> = host.chunks(row).map(|c| format!("{:016x}", fnv(c))).collect();
            lines += &format!("pos={pos} t={t} rank={r} {}\n", hs.join(" "));
        }
        tp.file.lock().unwrap().write_all(lines.as_bytes()).map_err(|e| Error::Other(format!("INTARSIA_DET_TAP: {e}")))
    }
}
