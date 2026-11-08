//! `INTARSIA_DSPINE_DUMP=<dir>`: what the DSpine drafter ingests and drafts, for the offline tap-replay test
//! (`tools/dspine/replay_taps.py` feeds the same target taps to Tencent's MLX reference and compares drafts and acceptance).
//!
//! One sub-directory `req-NNNN` per request (a request starts at the first ingest of position 0), files:
//!
//! ```text
//!   meta.json    hidden, taps (target layer ids), dtype of the rows, block, topk, layers, vocab, code, rank, dump_layers, head (bf16 | fp8: the
//!                head rows the draft's logits and tail read)
//!   taps.bin     every ingest, appended: `t` rows of `[n_taps, hidden]` in `dtype` (the tensor the draft's fc reads, tap-major
//!                inside a row: tap 0's `hidden` values first)
//!   passes.bin   per draft pass, appended: candidate ids i32 `[t-1, topk]` (ascending per row) then the live head's logits at
//!                them f32 `[t-1, topk]`
//!   layers.bin   (INTARSIA_DSPINE_DUMP_LAYERS=1) per draft pass, appended: for each draft layer `out`, `normed`, `raw` bf16 `[t, hidden]`
//!                (`out` = the residual stream after the layer's code write, `normed` = the next layer's (the final norm's) input,
//!                `raw` = the layer output before the write), then the last layer's rms f32 `[t]` and packed bf16 `[t, 3 * rank]`
//!   index.jsonl  one line per event, in order:
//!                {"ev":"ingest","pos0":P,"t":T,"off":BYTES}                 rows land at positions pos0 .. pos0 + T (an ingest at
//!                                                                          pos0 cuts whatever the cache held beyond it: a rollback)
//!                {"ev":"pass","p0":P,"anchor":A,"t":T,"drafts":[..],"tok_from":F,"new_tokens":[..],"cand_off":BYTES,"layers_off":BYTES|null}
//!                                                                          the block at p0 over the context rows < p0; `new_tokens`
//!                                                                          are the committed ids at positions F .. p0 (the prompt
//!                                                                          first), the anchor sits at p0
//! ```
//!
//! Single sequence, rank 0 (a tensor-parallel DSpine draft is not supported); the dump syncs the stream at every ingest, so it
//! slows the run but changes no token.

use super::*;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn dir() -> Option<&'static PathBuf> {
    static D: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    D.get_or_init(|| std::env::var("INTARSIA_DSPINE_DUMP").ok().filter(|s| !s.is_empty()).map(PathBuf::from)).as_ref()
}

/// Whether a dump is asked for.
pub(super) fn on() -> bool {
    dir().is_some()
}

/// `INTARSIA_DSPINE_DUMP_LAYERS=1`: also dump every layer's rows of each pass.
pub(super) fn layers_on() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    on() && *D.get_or_init(|| std::env::var("INTARSIA_DSPINE_DUMP_LAYERS").is_ok_and(|v| v == "1"))
}

struct Req {
    dir: PathBuf,
    taps: File,
    passes: File,
    layers: Option<File>,
    index: File,
    taps_off: u64,
    passes_off: u64,
    layers_off: u64,
    toks: usize,
}

struct State {
    n_req: usize,
    req: Option<Req>,
}

static STATE: Mutex<State> = Mutex::new(State { n_req: 0, req: None });

fn io<T>(r: std::io::Result<T>, what: &str) -> Result<T> {
    r.map_err(|e| Error::backend(format!("INTARSIA_DSPINE_DUMP: {what}: {e}")))
}

fn ids_json<T: std::fmt::Display>(v: &[T]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

fn dtype_name(d: DType) -> Result<&'static str> {
    match d {
        DType::Bf16 => Ok("bf16"),
        DType::F16 => Ok("f16"),
        DType::F32 => Ok("f32"),
        o => Err(Error::backend(format!("INTARSIA_DSPINE_DUMP: activation dtype {o:?}"))),
    }
}

impl Model {
    fn dump_open(&self, sp: &Spine, taps: &[usize], block: usize, st: &mut State) -> Result<()> {
        let root = dir().expect("dump on");
        let d = root.join(format!("req-{:04}", st.n_req));
        st.n_req += 1;
        io(std::fs::create_dir_all(&d), "create dir")?;
        let h = self.dims.hidden;
        let nl = self.dflash.as_ref().map_or(0, |x| x.n_layers());
        let (r, c) = (sp.rank, sp.code);
        // the head the draft's logits and tail read (the recipe's fp8 default for the 27B, or the checkpoint's bf16): the replay must read the same
        let head_fmt = match &self.dspine_tabs(sp)?.head {
            HeadRows::Bf16(_) => "bf16",
            HeadRows::Fp8 { .. } => "fp8",
        };
        let meta = format!(
            "{{\"hidden\":{h},\"taps\":{},\"dtype\":\"{}\",\"block\":{block},\"topk\":{},\"layers\":{nl},\"vocab\":{},\"code\":{c},\"rank\":{r},\"dump_layers\":{},\"head\":\"{head_fmt}\"}}\n",
            ids_json(taps),
            dtype_name(self.act)?,
            sp.topk,
            self.dims.vocab,
            layers_on()
        );
        io(std::fs::write(d.join("meta.json"), meta), "meta.json")?;
        let mk = |n: &str| io(File::create(d.join(n)), n);
        st.req = Some(Req {
            taps: mk("taps.bin")?,
            passes: mk("passes.bin")?,
            layers: if layers_on() { Some(mk("layers.bin")?) } else { None },
            index: mk("index.jsonl")?,
            dir: d,
            taps_off: 0,
            passes_off: 0,
            layers_off: 0,
            toks: 0,
        });
        Ok(())
    }

    /// The target's tap rows of a pass about to enter the draft's context at positions `pos0 .. pos0 + t` (`src`: the `[rows, H]`
    /// feature buffer of every tap). Nothing unless `INTARSIA_DSPINE_DUMP` is set and the draft is a DSpine one.
    pub(in crate::model) fn dspine_dump_taps(&self, head: &crate::model::dflash::DflashHead, pos0: usize, src: &[Buffer], t: usize) -> Result<()> {
        if !on() || t == 0 || self.recorder.lock().unwrap().is_some() {
            return Ok(());
        }
        let Some(sp) = head.v2.as_ref().and_then(|d| d.spine.as_deref()) else { return Ok(()) };
        let mut g = STATE.lock().unwrap();
        if pos0 == 0 {
            self.dump_open(sp, &head.taps, head.block, &mut g)?;
        }
        let Some(rq) = g.req.as_mut() else { return Ok(()) };
        let (h, esz, nt) = (self.dims.hidden, self.act.size_bytes(), src.len());
        let mut cols = Vec::with_capacity(nt);
        for b in src {
            let mut v = vec![0u8; t * h * esz];
            self.backend.download(b, 0, &mut v, self.stream)?;
            cols.push(v);
        }
        let mut out = Vec::with_capacity(t * nt * h * esz);
        for r in 0..t {
            for c in &cols {
                out.extend_from_slice(&c[r * h * esz..(r + 1) * h * esz]);
            }
        }
        io(rq.taps.write_all(&out), "taps.bin")?;
        io(writeln!(rq.index, "{{\"ev\":\"ingest\",\"pos0\":{pos0},\"t\":{t},\"off\":{}}}", rq.taps_off), "index")?;
        rq.taps_off += out.len() as u64;
        Ok(())
    }

    /// One finished draft pass: the block at `p0` (`t` rows, anchor `next`), the candidates and live-head logits per row 1 .. t - 1,
    /// the drafts of the tail, the committed ids not yet logged, and with `INTARSIA_DSPINE_DUMP_LAYERS=1` every layer's rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dspine_dump_pass(&self, sp: &Spine, seq: &[u32], next: u32, p0: usize, t: usize, cand: &[Vec<usize>], logits: &[f32], drafts: &[u32]) -> Result<()> {
        if !on() {
            return Ok(());
        }
        let mut g = STATE.lock().unwrap();
        let Some(rq) = g.req.as_mut() else { return Ok(()) };
        let (v, k) = (self.dims.vocab, sp.topk);
        let mut pb = Vec::with_capacity((t - 1) * k * 8);
        for row in cand {
            pb.extend(row.iter().flat_map(|&c| (c as i32).to_le_bytes()));
        }
        for (i, row) in cand.iter().enumerate() {
            pb.extend(row.iter().flat_map(|&c| logits[i * v + c].to_le_bytes()));
        }
        io(rq.passes.write_all(&pb), "passes.bin")?;
        let cand_off = rq.passes_off;
        rq.passes_off += pb.len() as u64;
        let layers_off = if let Some(f) = rq.layers.as_mut() {
            let log = sp.layers_log.lock().unwrap();
            let mut lb: Vec<u8> = Vec::new();
            let bf16s = |x: &[f32], out: &mut Vec<u8>| out.extend(x.iter().flat_map(|&e| f2bf(e).to_le_bytes()));
            for [out, normed, raw] in log.iter() {
                bf16s(out, &mut lb);
                bf16s(normed, &mut lb);
                bf16s(raw, &mut lb);
            }
            if let Some(l) = sp.last.lock().unwrap().as_ref() {
                lb.extend(l.rms.iter().flat_map(|x| x.to_le_bytes()));
                bf16s(&l.packed, &mut lb);
            }
            io(f.write_all(&lb), "layers.bin")?;
            let o = rq.layers_off;
            rq.layers_off += lb.len() as u64;
            Some(o)
        } else {
            None
        };
        let from = rq.toks.min(p0);
        let new_tokens = &seq[from.min(seq.len())..p0.min(seq.len())];
        rq.toks = p0;
        let lo = layers_off.map_or("null".to_string(), |o| o.to_string());
        io(
            writeln!(
                rq.index,
                "{{\"ev\":\"pass\",\"p0\":{p0},\"anchor\":{next},\"t\":{t},\"drafts\":{},\"tok_from\":{from},\"new_tokens\":{},\"cand_off\":{cand_off},\"layers_off\":{lo}}}",
                ids_json(drafts),
                ids_json(new_tokens)
            ),
            "index",
        )?;
        let _ = &rq.dir;
        Ok(())
    }
}
