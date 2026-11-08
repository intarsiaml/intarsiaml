//! DSpine block drafter (`--spec dflash` with a `tencent/Qwen3.8-27B-DSpine-b16` checkpoint), docs/plans/2026-11-07-dspine-port.md.
//! Checkpoint and semantics: Tencent/DSpine (Apache-2.0), read, not copied.
//!
//! What differs from the DFlash2 draft of [`super::dflash2`] (whose backbone path, taps, context ingest, state and rollback it
//! shares; a DSpine draft is a `Draft2` whose layers carry no conv and whose `spine` field is set):
//!
//! ```text
//!   layers    plain Qwen3 (x += attn(norm(x)); x += mlp(norm(x))), geometry of its own, causal inside the block: block row r sees
//!             every context key and block rows 0 ..= r (one `AttnDecode` per row at position p0 + r), all layers sliding window W
//!   bridge    after every layer l a "code write": raw = layer output;
//!               nrm = raw / (rms + eps_b)                       (BridgeNorm)
//!               code[t] = sqrt(C) * (nrm @ read_l)[t] / |.|      (BridgeCode; row 0's code is not used)
//!               packed = code @ [G_q; G_p; A]^T                  [q | s | v] per row
//!               msg[t] = sigmoid(q[t] + s[t-1] + bias) * v[t-1]  (BridgeGate; row 1's sender is the anchor token's TSV row)
//!               out = raw + rms * (msg @ write_l^T)              (BridgeWrite, row 0 untouched), then the next layer's norm
//!   tail      per block row 1 .. t-1: the K = 16 best candidates of the pass-1 logits (host, bf16 keys, lower id wins ties), then a
//!             greedy walk from the anchor: with sender s the last write is redone exactly through two vocabulary tables
//!               TG[v] = (head[v] * norm.w) @ write_4        TSV[v] = bf16(sqrt(C) * codebook[v]) @ [G_p; A]^T
//!             score_b = (raw * norm.w) . head[c_b]  +  rms * m . TG[c_b],  m = sigmoid(q + TSV[s].s + bias) * TSV[s].v
//! ```
//!
//! Drafts never change the output (every emitted token is a verify row's argmax); the numerics only move the acceptance rate.
//! The block runs `t = min(block, k + 1)` rows: it is causal, so the first rows of a shorter block equal those of the full one.

use super::dflash2::{bf, row_topk, Draft2};
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Rows of the target's head the tail's `base` scores read, dequantised to what the live head holds.
pub(super) enum HeadRows {
    Bf16(Vec<u16>),
    /// e4m3 bytes and `lut[b] = bf16(e4m3(b) * scale)` (as f32), the per-tensor scale folded in.
    Fp8 { bytes: Vec<u8>, lut: Box<[f32; 256]> },
}

impl HeadRows {
    /// Row `r` (`h` values) as f32.
    fn row_into(&self, r: usize, h: usize, out: &mut [f32]) {
        match self {
            HeadRows::Bf16(v) => v[r * h..(r + 1) * h].iter().zip(out.iter_mut()).for_each(|(&b, o)| *o = bf(b)),
            HeadRows::Fp8 { bytes, lut } => bytes[r * h..(r + 1) * h].iter().zip(out.iter_mut()).for_each(|(&b, o)| *o = lut[b as usize]),
        }
    }
    /// `sum_h x[h] * head[r][h]`, f32 accumulation in 8 lanes (then the `h % 8` trailing columns, in order).
    fn dot(&self, r: usize, h: usize, x: &[f32]) -> f32 {
        let mut acc = [0f32; 8];
        let main = h / 8 * 8;
        let mut tail = 0f32;
        match self {
            HeadRows::Bf16(v) => {
                let row = &v[r * h..(r + 1) * h];
                for (xs, hs) in x.chunks_exact(8).zip(row.chunks_exact(8)) {
                    for j in 0..8 {
                        acc[j] += xs[j] * bf(hs[j]);
                    }
                }
                for j in main..h {
                    tail += x[j] * bf(row[j]);
                }
            }
            HeadRows::Fp8 { bytes, lut } => {
                let row = &bytes[r * h..(r + 1) * h];
                for (xs, hs) in x.chunks_exact(8).zip(row.chunks_exact(8)) {
                    for j in 0..8 {
                        acc[j] += xs[j] * lut[hs[j] as usize];
                    }
                }
                for j in main..h {
                    tail += x[j] * lut[row[j] as usize];
                }
            }
        }
        ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
    }
}

/// The tail's tables and head rows (built once, after the plan admitted the build's matmuls).
pub(super) struct Tables {
    /// `[V, R]` bf16 bits.
    pub(super) tg: Vec<u16>,
    /// `[V, 2R]` bf16 bits: `[G_p e_v | A e_v]`.
    pub(super) tsv: Vec<u16>,
    pub(super) head: HeadRows,
}

pub(super) struct SpineBufs {
    /// `[block, H]` the layer output `raw` (the last layer's stays: the stash)
    raw: Buffer,
    /// `[block]` f32
    rms: Buffer,
    /// `[block, C]`
    p: Buffer,
    code: Buffer,
    /// `[block, 3R]`
    packed: Buffer,
    /// `[block, R]`
    msg: Buffer,
    /// `[block, H]`
    dlt: Buffer,
    /// `[2R]` the anchor token's TSV row
    s0: Buffer,
    /// table build staging: `[chunk, H]` and `[chunk, 2R]`
    stage_in: Buffer,
    stage_out: Buffer,
}

pub(super) struct Spine {
    pub(super) code: usize,
    pub(super) rank: usize,
    pub(super) eps_b: f32,
    pub(super) topk: usize,
    /// `[C, H]` per layer: `early_read_weights[l]` for l < L-1, `read_weight` for the last
    read: Vec<QTensor>,
    /// `[H, R]` per layer
    write: Vec<QTensor>,
    /// `[3R, C]` = `[gate; predecessor_gate; prev]`
    packed_w: QTensor,
    /// `[2R, C]` = `[predecessor_gate; prev]`
    pv_w: QTensor,
    /// `[R, H]`: the last write transposed (the TG build's weight)
    write4_t: QTensor,
    bias: TensorRef,
    /// `gate.bias` and the final `norm.weight` as f32 on the host (the tail)
    bias_f: Vec<f32>,
    norm_f: Vec<f32>,
    bufs: SpineBufs,
    /// Rows of one table-build chunk (the build's matmuls have M = chunk)
    chunk: usize,
    src: Arc<dyn WeightSource>,
    tables: Mutex<Option<Arc<Tables>>>,
    /// Tests: download every layer's `[res, normed]` rows after the layer (see [`Model::dspine_trace`])
    trace: AtomicBool,
    layers_log: Mutex<Vec<[Vec<f32>; 3]>>,
    /// Tests: the last pass's stash and candidates (see [`Model::dspine_last_pass`])
    last: Mutex<Option<LastPass>>,
}

/// What the tail saw in the last draft pass (tests).
#[derive(Clone, Debug, Default)]
pub struct LastPass {
    /// `[t, H]` the last layer's pre-write output (the stash's `raw`), f32 of the bf16 values
    pub raw: Vec<f32>,
    /// `[t]`
    pub rms: Vec<f32>,
    /// `[t, 3R]`
    pub packed: Vec<f32>,
    /// `[t - 1][K]` candidate ids, ascending
    pub cand: Vec<Vec<usize>>,
    /// `[t - 1, V]`, only with `dspine_keep_logits`
    pub drafts: Vec<u32>,
}

fn f2bf(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn le_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `f(chunk_start, chunk_rows, out_chunk)` over `n` rows of `width` elements, in parallel on the available threads.
fn par_fill<T: Send>(out: &mut [T], width: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = out.len() / width.max(1);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32).max(1);
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (ci, ch) in out.chunks_mut(per * width).enumerate() {
            let f = &f;
            sc.spawn(move || f(ci * per, ch));
        }
    });
}

/// A bf16 `[n, k]` dense tensor from raw bytes.
fn dense(b: &dyn Backend, bytes: &dyn Fn(u64), raw: &[u8], n: usize, k: usize) -> Result<QTensor> {
    debug_assert_eq!(raw.len(), n * k * 2);
    let buf = crate::backend::upload_new(b, raw)?;
    bytes(raw.len() as u64);
    Ok(QTensor { format: crate::quant::BF16, n, k, parts: vec![TensorRef::new(buf, DType::Bf16, &[n, k])], act_fp8: false, act_quant: false })
}

/// The bridge's weights, scratch and (once, lazily) tables of a DSpine draft; the layers themselves are the `Draft2`'s.
pub(super) fn load_spine(
    dl: &super::dflash::DflashLoad,
    sc: &super::dflash::SpineConfig,
    b: &dyn Backend,
    dims: &ModelDims,
    add_bytes: &dyn Fn(u64),
    rows: usize,
    act: DType,
) -> Result<Spine> {
    let (c, src) = (&dl.config, &*dl.src);
    let (h, l, cc, r, v) = (dims.hidden, c.n_layers, sc.code, sc.rank, dims.vocab);
    let get = |name: &str, dtype_shape: &[usize]| -> Result<&[u8]> {
        let full = resolve(src, &format!("selected_prefix_bridge.{name}"))?;
        let info = src.info(&full).ok_or_else(|| Error::model(format!("draft tensor '{full}' missing")))?;
        if info.dtype != DType::Bf16 || info.shape != dtype_shape {
            return Err(Error::model(format!("draft '{full}' is {} {:?}, expected bf16 {dtype_shape:?}", info.dtype, info.shape)));
        }
        src.bytes(&full)
    };
    // read weights: layers 0..L-1 take early_read_weights[l], the last layer read_weight
    let early = get("early_read_weights", &[l - 1, cc, h])?;
    let last_read = get("read_weight", &[cc, h])?;
    let rb = cc * h * 2;
    let mut read = Vec::with_capacity(l);
    for i in 0..l - 1 {
        read.push(dense(b, add_bytes, &early[i * rb..(i + 1) * rb], cc, h)?);
    }
    read.push(dense(b, add_bytes, last_read, cc, h)?);
    // write weights [H, R] per layer
    let writes = get("feature_write_weights", &[l, h, r])?;
    let wb = h * r * 2;
    let mut write = Vec::with_capacity(l);
    for i in 0..l {
        write.push(dense(b, add_bytes, &writes[i * wb..(i + 1) * wb], h, r)?);
    }
    // the last write transposed, [R, H]: the TG build's weight (TG = bf16(head * norm.w) @ write_4)
    let w4 = le_u16(&writes[(l - 1) * wb..l * wb]);
    let mut w4t = vec![0u16; r * h];
    for i in 0..h {
        for j in 0..r {
            w4t[j * h + i] = w4[i * r + j];
        }
    }
    let write4_t = dense(b, add_bytes, &u16_bytes(&w4t), r, h)?;
    let (gate_w, pred_w, prev_w) = (get("gate.weight", &[r, cc])?, get("predecessor_gate.weight", &[r, cc])?, get("prev.weight", &[r, cc])?);
    let packed_w = dense(b, add_bytes, &[gate_w, pred_w, prev_w].concat(), 3 * r, cc)?;
    let pv_w = dense(b, add_bytes, &[pred_w, prev_w].concat(), 2 * r, cc)?;
    let bias_raw = get("gate.bias", &[r])?;
    let bias_buf = crate::backend::upload_new(b, bias_raw)?;
    add_bytes(bias_raw.len() as u64);
    let bias = TensorRef::new(bias_buf, DType::Bf16, &[r]);
    let bias_f: Vec<f32> = le_u16(bias_raw).into_iter().map(bf).collect();
    let nname = resolve(src, "norm")?;
    let ninfo = src.info(&nname).ok_or_else(|| Error::model("draft tensor 'norm' missing"))?;
    if ninfo.dtype != DType::Bf16 || ninfo.shape != [h] {
        return Err(Error::model(format!("draft '{nname}' is {} {:?}, expected bf16 [{h}]", ninfo.dtype, ninfo.shape)));
    }
    let norm_f: Vec<f32> = le_u16(src.bytes(&nname)?).into_iter().map(bf).collect();
    let _ = v;
    // scratch (always bf16: the bridge rounds at the reference's points whatever the target's activations are)
    let t = c.block;
    let chunk = rows.clamp(1, 2048);
    debug_assert_eq!(act, DType::Bf16);
    let alloc = |bytes: usize| -> Result<Buffer> {
        add_bytes(bytes as u64);
        b.alloc(bytes.max(16), MemKind::Device)
    };
    let bufs = SpineBufs {
        raw: alloc(t * h * 2)?,
        rms: alloc(t * 4)?,
        p: alloc(t * cc * 2)?,
        code: alloc(t * cc * 2)?,
        packed: alloc(t * 3 * r * 2)?,
        msg: alloc(t * r * 2)?,
        dlt: alloc(t * h * 2)?,
        s0: alloc(2 * r * 2)?,
        stage_in: alloc(chunk * h * 2)?,
        stage_out: alloc(chunk * 2 * r * 2)?,
    };
    // The tail's host tables (TG [V, R] + TSV [V, 2R] bf16, and the head rows, at least fp8's one byte a value) are built later by
    // `dspine_tabs`. On a unified-memory backend they come out of the same pool as the weights, so count them against the budget.
    if b.info().unified {
        add_bytes((v * r * 2 + v * 2 * r * 2 + v * h) as u64);
    }
    Ok(Spine {
        code: cc,
        rank: r,
        eps_b: sc.eps_b,
        topk: sc.head_topk,
        read,
        write,
        packed_w,
        pv_w,
        write4_t,
        bias,
        bias_f,
        norm_f,
        bufs,
        chunk,
        src: dl.src.clone(),
        tables: Mutex::new(None),
        trace: AtomicBool::new(false),
        layers_log: Mutex::new(Vec::new()),
        last: Mutex::new(None),
    })
}

/// The `k` best candidates of one logits row: keys are the logits rounded to bf16 (the reference's logits are bf16, so ties are
/// common), the lower id wins a tie, and the ids come back ascending (lane order of the tail).
pub(super) fn candidates(row: &[f32], k: usize) -> Vec<usize> {
    let keys: Vec<f32> = row.iter().map(|&x| f32::from_bits((f2bf(x) as u32) << 16)).collect();
    let mut top = Vec::with_capacity(k + 1);
    row_topk(&keys, k, 0, &mut top);
    let mut ids: Vec<usize> = top.into_iter().map(|(_, i)| i).collect();
    ids.sort_unstable();
    ids
}

/// One row of the tail's walk step: `score_b = base_b + rms * (m . TG[c_b])` for every candidate, `m` from the sender's TSV row.
/// Returns the winning lane (the first best).
#[allow(clippy::too_many_arguments)]
fn best_lane(
    tabs: &Tables,
    h: usize,
    r: usize,
    x: &[f32],
    rms: f32,
    q: &[f32],
    bias: &[f32],
    sender: usize,
    cand: &[usize],
    m: &mut [f32],
) -> usize {
    let ts = &tabs.tsv[sender * 2 * r..(sender + 1) * 2 * r];
    for k in 0..r {
        let z = (q[k] + bf(ts[k])) + bias[k];
        m[k] = (1.0 / (1.0 + (-z).exp())) * bf(ts[r + k]);
    }
    let (mut best, mut best_s) = (0usize, f32::NEG_INFINITY);
    for (b, &c) in cand.iter().enumerate() {
        let base = tabs.head.dot(c, h, x);
        let tg = &tabs.tg[c * r..(c + 1) * r];
        let mut s = 0f32;
        for k in 0..r {
            s += m[k] * bf(tg[k]);
        }
        let score = base + rms * s;
        if score > best_s {
            best = b;
            best_s = score;
        }
    }
    best
}

/// The greedy tail: for block rows `1 .. t` (stash row `i + 1` is `raw[i]`, `rms[i]`, `q[i]`) the chain of drafts walking from
/// `anchor`, each row's candidate chosen under the previous draft as sender. `raw` is `[n, H]` bf16 bits; `q` `[n, R]` f32.
#[allow(clippy::too_many_arguments)]
pub(super) fn walk(tabs: &Tables, nw: &[f32], bias: &[f32], h: usize, r: usize, raw: &[u16], rms: &[f32], q: &[f32], cand: &[Vec<usize>], anchor: usize) -> Vec<u32> {
    let n = cand.len();
    let mut m = vec![0f32; r];
    let mut x = vec![0f32; h];
    let (mut sender, mut out) = (anchor, Vec::with_capacity(n));
    for i in 0..n {
        for j in 0..h {
            x[j] = bf(raw[i * h + j]) * nw[j];
        }
        let lane = best_lane(tabs, h, r, &x, rms[i], &q[i * r..(i + 1) * r], bias, sender, &cand[i], &mut m);
        sender = cand[i][lane];
        out.push(sender as u32);
    }
    out
}

/// Tests: the full `[n, K, K]` score table (sender lane `a`, receiver lane `b`; row 0's senders all the anchor) and the chain
/// over it: the reference's formulation of the tail, which [`walk`] must equal.
#[allow(clippy::too_many_arguments)]
pub(super) fn table_chain(tabs: &Tables, nw: &[f32], bias: &[f32], h: usize, r: usize, raw: &[u16], rms: &[f32], q: &[f32], cand: &[Vec<usize>], anchor: usize) -> (Vec<Vec<Vec<f32>>>, Vec<u32>) {
    let n = cand.len();
    let mut scores = Vec::with_capacity(n);
    let mut m = vec![0f32; r];
    for i in 0..n {
        let x: Vec<f32> = (0..h).map(|j| bf(raw[i * h + j]) * nw[j]).collect();
        let senders: Vec<usize> = if i == 0 { vec![anchor; cand[0].len()] } else { cand[i - 1].clone() };
        let mut rows = Vec::new();
        for &s in &senders {
            let ts = &tabs.tsv[s * 2 * r..(s + 1) * 2 * r];
            for k in 0..r {
                let z = (q[i * r + k] + bf(ts[k])) + bias[k];
                m[k] = (1.0 / (1.0 + (-z).exp())) * bf(ts[r + k]);
            }
            let row: Vec<f32> = cand[i]
                .iter()
                .map(|&c| {
                    let base = tabs.head.dot(c, h, &x);
                    let tg = &tabs.tg[c * r..(c + 1) * r];
                    let mut sv = 0f32;
                    for k in 0..r {
                        sv += m[k] * bf(tg[k]);
                    }
                    base + rms[i] * sv
                })
                .collect();
            rows.push(row);
        }
        scores.push(rows);
    }
    let argmax = |row: &[f32]| row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (j, &s)| if s > a.1 { (j, s) } else { a }).0;
    let mut out = Vec::with_capacity(n);
    let mut j = 0usize;
    for i in 0..n {
        j = argmax(&scores[i][if i == 0 { 0 } else { j }]);
        out.push(cand[i][j] as u32);
    }
    (scores, out)
}

impl Model {
    /// Tests: download `[res, normed, raw]` (post-write residual stream, the next layer's normed input and the layer's output
    /// before its write) after every layer of the following draft passes; read with [`Model::dspine_layers`].
    #[doc(hidden)]
    pub fn dspine_trace(&self, on: bool) -> Result<()> {
        let sp = self.dspine()?;
        sp.trace.store(on, Ordering::Relaxed);
        sp.layers_log.lock().unwrap().clear();
        Ok(())
    }

    /// Tests: the traced layers of the last draft pass (`[res, normed, raw]`, each `[t, H]` f32), in layer order.
    #[doc(hidden)]
    pub fn dspine_layers(&self) -> Result<Vec<[Vec<f32>; 3]>> {
        Ok(self.dspine()?.layers_log.lock().unwrap().clone())
    }

    /// Tests: the stash, candidates and drafts of the last draft pass.
    #[doc(hidden)]
    pub fn dspine_last_pass(&self) -> Result<LastPass> {
        self.dspine()?.last.lock().unwrap().clone().ok_or_else(|| Error::invalid("no DSpine draft pass yet"))
    }

    /// Tests: the tables (built if need be) as host values, `(TG [V, R], TSV [V, 2R])`, bf16 bits.
    #[doc(hidden)]
    pub fn dspine_tables(&self) -> Result<(Vec<u16>, Vec<u16>)> {
        let sp = self.dspine()?;
        let t = self.dspine_tabs(sp)?;
        Ok((t.tg.clone(), t.tsv.clone()))
    }

    /// Tests: the tail on host inputs (`raw` bf16 bits `[n, H]`, `rms` `[n]`, `q` `[n, R]`, candidate ids): `(walk, table + chain)`.
    #[doc(hidden)]
    pub fn dspine_tail_host(&self, raw: &[u16], rms: &[f32], q: &[f32], cand: &[Vec<usize>], anchor: u32) -> Result<(Vec<u32>, Vec<u32>)> {
        let sp = self.dspine()?;
        let t = self.dspine_tabs(sp)?;
        let (h, r) = (self.dims.hidden, sp.rank);
        let w = walk(&t, &sp.norm_f, &sp.bias_f, h, r, raw, rms, q, cand, anchor as usize);
        let (_, c) = table_chain(&t, &sp.norm_f, &sp.bias_f, h, r, raw, rms, q, cand, anchor as usize);
        Ok((w, c))
    }

    fn dspine(&self) -> Result<&Spine> {
        self.dflash
            .as_ref()
            .and_then(|h| h.v2.as_ref())
            .and_then(|d| d.spine.as_deref())
            .ok_or_else(|| Error::invalid("no DSpine draft is loaded"))
    }

    /// Whether a DSpine draft is loaded.
    pub fn has_dspine(&self) -> bool {
        self.dspine().is_ok()
    }

    /// Build the tables now (the loader does it right after the plan, so the first request does not pay for it and an unsupported
    /// head is refused at load; a draft pass that finds them missing still builds them). Needs the draft's admitted matmuls (the plan's dry
    /// run records them) and the target's head.
    pub fn dspine_prepare(&self) -> Result<()> {
        let sp = self.dspine()?;
        self.dspine_tabs(sp).map(|_| ())
    }

    fn dl_bytes(&self, t: &TensorRef) -> Result<Vec<u8>> {
        let mut b = vec![0u8; t.nbytes()];
        self.backend.download(&t.buf, t.offset, &mut b, self.stream)?;
        Ok(b)
    }

    /// The tables, built on first use: the live head's values (downloaded, dequantised as the head's kernels read them), TG and
    /// TSV through chunked `Matmul`s on the device.
    fn dspine_tabs(&self, sp: &Spine) -> Result<Arc<Tables>> {
        let mut slot = sp.tables.lock().unwrap();
        if let Some(t) = &*slot {
            return Ok(t.clone());
        }
        if self.recorder.lock().unwrap().is_some() {
            return Err(Error::invalid("DSpine tables requested during a plan dry run"));
        }
        let t0 = Instant::now();
        let (h, r, v) = (self.dims.hidden, sp.rank, self.dims.vocab);
        let head = self.head()?;
        let rows = match head.format {
            f if f == crate::quant::BF16 => {
                let t = &head.parts[0];
                if t.shape != [v, h] {
                    return Err(Error::Unsupported(format!("the DSpine tail with a head of shape {:?}", t.shape)));
                }
                match t.dtype {
                    DType::Bf16 => {
                        // straight into the u16 rows: no second copy of a 2.5 GB head
                        let mut rows16 = vec![0u16; v * h];
                        // SAFETY: a u16 slice's storage viewed as bytes (no padding, align 1)
                        let bytes = unsafe { std::slice::from_raw_parts_mut(rows16.as_mut_ptr() as *mut u8, rows16.len() * 2) };
                        self.backend.download(&t.buf, t.offset, bytes, self.stream)?;
                        rows16.iter_mut().for_each(|x| *x = u16::from_le(*x));
                        HeadRows::Bf16(rows16)
                    }
                    // a float head held in another dtype (the cpu backend's f32 test models): its bf16 rounding, which is
                    // what the reference's bf16 head rows are
                    DType::F32 | DType::F16 => {
                        let raw = self.dl_bytes(t)?;
                        let mut f = vec![0f32; v * h];
                        crate::dtype::decode_to_f32(t.dtype, &raw, &mut f);
                        HeadRows::Bf16(f.into_iter().map(f2bf).collect())
                    }
                    other => return Err(Error::Unsupported(format!("the DSpine tail with a {other} head"))),
                }
            }
            f if f == crate::quant::FP8_E4M3 => {
                let sc = self.dl_bytes(&head.parts[1])?;
                let ws = f32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
                let mut lut = Box::new([0f32; 256]);
                for (b, o) in lut.iter_mut().enumerate() {
                    *o = f32::from_bits((f2bf(crate::dtype::fp8_e4m3_to_f32(b as u8) * ws) as u32) << 16);
                }
                HeadRows::Fp8 { bytes: self.dl_bytes(&head.parts[0])?, lut }
            }
            _ => {
                return Err(Error::Unsupported(
                    "the DSpine tail with an mxfp4 lm_head (INTARSIA_LMHEAD_QUANT=mxfp4): use fp8 or off".into(),
                ))
            }
        };
        let cb_name = resolve(&*sp.src, "selected_prefix_bridge.codebook")?;
        let codebook = sp.src.bytes(&cb_name)?;
        let c = sp.code;
        let ch = sp.chunk;
        let mut tg = vec![0u16; v * r];
        let mut tsv = vec![0u16; v * 2 * r];
        let (stage_in, stage_out) = (&sp.bufs.stage_in, &sp.bufs.stage_out);
        let mut x = vec![0u16; ch * h];
        let mut y = vec![0u16; ch * c];
        let mut v0 = 0;
        while v0 < v {
            let n = ch.min(v - v0);
            // TG: x[v] = bf16(head[v] * norm.w), Matmul with write_4^T [R, H]
            x.iter_mut().for_each(|e| *e = 0);
            par_fill(&mut x[..n * h], h, |r0, out| {
                let mut row = vec![0f32; h];
                for (i, o) in out.chunks_mut(h).enumerate() {
                    rows.row_into(v0 + r0 + i, h, &mut row);
                    for j in 0..h {
                        o[j] = f2bf(row[j] * sp.norm_f[j]);
                    }
                }
            });
            self.backend.upload(stage_in, 0, &u16_bytes(&x), self.stream)?;
            let xin = self.view(stage_in, DType::Bf16, &[ch, h]);
            let out = self.view(stage_out, DType::Bf16, &[ch, r]);
            self.proj(&xin, &sp.write4_t, &out, false)?;
            let mut ob = vec![0u8; n * r * 2];
            self.backend.download(stage_out, 0, &mut ob, self.stream)?;
            tg[v0 * r..(v0 + n) * r].copy_from_slice(&le_u16(&ob));
            // TSV: y[v] = bf16(32 * codebook[v]) (a power of two: exact), Matmul with [G_p; A] [2R, C]
            y.iter_mut().for_each(|e| *e = 0);
            let scale = (c as f32).sqrt();
            par_fill(&mut y[..n * c], c, |r0, out| {
                for (i, o) in out.chunks_mut(c).enumerate() {
                    let src = &codebook[(v0 + r0 + i) * c * 2..(v0 + r0 + i + 1) * c * 2];
                    for j in 0..c {
                        o[j] = f2bf(scale * bf(u16::from_le_bytes([src[2 * j], src[2 * j + 1]])));
                    }
                }
            });
            self.backend.upload(stage_in, 0, &u16_bytes(&y), self.stream)?;
            let yin = self.view(stage_in, DType::Bf16, &[ch, c]);
            let out2 = self.view(stage_out, DType::Bf16, &[ch, 2 * r]);
            self.proj(&yin, &sp.pv_w, &out2, false)?;
            let mut ob2 = vec![0u8; n * 2 * r * 2];
            self.backend.download(stage_out, 0, &mut ob2, self.stream)?;
            tsv[v0 * 2 * r..(v0 + n) * 2 * r].copy_from_slice(&le_u16(&ob2));
            v0 += n;
        }
        eprintln!("dspine: tables built in {:.1} s (TG {} MiB, TSV {} MiB, head rows {} MiB)", t0.elapsed().as_secs_f64(), tg.len() * 2 >> 20, tsv.len() * 2 >> 20, match &rows {
            HeadRows::Bf16(b) => b.len() * 2 >> 20,
            HeadRows::Fp8 { bytes, .. } => bytes.len() >> 20,
        });
        let t = Arc::new(Tables { tg, tsv, head: rows });
        *slot = Some(t.clone());
        Ok(t)
    }

    /// Dry run: the table build's two matmuls (one chunk each), so the plan admits them.
    pub(super) fn dspine_dry_run_tables(&self, sp: &Spine) -> Result<()> {
        let (h, r, c) = (self.dims.hidden, sp.rank, sp.code);
        let ch = sp.chunk;
        let xin = self.view(&sp.bufs.stage_in, DType::Bf16, &[ch, h]);
        self.proj(&xin, &sp.write4_t, &self.view(&sp.bufs.stage_out, DType::Bf16, &[ch, r]), false)?;
        let yin = self.view(&sp.bufs.stage_in, DType::Bf16, &[ch, c]);
        self.proj(&yin, &sp.pv_w, &self.view(&sp.bufs.stage_out, DType::Bf16, &[ch, 2 * r]), false)
    }

    /// One block pass of a DSpine draft over `kv` (context of `kv.len` rows) with the block's ids already in `ws.ids` (`t` rows).
    /// Leaves the final normed rows in `ws.n` and the last layer's stash in the spine's buffers.
    pub(super) fn dspine_block(&self, d2: &Draft2, sp: &Spine, kv: &SequenceState, t: usize, profile: bool) -> Result<()> {
        let (h, act, ws) = (self.dims.hidden, self.act, &self.ws);
        let (nh, nkv, hd, inter) = (d2.n_heads, d2.n_kv_heads, d2.head_dim, d2.inter);
        let (c, r, nl) = (sp.code, sp.rank, d2.layers.len());
        let bu = &d2.bufs;
        let sb = &sp.bufs;
        let bf16 = DType::Bf16;
        let rows = |b: &Buffer, w: usize| self.view(b, act, &[t, w]);
        let (mut x, mut xn) = (rows(&bu.x[0], h), rows(&bu.x[1], h));
        let (n, o) = (rows(&bu.n, h), rows(&bu.o, h));
        let q3 = |i: usize| self.view(&bu.q[i], act, &[t, nh, hd]);
        let k3 = |i: usize| self.view(&bu.k[i], act, &[t, nkv, hd]);
        let v3 = self.view(&bu.v, act, &[t, nkv, hd]);
        let ao = self.view(&bu.ao, act, &[t, nh, hd]);
        let (g, u, a) = (rows(&bu.g, inter), rows(&bu.u, inter), rows(&bu.a, inter));
        let (raw, nrm) = (self.view(&sb.raw, bf16, &[t, h]), rows(&bu.c, h));
        let rms = self.view(&sb.rms, DType::F32, &[t]);
        let (pr, code, packed, msg, dlt) = (
            self.view(&sb.p, bf16, &[t, c]),
            self.view(&sb.code, bf16, &[t, c]),
            self.view(&sb.packed, bf16, &[t, 3 * r]),
            self.view(&sb.msg, bf16, &[t, r]),
            self.view(&sb.dlt, bf16, &[t, h]),
        );
        let s0 = self.view(&sb.s0, bf16, &[2 * r]);
        let final_n = self.view(&ws.n, act, &[t, h]);
        let p0 = kv.len;
        let add = |a: &TensorRef, b: &TensorRef, out: &TensorRef| self.exec(Op::Add, vec![a.clone(), b.clone()], vec![out.clone()], OpAttrs::default(), profile);
        self.exec(Op::Embedding, vec![self.view(&ws.ids, DType::I32, &[t]), self.embed.clone()], vec![x.clone()], OpAttrs::default(), profile)?;
        self.d2_norm(d2, &x, &d2.layers[0].ln1, &n, profile)?;
        if sp.trace.load(Ordering::Relaxed) {
            sp.layers_log.lock().unwrap().clear();
        }
        for (li, l) in d2.layers.iter().enumerate() {
            let slot = Self::attn_slot_at(kv, li)?;
            self.proj(&n, &l.wq, &q3(0).reshaped(&[t, nh * hd]), profile)?;
            self.proj(&n, &l.wk, &k3(0).reshaped(&[t, nkv * hd]), profile)?;
            self.proj(&n, &l.wv, &v3.reshaped(&[t, nkv * hd]), profile)?;
            self.d2_qk(d2, &q3(0), &l.q_norm, &q3(1), &q3(0), t, p0, profile)?;
            self.d2_qk(d2, &k3(0), &l.k_norm, &k3(1), &k3(0), t, p0, profile)?;
            // the block's own K/V go beyond the cache length (scratch, overwritten by the next ingest)
            self.d2_append(slot, &k3(0), &v3, p0, nkv, hd, profile)?;
            let shape = [nkv, slot.cap, hd];
            let kc = self.view(&slot.k, slot.dtype, &shape);
            let vc = self.view(&slot.v, slot.dtype, &shape);
            for ri in 0..t {
                // causal block: row ri at position p0 + ri sees the context and block rows 0 ..= ri, within the window
                let pos = p0 + ri;
                let win = if d2.window > 0 && pos + 1 > d2.window { d2.window } else { 0 };
                let attrs = OpAttrs {
                    pos0: pos,
                    scale: 1.0 / (hd as f32).sqrt(),
                    n_heads: nh,
                    n_kv_heads: nkv,
                    head_dim: hd,
                    kv_cap: slot.cap,
                    window: win,
                    win_split: d2.win_split,
                    ..Default::default()
                };
                self.exec(Op::AttnDecode, vec![q3(0).rows(ri, 1), kc.clone(), vc.clone()], vec![ao.rows(ri, 1)], attrs, profile)?;
            }
            self.d2_proj_out(&ao.reshaped(&[t, nh * hd]), &l.wo, &o, t, profile)?;
            add(&x, &o, &xn)?;
            std::mem::swap(&mut x, &mut xn);
            // feed-forward; the layer's output is `raw` (the bridge's input)
            self.d2_norm(d2, &x, &l.ln2, &n, profile)?;
            self.proj(&n, &l.gate, &g, profile)?;
            self.proj(&n, &l.up, &u, profile)?;
            self.exec(Op::SiluGlu, vec![g.clone(), u.clone()], vec![a.clone()], OpAttrs::default(), profile)?;
            self.d2_proj_out(&a, &l.down, &o, t, profile)?;
            add(&x, &o, &raw)?;
            // the code write
            let eb = OpAttrs { eps: sp.eps_b, ..Default::default() };
            self.exec(Op::BridgeNorm, vec![raw.clone()], vec![nrm.clone(), rms.clone()], eb.clone(), profile)?;
            self.proj(&nrm, &sp.read[li], &pr, profile)?;
            self.exec(Op::BridgeCode, vec![pr.clone()], vec![code.clone()], OpAttrs { scale: (c as f32).sqrt(), ..eb }, profile)?;
            self.proj(&code, &sp.packed_w, &packed, profile)?;
            self.exec(Op::BridgeGate, vec![packed.clone(), sp.bias.clone(), s0.clone()], vec![msg.clone()], OpAttrs::default(), profile)?;
            self.proj(&msg, &sp.write[li], &dlt, profile)?;
            // out -> xn (the next layer's residual), the next layer's norm (the final norm after the last layer) -> n (ws.n at the end)
            let (w_next, eps_n, nout) = if li + 1 < nl { (d2.layers[li + 1].ln1.clone(), d2.eps, n.clone()) } else { (self.dspine_final_norm()?, d2.eps, final_n.clone()) };
            self.exec(Op::BridgeWrite, vec![raw.clone(), rms.clone(), dlt.clone(), w_next], vec![xn.clone(), nout], OpAttrs { eps: eps_n, ..Default::default() }, profile)?;
            std::mem::swap(&mut x, &mut xn);
            if sp.trace.load(Ordering::Relaxed) && self.recorder.lock().unwrap().is_none() {
                let nrm_t = if li + 1 < nl { &n } else { &final_n };
                let dl = |tr: &TensorRef| -> Result<Vec<f32>> {
                    let b = self.dl_bytes(tr)?;
                    Ok(le_u16(&b).into_iter().map(bf).collect())
                };
                let pair = [dl(&x)?, dl(nrm_t)?, dl(&raw)?];
                sp.layers_log.lock().unwrap().push(pair);
            }
        }
        Ok(())
    }

    /// The draft's final `norm.weight` (the last write's norm weight), as the head's norm tensor of the DFlash head.
    fn dspine_final_norm(&self) -> Result<TensorRef> {
        self.dflash.as_ref().map(|h| h.norm.clone()).ok_or_else(|| Error::invalid("no DFlash head"))
    }

    /// Drafts of one DSpine pass: the block, the pass-1 logits on the live head, host candidates and the greedy tail. `t` rows
    /// (`2 ..= block`), the ids already in `ws.ids`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dspine_draft(&self, d2: &Draft2, sp: &Spine, kv: &SequenceState, next: u32, t: usize, rows_k: bool, profile: bool) -> Result<Vec<u32>> {
        let (h, v, act, ws) = (self.dims.hidden, self.dims.vocab, self.act, &self.ws);
        let recording = self.recorder.lock().unwrap().is_some();
        // INTARSIA_SPEC_TRACE=1: sync between the phases and print their host times
        let tr = crate::spec::trace_enabled() && !recording;
        let sync = |tr: bool| -> Result<Instant> {
            if tr {
                self.backend.synchronize(self.stream)?;
            }
            Ok(Instant::now())
        };
        let t_start = Instant::now();
        let tabs = if recording { None } else { Some(self.dspine_tabs(sp)?) };
        if let Some(tb) = &tabs {
            // the anchor token's TSV row is the sender of block row 1 (the bridge's row-0 packed sender is not used)
            let r2 = 2 * sp.rank;
            let a = next as usize;
            self.backend.upload(&sp.bufs.s0, 0, &u16_bytes(&tb.tsv[a * r2..(a + 1) * r2]), self.stream)?;
        } else {
            self.dspine_dry_run_tables(sp)?;
        }
        if rows_k {
            self.exact_rows.store(true, Ordering::Relaxed);
        }
        let mut t_block = t_start;
        let run = (|| -> Result<()> {
            self.dspine_block(d2, sp, kv, t, profile)?;
            t_block = sync(tr)?;
            let n = self.view(&ws.n, act, &[t, h]);
            let logits = self.view(&ws.logits, DType::F32, &[t - 1, v]);
            if t == 2 {
                // one head row: the row-exact (verify) kernels start at two rows, and the plain m = 1 head is what decode runs
                self.exact_rows.store(false, Ordering::Relaxed);
            }
            self.proj(&n.rows(1, t - 1), self.head()?, &logits, profile)
        })();
        if rows_k {
            self.exact_rows.store(false, Ordering::Relaxed);
        }
        run?;
        let Some(tabs) = tabs else { return Ok(Vec::new()) };
        let t_head = sync(tr)?;
        // downloads: logits rows, and the stash of the last layer (raw rows, rms, packed rows)
        let (n_rows, r) = (t - 1, sp.rank);
        let mut lb = vec![0u8; n_rows * v * 4];
        self.backend.download(&ws.logits, 0, &mut lb, self.stream)?;
        let mut logits = vec![0f32; n_rows * v];
        par_fill(&mut logits, v, |r0, out| {
            for (i, o) in out.chunks_mut(v).enumerate() {
                let src = &lb[(r0 + i) * v * 4..(r0 + i + 1) * v * 4];
                for (j, c) in src.chunks_exact(4).enumerate() {
                    o[j] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
        });
        let mut cand: Vec<Vec<usize>> = vec![Vec::new(); n_rows];
        {
            let logits = &logits;
            let k = sp.topk;
            std::thread::scope(|sc| {
                for (i, slot) in cand.iter_mut().enumerate() {
                    sc.spawn(move || *slot = candidates(&logits[i * v..(i + 1) * v], k));
                }
            });
        }
        let t_cand = Instant::now();
        let rawb = self.dl_bytes(&self.view(&sp.bufs.raw, DType::Bf16, &[t, h]))?;
        let rmsb = self.dl_bytes(&self.view(&sp.bufs.rms, DType::F32, &[t]))?;
        let pkb = self.dl_bytes(&self.view(&sp.bufs.packed, DType::Bf16, &[t, 3 * r]))?;
        let raw = le_u16(&rawb);
        let rms: Vec<f32> = rmsb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let packed = le_u16(&pkb);
        // rows 1 ..: raw, rms, and the receiver query q = packed[row, 0..R]
        let q: Vec<f32> = (1..t).flat_map(|row| packed[row * 3 * r..row * 3 * r + r].iter().map(|&b| bf(b)).collect::<Vec<_>>()).collect();
        let drafts = walk(&tabs, &sp.norm_f, &sp.bias_f, h, r, &raw[h..], &rms[1..], &q, &cand, next as usize);
        if tr {
            let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1e3;
            let t_end = Instant::now();
            eprintln!(
                "spec: dspine draft rows {t} block {:.3} head {:.3} logits+cand {:.3} stash+walk {:.3} total {:.3} ms",
                ms(t_start, t_block),
                ms(t_block, t_head),
                ms(t_head, t_cand),
                ms(t_cand, t_end),
                ms(t_start, t_end)
            );
        }
        *sp.last.lock().unwrap() = Some(LastPass {
            raw: raw.iter().map(|&b| bf(b)).collect(),
            rms: rms.clone(),
            packed: packed.iter().map(|&b| bf(b)).collect(),
            cand,
            drafts: drafts.clone(),
        });
        Ok(drafts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `HeadRows::dot` sums every column, also when `h` is not a multiple of 8 (bf16 and fp8 rows).
    #[test]
    fn head_dot_covers_the_remainder_columns() {
        for h in [1usize, 7, 8, 13, 20] {
            let x: Vec<f32> = (0..h).map(|j| 0.5 + j as f32 * 0.25).collect();
            let bits: Vec<u16> = (0..2 * h).map(|j| f2bf(1.0 + (j % 5) as f32)).collect();
            let rows = HeadRows::Bf16(bits.clone());
            for r in 0..2 {
                let want: f32 = (0..h).map(|j| x[j] * bf(bits[r * h + j])).sum();
                assert!((rows.dot(r, h, &x) - want).abs() <= 1e-4 * want.abs(), "bf16 h={h} row {r}");
            }
            let mut lut = Box::new([0f32; 256]);
            lut.iter_mut().enumerate().for_each(|(b, o)| *o = b as f32 * 0.125);
            let bytes: Vec<u8> = (0..2 * h).map(|j| (j * 7 % 251) as u8).collect();
            let rows = HeadRows::Fp8 { bytes: bytes.clone(), lut: lut.clone() };
            for r in 0..2 {
                let want: f32 = (0..h).map(|j| x[j] * lut[bytes[r * h + j] as usize]).sum();
                assert!((rows.dot(r, h, &x) - want).abs() <= 1e-4 * want.abs(), "fp8 h={h} row {r}");
            }
        }
    }
}
