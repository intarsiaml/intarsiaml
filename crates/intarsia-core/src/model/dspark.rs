//! Inkling-2 DSpark drafter (`mtp.safetensors`, `--spec dflash --spec-draft <model dir>`), lane ink-mtp,
//! docs/results/2026-11-21-inkling2-mtp.md. Numerics authority: vLLM's `models/inkling2/nvidia/dspark.py`, `ops/dspark.py` and
//! `speculator.py`; the Python oracle is `tools/inkling2_dspark_ref.py`.
//!
//! ```text
//!   taps    the partial entering target visits 5, 23, 41 and the final-norm output (`target_layer_ids` 5, 23, 41, 48),
//!           exported by `ink_pass` into the DFlash staging buffers `feat[0..4]`
//!   context per draft layer i: ctx_i = rms(sum_j softmax(fuse_weight)[i, j] tap_j; dflare_norm_i)
//!           K_i(p) = rope(k_norm_ctx(k_proj_ctx ctx_i)), V_i(p) = v_proj_ctx ctx_i       (cache slot i, absolute position p)
//!           f_rnn(p) = rms(sum_j softmax(fuse_weight)[3, j] tap_j; markov dflare_norm)  (kept on the host, a ring by position)
//!   block   rows [embed_norm(embed(anchor)), mask_embed x 7] at positions P .. P + 7 (P = target length); pre-norm dense layers
//!           with q / k RMSNorm + neox rope (theta 1e6); every row sees all 8 block keys and the context (local layers 0, 2: the
//!           1024 positions before the block), i.e. one `AttnDecode` per row at P + 7 with window 1032 (0: all keys)
//!   head    mask row i + 1 drafts the token at P + i + 1: base = unembed(h_{i+1}); Markov (rank 256) step i:
//!           raw = b + W [state, w1[prev], h_{i+1}, f_rnn(P - 1)], gate / cand / out = raw thirds (rounded to bf16),
//!           state = sigma(gate) state + (1 - sigma(gate)) tanh(cand), latent = tanh(out),
//!           logits = 30 tanh(bf(base + fold(latent)) / 30), fold = unembed . hidden_out folded at load ([V, 256] + [V])
//!   gate    (lane ink-mtp) drafting stops at the first step whose top-1 probability is below `INTARSIA_DSPARK_TAU`
//! ```
//!
//! The mixes, the Markov recurrence and the logits' reduction run on the host (a few vectors per step); the projections, attention,
//! MLPs, the unembed and the fold GEMV run on the device with existing ops. Drafts never change the output (every emitted token is
//! a verify row's argmax), the draft's numerics only move the acceptance rate.

use super::*;
use crate::state::DflashState;

/// Host ring of `f_rnn` rows (by position): a draft reads the row of the last committed position, which a rollback keeps.
pub(super) const FRNN_RING: usize = 64;
/// Rows of a draft block (anchor + masks); the masks are the proposals.
const BLOCK: usize = 8;
/// Default confidence gate (`INTARSIA_DSPARK_TAU`): measured on the oracle, docs/results/2026-11-21-inkling2-mtp.md.
const TAU_DEFAULT: f32 = 0.6;

/// `mtp_config` of the target's `config.json` (the loader's `open_dflash`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DsparkConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub eps: f32,
    pub theta: f32,
    pub local: Vec<usize>,
    pub window: usize,
    pub block: usize,
    pub rank: usize,
    /// Target visits whose input partial is tapped; the last entry (one past the last visit) is the final-norm output.
    pub taps: Vec<usize>,
    pub softcap: f32,
}

pub(super) struct DsLayer {
    ln1: TensorRef,
    ln2: TensorRef,
    dflare: Vec<f32>,
    wq: QTensor,
    wk: QTensor,
    wv: QTensor,
    wo: QTensor,
    q_norm: TensorRef,
    k_norm: TensorRef,
    k_norm_ctx: TensorRef,
    wk_ctx: QTensor,
    wv_ctx: QTensor,
    gate: QTensor,
    up: QTensor,
    down: QTensor,
    local: bool,
}

pub(super) struct Dspark {
    c: DsparkConfig,
    layers: Vec<DsLayer>,
    norm: TensorRef,
    /// `[7, H]` mask rows, copied into block rows 1..8.
    masks: TensorRef,
    /// softmax(fuse_weight) `[layers + 1, taps]`, host.
    alpha: Vec<f32>,
    frnn_norm: Vec<f32>,
    /// rnn_proj split by columns: `[state | prev]` on the host (f32 `[3r, 2r]`), `hidden` and `f_rnn` on the device (bf16).
    w_sp: Vec<f32>,
    w_h: TensorRef,
    w_f: TensorRef,
    rnn_bias: Vec<f32>,
    /// `markov_w1` `[V_padded, r]` bf16 bits (host).
    w1: Vec<u16>,
    /// Folded `unembed . hidden_out`: weight `[V, r]` (device), bias `[V]` (host, bf16-rounded).
    fold_w: QTensor,
    fold_b: Vec<f32>,
    tau: f32,
    bufs: DsBufs,
}

struct DsBufs {
    x: [Buffer; 2],
    n: Buffer,
    o: Buffer,
    q: [Buffer; 2],
    k: [Buffer; 2],
    v: Buffer,
    ao: Buffer,
    g: Buffer,
    u: Buffer,
    a: Buffer,
    emb: Buffer,
    /// context ingest: `[rows, H]` mixes (act, one per draft layer), `[rows, kvdim]` k / v
    cx: Vec<Buffer>,
    ik: [Buffer; 2],
    iv: Buffer,
    /// head: base logits `[7, V]` (act), fold output `[1, V]`, latent `[1, r]`, f_rnn `[1, H]`, context parts (f32)
    base: Buffer,
    fold: Buffer,
    latent: Buffer,
    frnn: Buffer,
    hw: Buffer,
    fw: Buffer,
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn rbf(x: f32) -> f32 {
    f32::from_bits((bf16_bits(x) as u32) << 16)
}

fn from_bits(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A host f32 vector of a bf16 / f32 tensor.
fn host_f32(src: &dyn WeightSource, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = src.info(name).ok_or_else(|| Error::model(format!("DSpark tensor '{name}' missing")))?;
    if info.shape != shape {
        return Err(Error::model(format!("DSpark tensor '{name}' is {:?}, expected {shape:?}", info.shape)));
    }
    let raw = src.bytes(name)?;
    let mut out = vec![0f32; shape.iter().product()];
    crate::dtype::decode_to_f32(info.dtype, raw, &mut out);
    Ok(out)
}

fn upload_bf16(b: &dyn Backend, add_bytes: &dyn Fn(u64), vals: &[f32], shape: &[usize]) -> Result<TensorRef> {
    let bytes: Vec<u8> = vals.iter().flat_map(|&v| bf16_bits(v).to_le_bytes()).collect();
    add_bytes(bytes.len() as u64);
    Ok(TensorRef::new(crate::backend::upload_new(b, &bytes)?, DType::Bf16, shape))
}

/// `unembed[0..v] . hidden_out` (`[v, r]`) and `unembed . hidden_out_bias` (`[v]`), f32 accumulation over the bf16 inputs, rounded
/// to bf16 (vLLM folds in fp64; the difference only moves draft logits by a bf16 ulp), on all host cores.
fn fold(unembed: &[u8], v: usize, h: usize, ho_w: &[f32], ho_b: &[f32], r: usize) -> (Vec<u16>, Vec<f32>) {
    // ho_w is [h, r]: transpose to [r, h] so each output is a contiguous dot product
    let mut wt = vec![0f32; r * h];
    for d in 0..h {
        for j in 0..r {
            wt[j * h + d] = ho_w[d * r + j];
        }
    }
    let mut fw = vec![0u16; v * r];
    let mut fb = vec![0f32; v];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
    let per = v.div_ceil(threads);
    std::thread::scope(|sc| {
        for (ci, (wc, bc)) in fw.chunks_mut(per * r).zip(fb.chunks_mut(per)).enumerate() {
            let wt = &wt;
            sc.spawn(move || {
                let mut row = vec![0f32; h];
                for (i, b) in bc.iter_mut().enumerate() {
                    let vi = ci * per + i;
                    let src = &unembed[vi * h * 2..(vi + 1) * h * 2];
                    for (x, c) in row.iter_mut().zip(src.chunks_exact(2)) {
                        *x = from_bits(u16::from_le_bytes([c[0], c[1]]));
                    }
                    for j in 0..r {
                        let w = &wt[j * h..(j + 1) * h];
                        let s: f32 = row.iter().zip(w).map(|(a, b)| a * b).sum();
                        wc[i * r + j] = bf16_bits(s);
                    }
                    *b = rbf(row.iter().zip(ho_b).map(|(a, b)| a * b).sum());
                }
            });
        }
    });
    (fw, fb)
}

/// The DSpark draft onto the device (`dflash::load_head` with `DflashConfig::dspark`). `target` is the target checkpoint (its unembed
/// is folded into the Markov head here); `rows` the most context rows one ingest takes.
#[allow(clippy::too_many_arguments)]
pub(super) fn load(
    dl: &super::dflash::DflashLoad,
    c: &DsparkConfig,
    target: &dyn WeightSource,
    unembed_name: &str,
    b: &dyn Backend,
    dims: &ModelDims,
    add_bytes: &dyn Fn(u64),
    rows: usize,
    act: DType,
) -> Result<(Dspark, Vec<Buffer>)> {
    let src = &*dl.src;
    let (h, hd, r) = (dims.hidden, c.head_dim, c.rank);
    if act != DType::Bf16 {
        return Err(Error::Unsupported(format!("the DSpark draft with {act} activations (bf16 only)")));
    }
    if c.taps.len() != 4 || c.n_heads % c.n_kv_heads.max(1) != 0 || c.block != BLOCK || c.n_layers == 0 {
        return Err(Error::model(format!("DSpark config {c:?}: needs 4 taps, block {BLOCK}, whole GQA groups")));
    }
    let (qdim, kvdim, inter) = (c.n_heads * hd, c.n_kv_heads * hd, c.intermediate);
    let lin = |p: &str, n: usize, k: usize| super::dflash::lin(src, b, add_bytes, p, n, k);
    let norm_w = |name: &str, n: usize| -> Result<TensorRef> {
        let t = load_tensor(b, src, name, Some(&[n]))?;
        add_bytes(t.nbytes() as u64);
        Ok(t)
    };
    let mut layers = Vec::with_capacity(c.n_layers);
    for l in 0..c.n_layers {
        let p = format!("layers.{l}");
        layers.push(DsLayer {
            ln1: norm_w(&format!("{p}.input_layernorm.weight"), h)?,
            ln2: norm_w(&format!("{p}.post_attention_layernorm.weight"), h)?,
            dflare: host_f32(src, &format!("{p}.dflare_norm.weight"), &[h])?,
            wq: lin(&format!("{p}.self_attn.q_proj"), qdim, h)?,
            wk: lin(&format!("{p}.self_attn.k_proj"), kvdim, h)?,
            wv: lin(&format!("{p}.self_attn.v_proj"), kvdim, h)?,
            wo: lin(&format!("{p}.self_attn.o_proj"), h, qdim)?,
            q_norm: norm_w(&format!("{p}.self_attn.q_norm.weight"), hd)?,
            k_norm: norm_w(&format!("{p}.self_attn.k_norm.weight"), hd)?,
            k_norm_ctx: norm_w(&format!("{p}.self_attn.k_norm_ctx.weight"), hd)?,
            wk_ctx: lin(&format!("{p}.self_attn.k_proj_ctx"), kvdim, h)?,
            wv_ctx: lin(&format!("{p}.self_attn.v_proj_ctx"), kvdim, h)?,
            gate: lin(&format!("{p}.mlp.gate_proj"), inter, h)?,
            up: lin(&format!("{p}.mlp.up_proj"), inter, h)?,
            down: lin(&format!("{p}.mlp.down_proj"), h, inter)?,
            local: c.local.contains(&l),
        });
    }
    let norm = norm_w("norm.weight", h)?;
    let mask = host_f32(src, "mask_embed", &[h])?;
    let masks = upload_bf16(b, add_bytes, &mask.repeat(BLOCK - 1), &[BLOCK - 1, h])?;
    let nt = c.taps.len();
    let fw = host_f32(src, "fuse_weight", &[c.n_layers + 1, nt])?;
    let mut alpha = vec![0f32; fw.len()];
    for (o, row) in alpha.chunks_mut(nt).zip(fw.chunks(nt)) {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let s: f32 = row.iter().map(|x| (x - m).exp()).sum();
        for (a, x) in o.iter_mut().zip(row) {
            *a = (x - m).exp() / s;
        }
    }
    let frnn_norm = host_f32(src, "markov_head.dflare_norm.weight", &[h])?;
    let wr = host_f32(src, "markov_head.rnn_proj.weight", &[3 * r, 2 * r + 2 * h])?;
    let rnn_bias = host_f32(src, "markov_head.rnn_proj.bias", &[3 * r])?;
    let width = 2 * r + 2 * h;
    let cols = |lo: usize, n: usize| -> Vec<f32> { (0..3 * r).flat_map(|i| wr[i * width + lo..i * width + lo + n].to_vec()).collect() };
    let w_sp = cols(0, 2 * r);
    let w_h = upload_bf16(b, add_bytes, &cols(2 * r, h), &[3 * r, h])?;
    let w_f = upload_bf16(b, add_bytes, &cols(2 * r + h, h), &[3 * r, h])?;
    let w1_info = src.info("markov_head.markov_w1.weight").ok_or_else(|| Error::model("DSpark tensor 'markov_head.markov_w1.weight' missing"))?;
    if w1_info.dtype != DType::Bf16 || w1_info.shape.len() != 2 || w1_info.shape[1] != r || w1_info.shape[0] < dims.vocab {
        return Err(Error::model(format!("DSpark markov_w1 is {} {:?}, expected bf16 [>= {}, {r}]", w1_info.dtype, w1_info.shape, dims.vocab)));
    }
    let w1: Vec<u16> = src.bytes("markov_head.markov_w1.weight")?.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
    // the fold of the target's unembed (its first `dims.vocab` rows, the ones the model's head uses)
    let ui = target.info(unembed_name).ok_or_else(|| Error::model(format!("DSpark fold: target tensor '{unembed_name}' missing")))?;
    if ui.dtype != DType::Bf16 || ui.shape.len() != 2 || ui.shape[1] != h || ui.shape[0] < dims.vocab {
        return Err(Error::model(format!("DSpark fold: '{unembed_name}' is {} {:?}, expected bf16 [>= {}, {h}]", ui.dtype, ui.shape, dims.vocab)));
    }
    let ho_w = host_f32(src, "markov_head.hidden_out.weight", &[h, r])?;
    let ho_b = host_f32(src, "markov_head.hidden_out.bias", &[h])?;
    let t0 = Instant::now();
    let (fw_bits, fold_b) = fold(target.bytes(unembed_name)?, dims.vocab, h, &ho_w, &ho_b, r);
    let fbytes: Vec<u8> = fw_bits.iter().flat_map(|x| x.to_le_bytes()).collect();
    add_bytes(fbytes.len() as u64);
    let fold_w = QTensor {
        format: crate::quant::BF16,
        n: dims.vocab,
        k: r,
        parts: vec![TensorRef::new(crate::backend::upload_new(b, &fbytes)?, DType::Bf16, &[dims.vocab, r])],
        act_fp8: false,
        act_quant: false,
    };
    eprintln!("dspark: Markov fold [{} x {r}] in {:.1} s", dims.vocab, t0.elapsed().as_secs_f64());
    let es = act.size_bytes();
    let alloc = |elems: usize| -> Result<Buffer> {
        add_bytes((elems * es) as u64);
        b.alloc((elems * es).max(16), MemKind::Device)
    };
    let f32buf = |elems: usize| -> Result<Buffer> {
        add_bytes((elems * 4) as u64);
        b.alloc((elems * 4).max(16), MemKind::Device)
    };
    let t = BLOCK;
    let bufs = DsBufs {
        x: [alloc(t * h)?, alloc(t * h)?],
        n: alloc(t * h)?,
        o: alloc(t * h)?,
        q: [alloc(t * qdim)?, alloc(t * qdim)?],
        k: [alloc(t * kvdim)?, alloc(t * kvdim)?],
        v: alloc(t * kvdim)?,
        ao: alloc(t * qdim)?,
        g: alloc(t * inter)?,
        u: alloc(t * inter)?,
        a: alloc(t * inter)?,
        emb: alloc(h)?,
        cx: (0..c.n_layers).map(|_| alloc(rows * h)).collect::<Result<_>>()?,
        ik: [alloc(rows * kvdim)?, alloc(rows * kvdim)?],
        iv: alloc(rows * kvdim)?,
        base: alloc((BLOCK - 1) * dims.vocab)?,
        fold: alloc(dims.vocab)?,
        latent: alloc(r)?,
        frnn: alloc(h)?,
        hw: f32buf((BLOCK - 1) * 3 * r)?,
        fw: f32buf(3 * r)?,
    };
    let mut feat = Vec::with_capacity(nt);
    for _ in 0..nt {
        feat.push(alloc(rows * h)?);
    }
    let tau = std::env::var("INTARSIA_DSPARK_TAU").ok().and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(TAU_DEFAULT);
    let ds = Dspark {
        c: c.clone(),
        layers,
        norm,
        masks,
        alpha,
        frnn_norm,
        w_sp,
        w_h,
        w_f,
        rnn_bias,
        w1,
        fold_w,
        fold_b,
        tau,
        bufs,
    };
    Ok((ds, feat))
}

impl Dspark {
    pub(super) fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub(super) fn norm_ref(&self) -> TensorRef {
        self.norm.clone()
    }

    pub(super) fn taps(&self) -> &[usize] {
        &self.c.taps
    }

    pub(super) fn slot_specs(&self) -> Vec<SlotSpec> {
        (0..self.layers.len()).map(|_| SlotSpec::AttnKv { n_kv_heads: self.c.n_kv_heads, head_dim: self.c.head_dim, dtype: DType::Bf16 }).collect()
    }

    /// `rms(sum_j alpha[row, j] tap_j; w)` of one position (taps `[4][H]` slices).
    fn mix(&self, row: usize, taps: &[&[f32]], w: &[f32], out: &mut [f32]) {
        let nt = self.c.taps.len();
        let a = &self.alpha[row * nt..(row + 1) * nt];
        for (d, o) in out.iter_mut().enumerate() {
            *o = taps.iter().zip(a).map(|(t, &al)| t[d] * al).sum();
        }
        let ms: f32 = out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32;
        let inv = 1.0 / (ms + self.c.eps).sqrt();
        for (o, &g) in out.iter_mut().zip(w) {
            *o = rbf(*o * inv * g);
        }
    }
}

impl Model {
    fn ds_download(&self, b: &Buffer, dtype: DType, off_elems: usize, n: usize) -> Result<Vec<f32>> {
        let es = dtype.size_bytes();
        let mut raw = vec![0u8; n * es];
        self.backend.download(b, off_elems * es, &mut raw, self.stream)?;
        let mut out = vec![0f32; n];
        crate::dtype::decode_to_f32(dtype, &raw, &mut out);
        Ok(out)
    }

    fn ds_upload_bf16(&self, b: &Buffer, vals: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = vals.iter().flat_map(|&v| bf16_bits(v).to_le_bytes()).collect();
        self.backend.upload(b, 0, &bytes, self.stream)
    }

    /// Context rows of `t` target positions (`kv.len ..`) from the tap buffers `src` into the draft caches, and their `f_rnn` rows
    /// into the host ring.
    pub(super) fn dspark_ingest(&self, ds: &Dspark, d: &mut DflashState, src: &[Buffer], t: usize, profile: bool) -> Result<()> {
        let (h, act) = (self.dims.hidden, self.act);
        let (nkv, hd) = (ds.c.n_kv_heads, ds.c.head_dim);
        if self.recorder.lock().unwrap().is_some() {
            return Ok(());
        }
        self.backend.synchronize(self.stream)?;
        let taps: Vec<Vec<f32>> = src.iter().map(|b| self.ds_download(b, act, 0, t * h)).collect::<Result<_>>()?;
        let pos0 = d.kv.len;
        let row = |p: usize| -> Vec<&[f32]> { taps.iter().map(|x| &x[p * h..(p + 1) * h]).collect() };
        // f_rnn of the last FRNN_RING rows
        if d.frnn.len() != FRNN_RING * h {
            d.frnn = vec![0f32; FRNN_RING * h];
        }
        for p in t.saturating_sub(FRNN_RING)..t {
            let slot = (pos0 + p) % FRNN_RING;
            let mut out = vec![0f32; h];
            ds.mix(ds.c.n_layers, &row(p), &ds.frnn_norm, &mut out);
            d.frnn[slot * h..(slot + 1) * h].copy_from_slice(&out);
        }
        let k0 = self.view(&ds.bufs.ik[0], act, &[t, nkv, hd]);
        let k1 = self.view(&ds.bufs.ik[1], act, &[t, nkv, hd]);
        let v = self.view(&ds.bufs.iv, act, &[t, nkv, hd]);
        let mut mixed = vec![0f32; t * h];
        for (li, l) in ds.layers.iter().enumerate() {
            for p in 0..t {
                ds.mix(li, &row(p), &l.dflare, &mut mixed[p * h..(p + 1) * h]);
            }
            self.ds_upload_bf16(&ds.bufs.cx[li], &mixed)?;
            let cx = self.view(&ds.bufs.cx[li], act, &[t, h]);
            let slot = Self::attn_slot_at(&d.kv, li)?;
            self.proj(&cx, &l.wk_ctx, &k0.reshaped(&[t, nkv * hd]), profile)?;
            self.proj(&cx, &l.wv_ctx, &v.reshaped(&[t, nkv * hd]), profile)?;
            self.ds_qk(ds, &k0, &l.k_norm_ctx, &k1, &k0, pos0, profile)?;
            self.d2_append(slot, &k0, &v, pos0, nkv, hd, profile)?;
        }
        d.kv.advance_rows(t);
        Ok(())
    }

    /// q / k norm then neox rope at positions `pos0 ..` (the draft's theta, full rotary): `x` -> `tmp` -> `out`.
    #[allow(clippy::too_many_arguments)]
    fn ds_qk(&self, ds: &Dspark, x: &TensorRef, w: &TensorRef, tmp: &TensorRef, out: &TensorRef, pos0: usize, profile: bool) -> Result<()> {
        self.exec(Op::QkNorm, vec![x.clone(), w.clone()], vec![tmp.clone()], OpAttrs { eps: ds.c.eps, w_offset: 0.0, ..Default::default() }, profile)?;
        let rope = OpAttrs { pos0, theta: ds.c.theta, head_dim: ds.c.head_dim, rotary_dim: 0, ..Default::default() };
        self.exec(Op::Rope, vec![tmp.clone()], vec![out.clone()], rope, profile)
    }

    /// Drafts after `next` (the token at position `st.len`): one block pass and up to `k` (at most 7) Markov steps, stopping at the
    /// first proposal whose top-1 probability is below the gate. Empty: no proposal cleared the gate (a plain step follows).
    pub(super) fn dspark_draft(&self, ds: &Dspark, st: &mut SequenceState, next: u32, k: usize) -> Result<Vec<u32>> {
        let d = st.dflash.as_ref().ok_or_else(|| Error::invalid("sequence without DFlash state"))?;
        let kmax = k.min(BLOCK - 1);
        let p0 = d.kv.len;
        if kmax == 0 || p0 == 0 || p0 + BLOCK > d.kv.cap || self.recorder.lock().unwrap().is_some() {
            return Ok(Vec::new());
        }
        if next as usize >= self.dims.vocab {
            return Err(Error::BadRequest(format!("token id {next} >= vocab {}", self.dims.vocab)));
        }
        let im = self.ink.as_ref().ok_or_else(|| Error::invalid("the DSpark draft on a model other than Inkling-2"))?;
        let head = self.lm_head.as_ref().ok_or_else(|| Error::invalid("the DSpark draft needs the model's lm_head"))?;
        let profile = false;
        let (h, act, t) = (self.dims.hidden, self.act, BLOCK);
        let (nh, nkv, hd, inter, r) = (ds.c.n_heads, ds.c.n_kv_heads, ds.c.head_dim, ds.c.intermediate, ds.c.rank);
        let bf = &ds.bufs;
        let rows = |b: &Buffer, w: usize| self.view(b, act, &[t, w]);
        let (mut x, mut xn) = (rows(&bf.x[0], h), rows(&bf.x[1], h));
        let (n, o) = (rows(&bf.n, h), rows(&bf.o, h));
        let q3 = |i: usize| self.view(&bf.q[i], act, &[t, nh, hd]);
        let k3 = |i: usize| self.view(&bf.k[i], act, &[t, nkv, hd]);
        let v3 = self.view(&bf.v, act, &[t, nkv, hd]);
        let ao = self.view(&bf.ao, act, &[t, nh, hd]);
        let (g, u, a) = (rows(&bf.g, inter), rows(&bf.u, inter), rows(&bf.a, inter));
        let add = |a: &TensorRef, b: &TensorRef, out: &TensorRef| self.exec(Op::Add, vec![a.clone(), b.clone()], vec![out.clone()], OpAttrs::default(), profile);
        let eps_t = OpAttrs { eps: self.dims.rms_eps, ..OpAttrs::default() };
        let eps_d = OpAttrs { eps: ds.c.eps, w_offset: 0.0, ..OpAttrs::default() };
        // block rows: the anchor's normed target embedding, then the masks
        self.backend.upload(&self.ws.ids, 0, &(next as i32).to_le_bytes(), self.stream)?;
        let emb = self.view(&bf.emb, act, &[1, h]);
        self.exec(Op::Embedding, vec![self.view(&self.ws.ids, DType::I32, &[1]), self.embed.clone()], vec![emb.clone()], OpAttrs::default(), profile)?;
        self.exec(Op::RmsNorm, vec![emb, im.embed_norm.clone()], vec![x.rows(0, 1)], eps_t, profile)?;
        let mrows = x.rows(1, t - 1);
        self.backend.copy(&mrows.buf, mrows.offset, &ds.masks.buf, ds.masks.offset, ds.masks.nbytes(), self.stream)?;
        let last = p0 + t - 1;
        for (li, l) in ds.layers.iter().enumerate() {
            let slot = Self::attn_slot_at(&st.dflash.as_ref().unwrap().kv, li)?;
            self.exec(Op::RmsNorm, vec![x.clone(), l.ln1.clone()], vec![n.clone()], eps_d.clone(), profile)?;
            self.proj(&n, &l.wq, &q3(0).reshaped(&[t, nh * hd]), profile)?;
            self.proj(&n, &l.wk, &k3(0).reshaped(&[t, nkv * hd]), profile)?;
            self.proj(&n, &l.wv, &v3.reshaped(&[t, nkv * hd]), profile)?;
            self.ds_qk(ds, &q3(0), &l.q_norm, &q3(1), &q3(0), p0, profile)?;
            self.ds_qk(ds, &k3(0), &l.k_norm, &k3(1), &k3(0), p0, profile)?;
            // the block's K / V beyond the context (scratch: the next ingest overwrites them)
            self.d2_append(slot, &k3(0), &v3, p0, nkv, hd, profile)?;
            let shape = [nkv, slot.cap, hd];
            let kc = self.view(&slot.k, slot.dtype, &shape);
            let vc = self.view(&slot.v, slot.dtype, &shape);
            let win = if l.local { ds.c.window + t } else { 0 };
            let win = if win == 0 || last < win { 0 } else { win };
            for rr in 0..t {
                let attrs = OpAttrs {
                    pos0: last,
                    scale: 1.0 / (hd as f32).sqrt(),
                    n_heads: nh,
                    n_kv_heads: nkv,
                    head_dim: hd,
                    kv_cap: slot.cap,
                    window: win,
                    ..Default::default()
                };
                self.exec(Op::AttnDecode, vec![q3(0).rows(rr, 1), kc.clone(), vc.clone()], vec![ao.rows(rr, 1)], attrs, profile)?;
            }
            self.proj(&ao.reshaped(&[t, nh * hd]), &l.wo, &o, profile)?;
            add(&x, &o, &xn)?;
            std::mem::swap(&mut x, &mut xn);
            self.exec(Op::RmsNorm, vec![x.clone(), l.ln2.clone()], vec![n.clone()], eps_d.clone(), profile)?;
            self.proj(&n, &l.gate, &g, profile)?;
            self.proj(&n, &l.up, &u, profile)?;
            self.exec(Op::SiluGlu, vec![g.clone(), u.clone()], vec![a.clone()], OpAttrs::default(), profile)?;
            self.proj(&a, &l.down, &o, profile)?;
            add(&x, &o, &xn)?;
            std::mem::swap(&mut x, &mut xn);
        }
        self.exec(Op::RmsNorm, vec![x.clone(), ds.norm.clone()], vec![n.clone()], eps_d, profile)?;
        // head inputs: base logits of the mask rows, the hidden part of rnn_proj; the f_rnn part of the last committed position
        let hs = n.rows(1, kmax);
        let vocab = self.dims.vocab;
        let base = self.view(&bf.base, act, &[kmax, vocab]);
        self.proj(&hs, head, &base, profile)?;
        let hw = self.view(&bf.hw, DType::F32, &[kmax, 3 * r]);
        self.exec(Op::Logits, vec![hs, ds.w_h.clone()], vec![hw], OpAttrs::default(), profile)?;
        let dstate = st.dflash.as_ref().unwrap();
        let slot = (p0 - 1) % FRNN_RING;
        if dstate.frnn.len() != FRNN_RING * h {
            return Ok(Vec::new());
        }
        self.ds_upload_bf16(&bf.frnn, &dstate.frnn[slot * h..(slot + 1) * h])?;
        let fwv = self.view(&bf.fw, DType::F32, &[1, 3 * r]);
        self.exec(Op::Logits, vec![self.view(&bf.frnn, act, &[1, h]), ds.w_f.clone()], vec![fwv], OpAttrs::default(), profile)?;
        self.backend.synchronize(self.stream)?;
        let hw = self.ds_download(&bf.hw, DType::F32, 0, kmax * 3 * r)?;
        let fw = self.ds_download(&bf.fw, DType::F32, 0, 3 * r)?;
        // the Markov recurrence, one proposal per step
        let mut state = vec![0f32; r];
        let mut prev = next as usize;
        let mut out = Vec::with_capacity(kmax);
        let sc = ds.c.softcap;
        let latent_t = self.view(&bf.latent, act, &[1, r]);
        let fold_t = self.view(&bf.fold, act, &[1, vocab]);
        for i in 0..kmax {
            let w1 = &ds.w1[prev * r..(prev + 1) * r];
            let mut raw = vec![0f32; 3 * r];
            for (j, z) in raw.iter_mut().enumerate() {
                let w = &ds.w_sp[j * 2 * r..(j + 1) * 2 * r];
                let mut s = ds.rnn_bias[j] + hw[i * 3 * r + j] + fw[j];
                for c in 0..r {
                    s += w[c] * state[c] + w[r + c] * from_bits(w1[c]);
                }
                *z = s;
            }
            let mut latent = vec![0f32; r];
            for c in 0..r {
                let gate = rbf(sigmoid(rbf(raw[c])));
                let cand = rbf(rbf(raw[r + c]).tanh());
                let keep = rbf(gate * state[c]);
                let upd = rbf(rbf(1.0 - gate) * cand);
                state[c] = rbf(keep + upd);
                latent[c] = rbf(rbf(raw[2 * r + c]).tanh());
            }
            self.ds_upload_bf16(&bf.latent, &latent)?;
            self.proj(&latent_t, &ds.fold_w, &fold_t, profile)?;
            self.backend.synchronize(self.stream)?;
            let fo = self.ds_download(&bf.fold, act, 0, vocab)?;
            let br = self.ds_download(&bf.base, act, i * vocab, vocab)?;
            // x = bf(base + bf(fold + fold_b)); softcap is monotone: argmax on x (first index on ties)
            let xs: Vec<f32> = br.iter().zip(&fo).zip(&ds.fold_b).map(|((b, f), fb)| rbf(b + rbf(f + fb))).collect();
            let (mut best, mut bx) = (0usize, f32::NEG_INFINITY);
            for (j, &v) in xs.iter().enumerate() {
                if v > bx {
                    bx = v;
                    best = j;
                }
            }
            // top-1 probability of the softcapped logits; terms more than 25 below the max (in capped units) are negligible
            let cmax = sc * (bx / sc).tanh();
            let lo = cmax - 25.0;
            let thr = if lo / sc <= -1.0 { f32::NEG_INFINITY } else { sc * (lo / sc).atanh() };
            let z: f32 = xs.iter().filter(|&&v| v > thr).map(|&v| (sc * (v / sc).tanh() - cmax).exp()).sum();
            let p1 = 1.0 / z;
            if crate::spec::trace_enabled() {
                eprintln!("dspark: pos {p0} step {i} draft {best} p1 {p1:.3}");
            }
            if p1 < ds.tau {
                break;
            }
            out.push(best as u32);
            prev = best;
        }
        Ok(out)
    }
}
