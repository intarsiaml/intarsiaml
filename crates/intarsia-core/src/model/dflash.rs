//! DFlash parallel block drafter (`--spec dflash --spec-draft DIR`), lane dc, docs/results/2026-10-25-dc-dflash-dcoco.md.
//!
//! The draft checkpoint (`TD-Labs/Qwen3-8B-DCOCO` layout: `layers.N.*`, `fc`, `hidden_norm`, `norm`; embedding and lm_head are
//! the target's, byte for byte) is five dense Qwen3 layers with non-causal attention:
//!
//! ```text
//!   taps    feat_p = concat(out_l(p) for l in layer_ids)        out_l = the target's residual stream after layer l (0-based)
//!   context h_ctx  = hidden_norm(fc(feat_p))                    for every target position p; fc [H, taps * H]
//!           K_l(p) = rope(k_norm(k_proj_l(h_ctx)), p)           V_l(p) = v_proj_l(h_ctx)       (layer l of the draft, cached)
//!   block   rows   = [anchor, MASK x (block - 1)]               at positions P .. P + block - 1, P = target length
//!           x      = embed(rows); per layer: x += o_proj(attn(input_layernorm(x))); x += mlp(post_attention_layernorm(x))
//!           attn   : the rows' own K/V are appended after the context, every row sees every key (no causal mask)
//!   drafts  argmax(lm_head(norm(x))) of ALL block rows, row r drafts the token at position P + r + 1
//! ```
//!
//! The context cache is written right after each target pass that processed the positions (`dflash_after_forward`, from the
//! taps' staging buffers), so back-off steps leave no hole; a target rollback shortens it to the same length. The block's K/V
//! are scratch beyond the cache length. Attention for the block runs one `AttnDecode` per row at the block's last position
//! (every key visible, no new kernel). Drafts never change the output (every emitted token is a verify row's argmax), the draft's
//! numerics only move the acceptance rate; the exactness of the verify pass itself is `Model::verify_rows_attn`.

use super::*;
use crate::state::DflashState;

/// Rows of single-token decode steps the draft keeps back before one batched ingest (`DflashState::store`, 8 KB per tap and row).
const PEND_MAX: usize = 512;

/// What the draft checkpoint's `config.json` (`dflash_config`) says, parsed by the loader.
#[derive(Clone, Debug, Default)]
pub struct DflashConfig {
    /// 0-based target layers whose outputs feed the draft, in `fc` column order.
    pub layer_ids: Vec<usize>,
    pub mask_id: u32,
    /// Query rows of the block = tokens drafted per pass (`dflash_config.block_size`).
    pub block: usize,
    /// The draft's own geometry, checked against the target's.
    pub hidden: usize,
    pub n_layers: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    /// DCOCO corrector geometry (`dcoco_candidate_topk`, `dcoco_emb_dim`, `dcoco_state_dim`, `dcoco_mlp_dim`); 0 when absent.
    pub topk: usize,
    pub emb_dim: usize,
    pub state_dim: usize,
    pub mlp_dim: usize,
    /// A `DFlash2DraftModel` checkpoint (`model/dflash2.rs`): its own geometry, norms and rope, the grouped dynamic conv, the
    /// sliding window, drafts from block rows 1.. (lane dfh). The fields below are read for it only.
    pub v2: bool,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// `sliding_window` (0 = the whole context).
    pub sliding_window: usize,
    /// `dflash_config.conv_kernel_size` / `conv_group_size`.
    pub conv_kernel: usize,
    pub conv_group: usize,
    /// `dflash_config.selector_rank` / `selector_top_k` (0: no candidate selector, drafts are the rows' argmax).
    pub selector_rank: usize,
    pub selector_top_k: usize,
}

/// `LoadOptions::dflash`: the draft checkpoint's weights and config.
#[derive(Clone)]
pub struct DflashLoad {
    pub src: Arc<dyn WeightSource>,
    pub config: DflashConfig,
    /// `--dcoco`: also load the corrector (`model/dcoco.rs`) and let it pick the drafts.
    pub dcoco: bool,
}

impl std::fmt::Debug for DflashLoad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DflashLoad").field("config", &self.config).finish_non_exhaustive()
    }
}

pub(super) struct DflashHead {
    /// Draft layer l: its attention (cache slot l of [`DflashState::kv`]) and feed-forward block.
    pub(super) layers: Vec<(AttnBlock, FfnBlock)>,
    /// `fc` split by columns into one `[H, H]` projection per tap (summed), so no concat buffer is needed.
    fc: Vec<QTensor>,
    hidden_norm: TensorRef,
    norm: TensorRef,
    /// Target layers tapped; `feat[i]` is the staging buffer of tap `i`, `[max rows, H]`.
    pub(super) taps: Vec<usize>,
    pub(super) feat: Vec<Buffer>,
    mask_id: u32,
    /// Query rows (= drafts) per pass.
    pub(super) block: usize,
    /// The DCOCO corrector (`--dcoco`).
    pub(super) dcoco: Option<super::dcoco::Corrector>,
    /// A DFlash2 draft (`model/dflash2.rs`): its layers (then `layers` is empty).
    pub(super) v2: Option<super::dflash2::Draft2>,
    /// The context and final norms' eps and scale offset: the target recipe's for DFlash, standard RMSNorm for DFlash2.
    eps: f32,
    norm_offset: f32,
    /// Tests: replaces the drafts (after the draft ran), see [`Model::dflash_script`].
    script: Mutex<Option<super::DraftScript>>,
    /// Probes: profile the draft's ops (synchronised, timed per op), see [`Model::dflash_profile`].
    profile: std::sync::atomic::AtomicBool,
    /// Rank 0 of a tensor-parallel group with a vocab-parallel head: a whole copy of the head for the draft's logits (the
    /// group's head is split across the ranks). `None`: the model's own head is whole.
    pub(super) full_head: Option<QTensor>,
    /// Run the DFlash2 block pass and its head on the row-exact (`_rows`) kernels (`INTARSIA_DFLASH_ROWS_KERNELS`, default
    /// off; [`Model::dflash_set_rows_kernels`]): a speed choice only, the drafts never reach the output.
    rows_kernels: std::sync::atomic::AtomicBool,
}

impl DflashHead {
    /// Draft layers (either kind).
    pub(super) fn n_layers(&self) -> usize {
        self.v2.as_ref().map_or(self.layers.len(), |d| d.layers.len())
    }
}

/// bf16 `[n, k]` linear weight `prefix` of the draft checkpoint.
pub(super) fn lin(src: &dyn WeightSource, b: &dyn Backend, bytes: &dyn Fn(u64), prefix: &str, n: usize, k: usize) -> Result<QTensor> {
    let q = crate::quant::load_linear(src, prefix, b, &QuantLoadOpts::default())?;
    if (q.n, q.k) != (n, k) || q.format != crate::quant::BF16 {
        return Err(Error::model(format!("draft tensor '{prefix}' is [{}, {}] {:?}, expected bf16 [{n}, {k}]", q.n, q.k, q.format)));
    }
    bytes(q.weight_bytes() as u64);
    Ok(q)
}

/// The draft's weights onto the device, checked against the target (`Model::load_shard`, `opts.dflash`).
#[allow(clippy::too_many_arguments)]
pub(super) fn load_head(
    dl: &DflashLoad,
    b: &dyn Backend,
    dims: &ModelDims,
    _recipe: &Recipe,
    main_attn: Option<&AttnBlock>,
    add_bytes: &dyn Fn(u64),
    rows: usize,
    act: DType,
) -> Result<DflashHead> {
    let (c, src) = (&dl.config, &*dl.src);
    if c.v2 {
        return load_head2(dl, b, dims, add_bytes, rows, act);
    }
    let main_attn = main_attn.ok_or_else(|| Error::model("--spec dflash needs a model with full-attention layers"))?;
    if main_attn.gated || dims.moe.is_some() || dims.gemma.is_some() || dims.intermediate == 0 {
        return Err(Error::Unsupported("--spec dflash needs a dense, ungated attention model (Qwen3)".into()));
    }
    let same = (c.hidden, c.intermediate, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab)
        == (dims.hidden, dims.intermediate, dims.n_heads, dims.n_kv_heads, dims.head_dim, dims.vocab);
    if !same {
        return Err(Error::model(format!(
            "the draft's geometry (hidden {}, ffn {}, heads {}/{}, head_dim {}, vocab {}) differs from the target's \
             (hidden {}, ffn {}, heads {}/{}, head_dim {}, vocab {})",
            c.hidden, c.intermediate, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab,
            dims.hidden, dims.intermediate, dims.n_heads, dims.n_kv_heads, dims.head_dim, dims.vocab
        )));
    }
    if c.layer_ids.is_empty() || c.layer_ids.iter().any(|&l| l >= dims.n_layers) || c.block == 0 || c.block > Model::VERIFY_MAX_ROWS - 1 {
        return Err(Error::model(format!(
            "draft tap layers {:?} / block {} do not fit a {}-layer target with {}-row verify passes",
            c.layer_ids,
            c.block,
            dims.n_layers,
            Model::VERIFY_MAX_ROWS
        )));
    }
    if c.mask_id as usize >= dims.vocab {
        return Err(Error::model(format!("draft mask token {} outside the vocabulary", c.mask_id)));
    }
    let (h, qdim, kvdim, inter) = (dims.hidden, dims.n_heads * dims.head_dim, dims.n_kv_heads * dims.head_dim, dims.intermediate);
    let norm_w = |name: &str, n: usize| -> Result<TensorRef> {
        let t = load_tensor(b, src, &resolve(src, name)?, Some(&[n]))?;
        add_bytes(t.nbytes() as u64);
        Ok(t)
    };
    let mut layers = Vec::with_capacity(c.n_layers);
    for l in 0..c.n_layers {
        let p = format!("layers.{l}");
        let attn = AttnBlock {
            pre_norm: norm_w(&format!("{p}.input_layernorm"), h)?,
            wq: lin(src, b, add_bytes, &format!("{p}.self_attn.q_proj"), qdim, h)?,
            wk: lin(src, b, add_bytes, &format!("{p}.self_attn.k_proj"), kvdim, h)?,
            wv: lin(src, b, add_bytes, &format!("{p}.self_attn.v_proj"), kvdim, h)?,
            wo: lin(src, b, add_bytes, &format!("{p}.self_attn.o_proj"), h, qdim)?,
            q_norm: Some(norm_w(&format!("{p}.self_attn.q_norm"), dims.head_dim)?),
            k_norm: Some(norm_w(&format!("{p}.self_attn.k_norm"), dims.head_dim)?),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            slot: l,
            gated: false,
            rotary_dim: main_attn.rotary_dim,
        };
        let ffn = FfnBlock {
            pre_norm: norm_w(&format!("{p}.post_attention_layernorm"), h)?,
            gate: lin(src, b, add_bytes, &format!("{p}.mlp.gate_proj"), inter, h)?,
            up: lin(src, b, add_bytes, &format!("{p}.mlp.up_proj"), inter, h)?,
            down: lin(src, b, add_bytes, &format!("{p}.mlp.down_proj"), h, inter)?,
        };
        layers.push((attn, ffn));
    }
    // fc [H, taps * H] split by columns
    let taps = c.layer_ids.len();
    let fc = load_fc(src, b, add_bytes, h, taps)?;
    let hidden_norm = norm_w("hidden_norm", h)?;
    let norm = norm_w("norm", h)?;
    let mut feat = Vec::with_capacity(taps);
    for _ in 0..taps {
        let bytes = rows * h * act.size_bytes();
        feat.push(b.alloc(bytes.max(16), MemKind::Device)?);
        add_bytes(bytes as u64);
    }
    let dcoco = if dl.dcoco {
        if c.topk == 0 || c.emb_dim == 0 || c.mlp_dim == 0 {
            return Err(Error::model("--dcoco: the draft checkpoint's dflash_config has no dcoco_* fields (not a DCOCO checkpoint)"));
        }
        if c.emb_dim != c.state_dim {
            return Err(Error::model("--dcoco: dcoco_emb_dim != dcoco_state_dim is not supported"));
        }
        Some(super::dcoco::Corrector::load(src, b, h, c.topk, c.emb_dim, c.mlp_dim, rows)?)
    } else {
        None
    };
    Ok(DflashHead {
        layers,
        fc,
        hidden_norm,
        norm,
        taps: c.layer_ids.clone(),
        feat,
        mask_id: c.mask_id,
        block: c.block,
        dcoco,
        v2: None,
        eps: dims.rms_eps,
        norm_offset: _recipe.model.norm_offset,
        script: Mutex::new(None),
        profile: Default::default(),
        full_head: None,
        rows_kernels: std::sync::atomic::AtomicBool::new(std::env::var("INTARSIA_DFLASH_ROWS_KERNELS").is_ok_and(|v| v == "1")),
    })
}

/// `fc` `[H, taps * H]` split by columns into one `[H, H]` projection per tap.
fn load_fc(src: &dyn WeightSource, b: &dyn Backend, add_bytes: &dyn Fn(u64), h: usize, taps: usize) -> Result<Vec<QTensor>> {
    let fc_name = resolve(src, "fc")?;
    let info = src.info(&fc_name).ok_or_else(|| Error::model("draft tensor 'fc' missing"))?;
    if info.dtype != DType::Bf16 || info.shape != [h, taps * h] {
        return Err(Error::model(format!("draft '{fc_name}' is {} {:?}, expected bf16 [{h}, {}]", info.dtype, info.shape, taps * h)));
    }
    let raw = src.bytes(&fc_name)?;
    let mut fc = Vec::with_capacity(taps);
    for i in 0..taps {
        let mut part = Vec::with_capacity(h * h * 2);
        for r in 0..h {
            let row = &raw[r * taps * h * 2..(r + 1) * taps * h * 2];
            part.extend_from_slice(&row[i * h * 2..(i + 1) * h * 2]);
        }
        let buf = crate::backend::upload_new(b, &part)?;
        add_bytes(part.len() as u64);
        fc.push(QTensor { format: crate::quant::BF16, n: h, k: h, parts: vec![TensorRef::new(buf, DType::Bf16, &[h, h])], act_quant: false });
    }
    Ok(fc)
}

/// A DFlash2 checkpoint (`DflashConfig::v2`): the shared pieces (taps, `fc`, `hidden_norm`, `norm`, staging) and its layers.
fn load_head2(dl: &DflashLoad, b: &dyn Backend, dims: &ModelDims, add_bytes: &dyn Fn(u64), rows: usize, act: DType) -> Result<DflashHead> {
    let (c, src) = (&dl.config, &*dl.src);
    if dl.dcoco {
        return Err(Error::Unsupported("--dcoco with a DFlash2 draft".into()));
    }
    if c.layer_ids.is_empty() || c.layer_ids.iter().any(|&l| l >= dims.n_layers) || c.block < 2 || c.block > Model::VERIFY_MAX_ROWS {
        return Err(Error::model(format!(
            "DFlash2 draft tap layers {:?} / block {} do not fit a {}-layer target with {}-row verify passes",
            c.layer_ids,
            c.block,
            dims.n_layers,
            Model::VERIFY_MAX_ROWS
        )));
    }
    if c.mask_id as usize >= dims.vocab {
        return Err(Error::model(format!("draft mask token {} outside the vocabulary", c.mask_id)));
    }
    let h = dims.hidden;
    let v2 = super::dflash2::load2(dl, b, dims, add_bytes, rows, act)?;
    let fc = load_fc(src, b, add_bytes, h, c.layer_ids.len())?;
    let norm_w = |name: &str| -> Result<TensorRef> {
        let t = load_tensor(b, src, &resolve(src, name)?, Some(&[h]))?;
        add_bytes(t.nbytes() as u64);
        Ok(t)
    };
    let (hidden_norm, norm) = (norm_w("hidden_norm")?, norm_w("norm")?);
    let mut feat = Vec::with_capacity(c.layer_ids.len());
    for _ in 0..c.layer_ids.len() {
        let bytes = rows * h * act.size_bytes();
        feat.push(b.alloc(bytes.max(16), MemKind::Device)?);
        add_bytes(bytes as u64);
    }
    Ok(DflashHead {
        layers: Vec::new(),
        fc,
        hidden_norm,
        norm,
        taps: c.layer_ids.clone(),
        feat,
        mask_id: c.mask_id,
        block: c.block,
        dcoco: None,
        v2: Some(v2),
        eps: c.rms_eps,
        norm_offset: 0.0,
        script: Mutex::new(None),
        profile: Default::default(),
        full_head: None,
        rows_kernels: std::sync::atomic::AtomicBool::new(std::env::var("INTARSIA_DFLASH_ROWS_KERNELS").is_ok_and(|v| v == "1")),
    })
}

impl Model {
    /// Whether the DFlash drafter is loaded (`LoadOptions::dflash`).
    pub fn has_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    /// Drafts per pass of the loaded DFlash drafter (its block), or 0.
    pub fn dflash_block(&self) -> usize {
        self.dflash.as_ref().map_or(0, |h| h.block)
    }

    /// The drafter state a new sequence gets (`None` without the drafter).
    pub(super) fn new_dflash_state(&self, cap: usize) -> Result<Option<Box<DflashState>>> {
        let Some(head) = &self.dflash else { return Ok(None) };
        let specs = self.dflash_slot_specs(head);
        Ok(Some(Box::new(DflashState { kv: SequenceState::new(&*self.backend, &specs, cap)?, active: true, synced: true, pending: 0, store: Vec::new() })))
    }

    /// The draft's context cache slots: one attention slot per draft layer, in the draft's own geometry for DFlash2 (activation
    /// dtype), the target's for DFlash.
    fn dflash_slot_specs(&self, head: &DflashHead) -> Vec<SlotSpec> {
        match &head.v2 {
            Some(d2) => (0..d2.layers.len())
                .map(|_| SlotSpec::AttnKv { n_kv_heads: d2.n_kv_heads, head_dim: d2.head_dim, dtype: self.act })
                .collect(),
            None => (0..head.layers.len())
                .map(|_| SlotSpec::AttnKv { n_kv_heads: self.dims.n_kv_heads, head_dim: self.dims.head_dim, dtype: self.kv })
                .collect(),
        }
    }

    /// Probes: the DFlash2 draft on the row-exact kernels or not (see `DflashHead::rows_kernels`).
    #[doc(hidden)]
    pub fn dflash_set_rows_kernels(&self, on: bool) {
        if let Some(h) = &self.dflash {
            h.rows_kernels.store(on, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Probes: time every op of the following draft passes (telemetry `ops`), as a profiled decode step does.
    #[doc(hidden)]
    pub fn dflash_profile(&self, on: bool) {
        if let Some(h) = &self.dflash {
            h.profile.store(on, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Tests: draft with `f` instead of the draft's argmax (the draft still runs, so its cache behaves as in real use).
    /// Exercises every acceptance pattern of the verify and rollback path with known drafts.
    #[doc(hidden)]
    pub fn dflash_script(&self, f: Option<super::DraftScript>) -> Result<()> {
        let head = self.dflash.as_ref().ok_or_else(|| Error::invalid("no DFlash draft"))?;
        *head.script.lock().unwrap() = f;
        Ok(())
    }

    /// After a target pass over the `t` rows just committed to `st` (positions `st.len - t .. st.len`; the taps' staging
    /// buffers hold their features): put the rows into the draft's context cache. Nothing for a request that does not draft.
    pub(super) fn dflash_after_forward(&self, st: &mut SequenceState, t: usize, profile: bool) -> Result<()> {
        let Some(head) = &self.dflash else { return Ok(()) };
        let Some(d) = st.dflash.as_mut() else { return Ok(()) };
        if !d.active || !d.synced {
            // a request that never drafts does not pay for the draft (its rows are not ingested)
            d.synced = false;
            return Ok(());
        }
        if d.kv.len + d.pending + t != st.len || st.len > d.kv.cap {
            d.synced = false;
            return Ok(());
        }
        let mut d = st.dflash.take().unwrap();
        let r = (|| -> Result<()> {
            // a batch that would overflow the store goes in first (from the store, so the staging rows stay untouched)
            if d.pending > 0 && d.pending + t > self.dflash_pend_max() {
                self.dflash_flush(head, &mut d, profile)?;
            }
            if t == 1 || d.pending > 0 {
                // single-token decode steps are stashed; any other pass behind pending rows goes in with them
                self.dflash_stash(head, &mut d, t)?;
                if t != 1 {
                    self.dflash_flush(head, &mut d, profile)?;
                }
                Ok(())
            } else {
                self.dflash_ingest(head, &mut d.kv, &head.feat, t, profile)
            }
        })();
        if r.is_err() {
            d.synced = false;
        }
        st.dflash = Some(d);
        r
    }

    /// Rows the store holds: a flush is one ingest pass, which the workspace limits to `max_tokens` rows.
    fn dflash_pend_max(&self) -> usize {
        PEND_MAX.min(self.ws.max_tokens)
    }

    /// Copy the staging rows of the pass just run (`t` rows) behind the pending rows of the store.
    fn dflash_stash(&self, head: &DflashHead, d: &mut DflashState, t: usize) -> Result<()> {
        let (h, esz) = (self.dims.hidden, self.act.size_bytes());
        if d.store.is_empty() {
            for _ in 0..head.taps.len() {
                d.store.push(self.backend.alloc((self.dflash_pend_max() * h * esz).max(16), MemKind::Device)?);
            }
        }
        for (src, dst) in head.feat.iter().zip(&d.store) {
            self.backend.copy(dst, d.pending * h * esz, src, 0, t * h * esz, self.stream)?;
        }
        d.pending += t;
        Ok(())
    }

    /// The pending rows into the draft's context cache, one batch.
    fn dflash_flush(&self, head: &DflashHead, d: &mut DflashState, profile: bool) -> Result<()> {
        let n = d.pending;
        if n == 0 {
            return Ok(());
        }
        self.dflash_ingest(head, &mut d.kv, &d.store, n, profile)?;
        d.pending = 0;
        Ok(())
    }

    /// Context rows of `t` target positions into the draft caches: `h_ctx = hidden_norm(sum_i fc_i feat_i)`, then every draft
    /// layer's k/v projection, k-norm, rope at the absolute positions and cache append (`attn_block_kv_only`).
    fn dflash_ingest(&self, head: &DflashHead, kv: &mut SequenceState, src: &[Buffer], t: usize, profile: bool) -> Result<()> {
        let (h, act, ws) = (self.dims.hidden, self.act, &self.ws);
        let v = |b: &Buffer| self.view(b, act, &[t, h]);
        let (h0, h1, blk, n) = (v(&ws.h[0]), v(&ws.h[1]), v(&ws.blk), v(&ws.n));
        let add = |a: &TensorRef, b: &TensorRef, out: &TensorRef| {
            self.exec(Op::Add, vec![a.clone(), b.clone()], vec![out.clone()], OpAttrs::default(), profile)
        };
        // sum over the taps, ping-ponging h0 / blk (an Add never writes one of its inputs)
        let feat = |i: usize| self.view(&src[i], act, &[t, h]);
        self.proj(&feat(0), &head.fc[0], &h0, profile)?;
        let (mut acc, mut other) = (h0.clone(), blk.clone());
        for i in 1..head.fc.len() {
            self.proj(&feat(i), &head.fc[i], &h1, profile)?;
            add(&acc, &h1, &other)?;
            std::mem::swap(&mut acc, &mut other);
        }
        let eps = OpAttrs { eps: head.eps, w_offset: head.norm_offset, ..Default::default() };
        self.exec(Op::RmsNorm, vec![acc, head.hidden_norm.clone()], vec![n.clone()], eps, profile)?;
        let pos0 = kv.len;
        if let Some(d2) = &head.v2 {
            self.dflash2_ingest_kv(d2, kv, &n, t, profile)?;
        }
        for (a, _) in &head.layers {
            self.attn_block_kv_only(a, kv, &n, t, pos0, false, profile)?;
        }
        kv.advance_rows(t);
        Ok(())
    }

    /// Drafts after `next` (the token the target emits at position `st.len`): one block pass of the draft over the cached
    /// context. Up to `k` tokens (the block holds `dflash_block()`), all from this one pass.
    pub fn dflash_draft(&self, st: &mut SequenceState, next: u32, k: usize) -> Result<Vec<u32>> {
        let head = self.dflash.as_ref().ok_or_else(|| Error::invalid("--spec dflash without the draft (LoadOptions::dflash)"))?;
        let d = st.dflash.as_ref().ok_or_else(|| Error::invalid("sequence without DFlash state"))?;
        if k == 0 {
            return Ok(Vec::new());
        }
        if d.active && !d.synced {
            return Ok(Vec::new()); // the context cache lost a pass: this request decodes plainly from here
        }
        if d.pending > 0 {
            let mut d = st.dflash.take().unwrap();
            let r = self.dflash_flush(head, &mut d, false);
            if r.is_err() {
                d.synced = false;
            }
            st.dflash = Some(d);
            r?;
        }
        let d = st.dflash.as_ref().unwrap();
        if !d.active || !d.synced || d.kv.len != st.len {
            return Err(Error::invalid(format!(
                "DFlash state at {} (active {}, synced {}), sequence at {}",
                d.kv.len, d.active, d.synced, st.len
            )));
        }
        let t = head.block;
        let p0 = d.kv.len;
        if p0 + t > d.kv.cap {
            return Ok(Vec::new());
        }
        let profile = head.profile.load(std::sync::atomic::Ordering::Relaxed);
        let (hd, act, ws) = (self.dims.hidden, self.act, &self.ws);
        if self.recorder.lock().unwrap().is_none() {
            let mut ids = vec![head.mask_id as i32; t];
            ids[0] = next as i32;
            if next as usize >= self.dims.vocab {
                return Err(Error::BadRequest(format!("token id {next} >= vocab {}", self.dims.vocab)));
            }
            let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
            self.backend.upload(&ws.ids, 0, &bytes, self.stream)?;
        }
        if let Some(d2) = &head.v2 {
            // DFlash2: rows 1 .. t draft the tokens at p0 + 1 .. p0 + t - 1 (`logits_start = 1`)
            let rows_k = head.rows_kernels.load(std::sync::atomic::Ordering::Relaxed);
            if rows_k {
                self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let x = self.dflash2_block(d2, &d.kv, t, profile);
            let n = self.view(&ws.n, act, &[t, hd]);
            // the draft's logits and argmax on this rank alone (the target's verify head would all-gather under a vocab-parallel head)
            let headed = x.and_then(|x| {
                let eps = OpAttrs { eps: head.eps, w_offset: head.norm_offset, ..Default::default() };
                self.exec(Op::RmsNorm, vec![x, head.norm.clone()], vec![n.clone()], eps, profile)?;
                let v = self.dims.vocab;
                let tok = self.verify_tok.as_ref().ok_or_else(|| Error::invalid("--spec dflash needs verify passes enabled (Model::enable_verify)"))?;
                let logits = self.view(&ws.logits, DType::F32, &[t - 1, v]);
                let dhead = match &head.full_head {
                    Some(h) => h,
                    None => self.head()?,
                };
                self.proj(&n.rows(1, t - 1), dhead, &logits, profile)?;
                self.exec(Op::Argmax, vec![logits], vec![self.view(tok, DType::I32, &[t - 1])], OpAttrs::default(), profile)
            });
            if rows_k {
                self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
            }
            headed?;
            let sel = d2.selector.as_ref();
            if let Some(sl) = sel {
                self.proj(&n.rows(1, t - 1), &sl.proj, &self.view(&d2.sel_out, act, &[t - 1, sl.rank]), profile)?;
            }
            let tok = self.verify_tok.as_ref().ok_or_else(|| Error::invalid("--spec dflash needs verify passes enabled (Model::enable_verify)"))?;
            if self.recorder.lock().unwrap().is_some() {
                return Ok(Vec::new());
            }
            let mut drafts: Vec<u32> = match sel {
                None => {
                    let mut b = vec![0u8; (t - 1) * 4];
                    self.backend.download(tok, 0, &mut b, self.stream)?;
                    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect()
                }
                Some(sl) => {
                    let v = self.dims.vocab;
                    let mut lb = vec![0u8; (t - 1) * v * 4];
                    self.backend.download(&ws.logits, 0, &mut lb, self.stream)?;
                    let logits: Vec<f32> = lb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let mut hb = vec![0u8; (t - 1) * sl.rank * act.size_bytes()];
                    self.backend.download(&d2.sel_out, 0, &mut hb, self.stream)?;
                    let mut hp = vec![0f32; (t - 1) * sl.rank];
                    crate::dtype::decode_to_f32(act, &hb, &mut hp);
                    sl.select(&logits, v, &hp, next)
                }
            };
            drafts.truncate(k);
            if let Some(f) = head.script.lock().unwrap().as_ref() {
                let mut seq = st.tokens.clone();
                seq.push(next);
                let s: Vec<u32> = f(&seq, drafts.len()).into_iter().filter(|&x| (x as usize) < self.dims.vocab).take(drafts.len()).collect();
                if !s.is_empty() {
                    drafts = s;
                }
            }
            return Ok(drafts);
        }
        let v = |b: &Buffer| self.view(b, act, &[t, hd]);
        let (h0, h1, n, blk) = (v(&ws.h[0]), v(&ws.h[1]), v(&ws.n), v(&ws.blk));
        let eps = OpAttrs { eps: self.dims.rms_eps, w_offset: self.recipe.model.norm_offset, ..Default::default() };
        self.exec(
            Op::Embedding,
            vec![self.view(&ws.ids, DType::I32, &[t]), self.embed.clone()],
            vec![h0.clone()],
            OpAttrs::default(),
            profile,
        )?;
        self.exec(Op::RmsNorm, vec![h0.clone(), head.layers[0].0.pre_norm.clone()], vec![n.clone()], eps.clone(), profile)?;
        self.noncausal_rows.store(true, std::sync::atomic::Ordering::Relaxed);
        let run = (|| -> Result<()> {
            for (i, (a, f)) in head.layers.iter().enumerate() {
                self.attn_block(a, &d.kv, &n, &blk, t, p0, None, false, profile)?;
                self.exec(
                    Op::RmsNormAdd,
                    vec![blk.clone(), h0.clone(), f.pre_norm.clone()],
                    vec![n.clone(), h1.clone()],
                    eps.clone(),
                    profile,
                )?;
                self.ffn_block(f, &n, &blk, t, profile)?;
                let next_w = head.layers.get(i + 1).map_or(&head.norm, |(a, _)| &a.pre_norm);
                self.exec(Op::RmsNormAdd, vec![blk.clone(), h1.clone(), next_w.clone()], vec![n.clone(), h0.clone()], eps.clone(), profile)?;
            }
            Ok(())
        })();
        self.noncausal_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        run?;
        if let Some(c) = &head.dcoco {
            // the corrector chooses the tokens among each slot's top-K candidates
            let mut drafts = self.dcoco_draft(c, &n, t, profile)?;
            drafts.truncate(k);
            return Ok(drafts);
        }
        // greedy token of every row: lm_head over the t rows, argmax on the device
        self.verify_head(&n, t, profile)?;
        let tok = self.verify_tok.as_ref().ok_or_else(|| Error::invalid("--spec dflash needs verify passes enabled (Model::enable_verify)"))?;
        if self.recorder.lock().unwrap().is_some() {
            return Ok(Vec::new()); // plan dry run: the launches are recorded, nothing to read back
        }
        let mut b = vec![0u8; t * 4];
        self.backend.download(tok, 0, &mut b, self.stream)?;
        let mut drafts: Vec<u32> = b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();
        drafts.truncate(k);
        Ok(drafts)
    }

    /// Tests: put `rows` context positions into the draft from host features (`[rows, taps * H]` f32, rounded to the activation
    /// dtype) as if the target had just processed them; advances `st` (target length and draft cache) by `rows`.
    #[doc(hidden)]
    pub fn dflash_ingest_host(&self, st: &mut SequenceState, feat: &[f32], rows: usize) -> Result<()> {
        let head = self.dflash.as_ref().ok_or_else(|| Error::invalid("no DFlash draft"))?;
        let (h, taps) = (self.dims.hidden, head.taps.len());
        if feat.len() != rows * taps * h {
            return Err(Error::invalid(format!("dflash_ingest_host: {} values for {rows} rows x {taps} taps x {h}", feat.len())));
        }
        let mut d = st.dflash.take().ok_or_else(|| Error::invalid("sequence without DFlash state"))?;
        let r = (|| -> Result<()> {
            let mut done = 0;
            while done < rows {
                let t = (rows - done).min(self.ws.max_tokens);
                for i in 0..taps {
                    let mut part = Vec::with_capacity(t * h);
                    for r in 0..t {
                        let base = (done + r) * taps * h + i * h;
                        part.extend_from_slice(&feat[base..base + h]);
                    }
                    let host = crate::tensor::HostTensor::from_f32_as(self.act, &[t, h], &part);
                    self.backend.upload(&head.feat[i], 0, &host.data, self.stream)?;
                }
                self.dflash_ingest(head, &mut d.kv, &head.feat, t, false)?;
                done += t;
            }
            Ok(())
        })();
        st.dflash = Some(d);
        r?;
        st.advance_rows(rows);
        Ok(())
    }

    /// Tests: the feature taps of the last target pass (`[rows, H]` f32 per tapped layer, in `taps` order): the target's residual
    /// stream after each tapped layer, as the draft reads it.
    #[doc(hidden)]
    pub fn dflash_taps(&self, rows: usize) -> Result<Vec<Vec<f32>>> {
        let head = self.dflash.as_ref().ok_or_else(|| Error::invalid("no DFlash draft"))?;
        let n = rows * self.dims.hidden;
        head.feat
            .iter()
            .map(|f| {
                let mut b = vec![0u8; n * self.act.size_bytes()];
                self.backend.download(f, 0, &mut b, self.stream)?;
                let mut out = vec![0f32; n];
                crate::dtype::decode_to_f32(self.act, &b, &mut out);
                Ok(out)
            })
            .collect()
    }

    /// Tests: the draft's last normed block output (`[rows, H]` f32), valid right after [`Model::dflash_draft`].
    #[doc(hidden)]
    pub fn dflash_last_hidden(&self, rows: usize) -> Result<Vec<f32>> {
        let n = rows * self.dims.hidden;
        let mut b = vec![0u8; n * self.act.size_bytes()];
        self.backend.download(&self.ws.n, 0, &mut b, self.stream)?;
        let mut out = vec![0f32; n];
        crate::dtype::decode_to_f32(self.act, &b, &mut out);
        Ok(out)
    }

    /// Plan dry-run (`Model::dry_run`, recorder installed): record the ingest and block passes so plan resolution and
    /// admission cover them. Nothing without the drafter.
    pub(super) fn dflash_dry_run(&self) -> Result<()> {
        let Some(head) = &self.dflash else { return Ok(()) };
        if self.recorder.lock().unwrap().is_none() {
            return Err(Error::invalid("dflash_dry_run outside a dry run"));
        }
        if self.verify_tok.is_none() {
            return Ok(()); // verify passes not enabled: the draft cannot run either
        }
        let specs = self.dflash_slot_specs(head);
        let mut kv = crate::plan::phantom_state(&specs, self.plan.kv_cap_tokens, 0);
        let extra = self.recorder.lock().unwrap().as_ref().is_some_and(|r| r.extra);
        if let Some(r) = self.recorder.lock().unwrap().as_mut() {
            r.extra = true;
        }
        let exact = self.exact_rows.swap(false, std::sync::atomic::Ordering::Relaxed);
        let t = head.block;
        let r = self.dflash_ingest(head, &mut kv, &head.feat, t, false).and_then(|_| {
            let mut st = crate::plan::phantom_state(&[], self.plan.kv_cap_tokens, kv.len);
            st.dflash = Some(Box::new(DflashState { kv, active: true, synced: true, pending: 0, store: Vec::new() }));
            self.dflash_draft(&mut st, 0, t).map(|_| ())
        });
        self.exact_rows.store(exact, std::sync::atomic::Ordering::Relaxed);
        if let Some(r) = self.recorder.lock().unwrap().as_mut() {
            r.extra = extra;
        }
        r
    }
}
