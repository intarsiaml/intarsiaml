//! DFlash2 block drafter (`--spec dflash` with a `DFlash2DraftModel` checkpoint), lane dfh,
//! docs/results/2026-10-26-dfh-hip-dflash.md. Checkpoint `z-lab/Qwen3.8-27B-DFlash2` (Apache-2.0); semantics from z-lab's
//! reference `dflash/model_mlx.py` (MIT, vendored unmodified in TensorFold), read, not copied.
//!
//! What differs from the DFlash of [`super::dflash`] (whose taps, context ingest scheduling, state and rollback it shares):
//!
//! ```text
//!   geometry  its own: heads / kv heads / head_dim / ffn need not match the target's; hidden and vocab must
//!   norms     standard RMSNorm (x * w; the target's recipe may be zero-centred), its own eps; rope its own theta, full rotary
//!   context   h_ctx = hidden_norm(fc(concat taps)); per layer K = rope(k_norm(k_proj h_ctx)), V = v_proj h_ctx  (as DFlash)
//!   block     rows [anchor, MASK x (block - 1)] at positions P .. P + block - 1; drafts = argmax of rows 1 .. block - 1
//!             (row r drafts the token at P + r; `logits_start = 1`)
//!   layer     dyn_a = kp_attn(n1) split in two branches;  x += conv(attn(conv(n1, dyn_a0, base_a0)), dyn_a1, base_a1)
//!             dyn_m = kp_mlp(n2) likewise;                 x += conv(mlp(conv(n2, dyn_m0, base_m0)), dyn_m1, base_m1)
//!             n1 = rms(x, ln1), n2 = rms(x, ln2); conv = `Op::DynConv` (K taps along the block rows, G channel groups)
//!   attention every block row sees every block key (non-causal) and the context keys within the sliding window: row r keeps
//!             the last `window - 1 - r` context keys (the reference's rotating cache of `window - 1` keys and its mask
//!             `query - key < window`), i.e. one `AttnDecode` per row at the block's last position with window
//!             `window - 1 - r + block`
//! ```
//!
//! The candidate selector (`candidate_selector.*`) only chooses which tokens are proposed; it is not loaded yet (drafts are the
//! rows' argmax). Drafts never change the output: every emitted token is a verify row's argmax.

use super::*;

/// One grouped dynamic causal conv (`attention_conv` / `mlp_conv`): two branches, each a `[K*G, H]` kernel projection and a
/// `[K, H]` base kernel. Branch 0 runs on the sub-block's input, branch 1 on its output.
pub(super) struct Conv2 {
    kp: [QTensor; 2],
    base: [TensorRef; 2],
}

pub(super) struct Layer2 {
    ln1: TensorRef,
    ln2: TensorRef,
    wq: QTensor,
    wk: QTensor,
    wv: QTensor,
    wo: QTensor,
    q_norm: TensorRef,
    k_norm: TensorRef,
    gate: QTensor,
    up: QTensor,
    down: QTensor,
    aconv: Conv2,
    mconv: Conv2,
}

/// The draft's own scratch, so its geometry never has to fit the target's (per-rank) workspace.
struct Bufs2 {
    /// block rows `[block, ...]`
    x: [Buffer; 2],
    n: Buffer,
    c: Buffer,
    o: Buffer,
    q: [Buffer; 2],
    k: [Buffer; 2],
    v: Buffer,
    ao: Buffer,
    g: Buffer,
    u: Buffer,
    a: Buffer,
    dyn_: [Buffer; 2],
    /// context ingest rows `[max rows, nkv * hd]`
    ik: [Buffer; 2],
    iv: Buffer,
}

/// `candidate_selector.*` (DFlash2's drafts): of each block row's `top_k` highest logits, take the one whose successor code best
/// fits the previous draft's predecessor code under the row's projected hidden state, walking from the anchor (greedy:
/// `argmax(logit + sum_r pred[prev][r] * proj(h)[r] * succ[cand][r])`). Codebooks on the host (bf16 bits, `[V, rank]`).
pub(super) struct Selector {
    pub(super) proj: QTensor,
    pub(super) rank: usize,
    top_k: usize,
    pred: Vec<u16>,
    succ: Vec<u16>,
}

fn bf(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

impl Selector {
    /// Drafts for `rows = logits.len() / v` block rows: `logits` `[rows, v]`, `hp` `[rows, rank]`, `anchor` the token before row 0.
    pub(super) fn select(&self, logits: &[f32], v: usize, hp: &[f32], anchor: u32) -> Vec<u32> {
        let rows = logits.len() / v;
        let mut prev = anchor as usize;
        let mut out = Vec::with_capacity(rows);
        let mut top: Vec<(f32, usize)> = Vec::with_capacity(self.top_k + 1);
        for p in 0..rows {
            let row = &logits[p * v..(p + 1) * v];
            top.clear();
            let mut floor = f32::NEG_INFINITY;
            for (i, &x) in row.iter().enumerate() {
                if top.len() < self.top_k || x > floor {
                    let at = top.partition_point(|e| e.0 >= x);
                    top.insert(at, (x, i));
                    if top.len() > self.top_k {
                        top.pop();
                    }
                    if top.len() == self.top_k {
                        floor = top[self.top_k - 1].0;
                    }
                }
            }
            let (r, h) = (self.rank, &hp[p * self.rank..(p + 1) * self.rank]);
            let pc = &self.pred[prev * r..(prev + 1) * r];
            let mut best = (f32::NEG_INFINITY, top[0].1);
            for &(u, c) in &top {
                let sc = &self.succ[c * r..(c + 1) * r];
                let mut e = 0f32;
                for j in 0..r {
                    e += bf(pc[j]) * h[j] * bf(sc[j]);
                }
                if u + e > best.0 {
                    best = (u + e, c);
                }
            }
            prev = best.1;
            out.push(prev as u32);
        }
        out
    }
}

pub(super) struct Draft2 {
    pub(super) layers: Vec<Layer2>,
    pub(super) n_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) head_dim: usize,
    inter: usize,
    eps: f32,
    theta: f32,
    /// `sliding_window` (0 = full context)
    window: usize,
    conv_k: usize,
    conv_g: usize,
    bufs: Bufs2,
    pub(super) selector: Option<Selector>,
    /// `[block, rank]` the selector's projected hidden rows
    pub(super) sel_out: Buffer,
}

/// The `[2*K*G, H]` kernel projection split into its two branches (rows `0..K*G` and `K*G..2*K*G`), each its own buffer.
fn split_rows(src: &dyn WeightSource, b: &dyn Backend, bytes: &dyn Fn(u64), name: &str, n: usize, k: usize) -> Result<[QTensor; 2]> {
    let info = src.info(name).ok_or_else(|| Error::model(format!("draft tensor '{name}' missing")))?;
    if info.dtype != DType::Bf16 || info.shape != [2 * n, k] {
        return Err(Error::model(format!("draft '{name}' is {} {:?}, expected bf16 [{}, {k}]", info.dtype, info.shape, 2 * n)));
    }
    let raw = src.bytes(name)?;
    let half = n * k * 2;
    let mk = |part: &[u8]| -> Result<QTensor> {
        let buf = crate::backend::upload_new(b, part)?;
        bytes(part.len() as u64);
        Ok(QTensor { format: crate::quant::BF16, n, k, parts: vec![TensorRef::new(buf, DType::Bf16, &[n, k])], act_quant: false })
    };
    Ok([mk(&raw[..half])?, mk(&raw[half..2 * half])?])
}

/// `[2, K, H]` base kernel as two `[K, H]` tensors in the activation dtype (`Op::DynConv` takes all three inputs in one dtype).
fn base2(src: &dyn WeightSource, b: &dyn Backend, bytes: &dyn Fn(u64), name: &str, kk: usize, h: usize, act: DType) -> Result<[TensorRef; 2]> {
    let info = src.info(name).ok_or_else(|| Error::model(format!("draft tensor '{name}' missing")))?;
    if info.dtype != DType::Bf16 || info.shape != [2, kk, h] {
        return Err(Error::model(format!("draft '{name}' is {} {:?}, expected bf16 [2, {kk}, {h}]", info.dtype, info.shape)));
    }
    let host = crate::tensor::HostTensor::new(DType::Bf16, vec![2 * kk, h], src.bytes(name)?.to_vec())?;
    let vals = host.to_f32();
    let mk = |part: &[f32]| -> Result<TensorRef> {
        let t = crate::tensor::HostTensor::from_f32_as(act, &[kk, h], part);
        let buf = crate::backend::upload_new(b, &t.data)?;
        bytes(t.data.len() as u64);
        Ok(TensorRef::new(buf, act, &[kk, h]))
    };
    Ok([mk(&vals[..kk * h])?, mk(&vals[kk * h..])?])
}

/// The DFlash2 draft's layers onto the device (`dflash::load_head`, a `DFlash2DraftModel` checkpoint). `rows`: the most context
/// rows one ingest takes (the target's chunk).
pub(super) fn load2(dl: &super::dflash::DflashLoad, b: &dyn Backend, dims: &ModelDims, add_bytes: &dyn Fn(u64), rows: usize, act: DType) -> Result<Draft2> {
    let (c, src) = (&dl.config, &*dl.src);
    if c.hidden != dims.hidden || c.vocab != dims.vocab {
        return Err(Error::model(format!(
            "the DFlash2 draft's hidden {} / vocab {} differ from the target's {} / {}",
            c.hidden, c.vocab, dims.hidden, dims.vocab
        )));
    }
    if c.conv_kernel == 0 || c.conv_group == 0 || c.hidden % c.conv_group != 0 || c.n_kv_heads == 0 || c.n_heads % c.n_kv_heads != 0 {
        return Err(Error::model(format!(
            "DFlash2 draft: conv kernel {} / group {} / heads {}/{} do not fit hidden {}",
            c.conv_kernel, c.conv_group, c.n_heads, c.n_kv_heads, c.hidden
        )));
    }
    if act != DType::Bf16 && act != DType::F32 {
        return Err(Error::Unsupported(format!("the DFlash2 draft with {act} activations")));
    }
    let (h, hd, nh, nkv, inter) = (c.hidden, c.head_dim, c.n_heads, c.n_kv_heads, c.intermediate);
    let (qdim, kvdim) = (nh * hd, nkv * hd);
    let (kk, g) = (c.conv_kernel, c.hidden / c.conv_group);
    let norm_w = |name: &str, n: usize| -> Result<TensorRef> {
        let t = load_tensor(b, src, &resolve(src, name)?, Some(&[n]))?;
        add_bytes(t.nbytes() as u64);
        Ok(t)
    };
    let lin = |p: &str, n: usize, k: usize| super::dflash::lin(src, b, add_bytes, p, n, k);
    let mut layers = Vec::with_capacity(c.n_layers);
    for l in 0..c.n_layers {
        let p = format!("layers.{l}");
        let conv = |name: &str| -> Result<Conv2> {
            Ok(Conv2 {
                kp: split_rows(src, b, add_bytes, &resolve(src, &format!("{p}.{name}.kernel_projection"))?, kk * g, h)?,
                base: base2(src, b, add_bytes, &format!("{p}.{name}.base_kernel"), kk, h, act)?,
            })
        };
        layers.push(Layer2 {
            ln1: norm_w(&format!("{p}.input_layernorm"), h)?,
            ln2: norm_w(&format!("{p}.post_attention_layernorm"), h)?,
            wq: lin(&format!("{p}.self_attn.q_proj"), qdim, h)?,
            wk: lin(&format!("{p}.self_attn.k_proj"), kvdim, h)?,
            wv: lin(&format!("{p}.self_attn.v_proj"), kvdim, h)?,
            wo: lin(&format!("{p}.self_attn.o_proj"), h, qdim)?,
            q_norm: norm_w(&format!("{p}.self_attn.q_norm"), hd)?,
            k_norm: norm_w(&format!("{p}.self_attn.k_norm"), hd)?,
            gate: lin(&format!("{p}.mlp.gate_proj"), inter, h)?,
            up: lin(&format!("{p}.mlp.up_proj"), inter, h)?,
            down: lin(&format!("{p}.mlp.down_proj"), h, inter)?,
            aconv: conv("attention_conv")?,
            mconv: conv("mlp_conv")?,
        });
    }
    let t = c.block;
    let es = act.size_bytes();
    let alloc = |elems: usize| -> Result<Buffer> {
        add_bytes((elems * es) as u64);
        b.alloc((elems * es).max(16), MemKind::Device)
    };
    let bufs = Bufs2 {
        x: [alloc(t * h)?, alloc(t * h)?],
        n: alloc(t * h)?,
        c: alloc(t * h)?,
        o: alloc(t * h)?,
        q: [alloc(t * qdim)?, alloc(t * qdim)?],
        k: [alloc(t * kvdim)?, alloc(t * kvdim)?],
        v: alloc(t * kvdim)?,
        ao: alloc(t * qdim)?,
        g: alloc(t * inter)?,
        u: alloc(t * inter)?,
        a: alloc(t * inter)?,
        dyn_: [alloc(t * kk * g)?, alloc(t * kk * g)?],
        ik: [alloc(rows * kvdim)?, alloc(rows * kvdim)?],
        iv: alloc(rows * kvdim)?,
    };
    let selector = if c.selector_rank > 0 && c.selector_top_k > 0 {
        let codebook = |name: &str| -> Result<Vec<u16>> {
            let info = src.info(name).ok_or_else(|| Error::model(format!("draft tensor '{name}' missing")))?;
            if info.dtype != DType::Bf16 || info.shape != [c.vocab, c.selector_rank] {
                return Err(Error::model(format!("draft '{name}' is {} {:?}, expected bf16 [{}, {}]", info.dtype, info.shape, c.vocab, c.selector_rank)));
            }
            Ok(src.bytes(name)?.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect())
        };
        Some(Selector {
            proj: super::dflash::lin(src, b, add_bytes, "candidate_selector.hidden_projection", c.selector_rank, h)?,
            rank: c.selector_rank,
            top_k: c.selector_top_k.min(c.vocab),
            pred: codebook("candidate_selector.predecessor_codebook")?,
            succ: codebook("candidate_selector.successor_codebook")?,
        })
    } else {
        None
    };
    let sel_out = alloc(t * c.selector_rank.max(1))?;
    Ok(Draft2 {
        selector,
        sel_out,
        layers,
        n_heads: nh,
        n_kv_heads: nkv,
        head_dim: hd,
        inter,
        eps: c.rms_eps,
        theta: c.rope_theta,
        window: c.sliding_window,
        conv_k: kk,
        conv_g: g,
        bufs,
    })
}

impl Model {
    /// The draft's norms: standard RMSNorm (`w_offset` 0) with the draft's eps, whatever the target's recipe says.
    fn d2_norm(&self, d2: &Draft2, x: &TensorRef, w: &TensorRef, out: &TensorRef, profile: bool) -> Result<()> {
        self.exec(Op::RmsNorm, vec![x.clone(), w.clone()], vec![out.clone()], OpAttrs { eps: d2.eps, w_offset: 0.0, ..Default::default() }, profile)
    }

    /// Context ingest of `t` rows whose `h_ctx` (already normed) is `n` (`[t, H]`): every draft layer's K/V at positions
    /// `kv.len ..`, appended to its cache slot. Does not advance `kv`.
    pub(super) fn dflash2_ingest_kv(&self, d2: &Draft2, kv: &SequenceState, n: &TensorRef, t: usize, profile: bool) -> Result<()> {
        let (nkv, hd, act) = (d2.n_kv_heads, d2.head_dim, self.act);
        let pos0 = kv.len;
        let k0 = self.view(&d2.bufs.ik[0], act, &[t, nkv, hd]);
        let k1 = self.view(&d2.bufs.ik[1], act, &[t, nkv, hd]);
        let v = self.view(&d2.bufs.iv, act, &[t, nkv, hd]);
        for (li, l) in d2.layers.iter().enumerate() {
            let slot = Self::attn_slot_at(kv, li)?;
            self.proj(n, &l.wk, &k0.reshaped(&[t, nkv * hd]), profile)?;
            self.proj(n, &l.wv, &v.reshaped(&[t, nkv * hd]), profile)?;
            self.d2_qk(d2, &k0, &l.k_norm, &k1, &k0, t, pos0, profile)?;
            self.d2_append(slot, &k0, &v, pos0, nkv, hd, profile)?;
        }
        Ok(())
    }

    fn attn_slot_at(st: &SequenceState, slot: usize) -> Result<&crate::state::AttnKvState> {
        match st.slots.get(slot) {
            Some(StateSlot::AttnKv(s)) => Ok(s),
            _ => Err(Error::invalid("DFlash2 draft layer without its attention cache slot")),
        }
    }

    /// q/k norm then rope at positions `pos0 ..` (full rotary, the draft's theta): `x` -> `tmp` (norm) -> `out` (rope).
    #[allow(clippy::too_many_arguments)]
    fn d2_qk(&self, d2: &Draft2, x: &TensorRef, w: &TensorRef, tmp: &TensorRef, out: &TensorRef, _t: usize, pos0: usize, profile: bool) -> Result<()> {
        self.exec(Op::QkNorm, vec![x.clone(), w.clone()], vec![tmp.clone()], OpAttrs { eps: d2.eps, w_offset: 0.0, ..Default::default() }, profile)?;
        let rope = OpAttrs { pos0, theta: d2.theta, head_dim: d2.head_dim, rotary_dim: 0, ..Default::default() };
        self.exec(Op::Rope, vec![tmp.clone()], vec![out.clone()], rope, profile)
    }

    #[allow(clippy::too_many_arguments)]
    fn d2_append(&self, slot: &crate::state::AttnKvState, k: &TensorRef, v: &TensorRef, pos0: usize, nkv: usize, hd: usize, profile: bool) -> Result<()> {
        let shape = [nkv, slot.cap, hd];
        let kc = self.view(&slot.k, slot.dtype, &shape);
        let vc = self.view(&slot.v, slot.dtype, &shape);
        let append = OpAttrs { pos0, kv_cap: slot.cap, n_kv_heads: nkv, head_dim: hd, ..Default::default() };
        self.exec(Op::KvAppend, vec![k.clone()], vec![kc], append.clone(), profile)?;
        self.exec(Op::KvAppend, vec![v.clone()], vec![vc], append, profile)
    }

    /// `conv(x, dyn, base)` of one branch.
    fn d2_conv(&self, d2: &Draft2, x: &TensorRef, dyn_: &TensorRef, base: &TensorRef, out: &TensorRef, profile: bool) -> Result<()> {
        let _ = d2;
        self.exec(Op::DynConv, vec![x.clone(), dyn_.clone(), base.clone()], vec![out.clone()], OpAttrs::default(), profile)
    }

    /// One block pass of the DFlash2 draft over `kv` (context of `kv.len` rows) with the block's ids already in `ws.ids`
    /// (`t` rows). Leaves the final normed rows in the draft's `n` buffer and returns that view (`[t, H]`).
    pub(super) fn dflash2_block(&self, d2: &Draft2, kv: &SequenceState, t: usize, profile: bool) -> Result<TensorRef> {
        let (h, act, ws) = (self.dims.hidden, self.act, &self.ws);
        let (nh, nkv, hd, inter) = (d2.n_heads, d2.n_kv_heads, d2.head_dim, d2.inter);
        let bf = &d2.bufs;
        let rows = |b: &Buffer, w: usize| self.view(b, act, &[t, w]);
        let (mut x, mut xn) = (rows(&bf.x[0], h), rows(&bf.x[1], h));
        let (n, cbuf, o) = (rows(&bf.n, h), rows(&bf.c, h), rows(&bf.o, h));
        let kg = d2.conv_k * d2.conv_g;
        let dyn0 = rows(&bf.dyn_[0], kg);
        let dyn1 = rows(&bf.dyn_[1], kg);
        let q3 = |i: usize| self.view(&bf.q[i], act, &[t, nh, hd]);
        let k3 = |i: usize| self.view(&bf.k[i], act, &[t, nkv, hd]);
        let v3 = self.view(&bf.v, act, &[t, nkv, hd]);
        let ao = self.view(&bf.ao, act, &[t, nh, hd]);
        let (g, u, a) = (rows(&bf.g, inter), rows(&bf.u, inter), rows(&bf.a, inter));
        self.exec(Op::Embedding, vec![self.view(&ws.ids, DType::I32, &[t]), self.embed.clone()], vec![x.clone()], OpAttrs::default(), profile)?;
        let p0 = kv.len;
        let add = |a: &TensorRef, b: &TensorRef, out: &TensorRef| self.exec(Op::Add, vec![a.clone(), b.clone()], vec![out.clone()], OpAttrs::default(), profile);
        for (li, l) in d2.layers.iter().enumerate() {
            let slot = Self::attn_slot_at(kv, li)?;
            // attention sub-block
            self.d2_norm(d2, &x, &l.ln1, &n, profile)?;
            self.proj(&n, &l.aconv.kp[0], &dyn0, profile)?;
            self.proj(&n, &l.aconv.kp[1], &dyn1, profile)?;
            self.d2_conv(d2, &n, &dyn0, &l.aconv.base[0], &cbuf, profile)?;
            self.proj(&cbuf, &l.wq, &q3(0).reshaped(&[t, nh * hd]), profile)?;
            self.proj(&cbuf, &l.wk, &k3(0).reshaped(&[t, nkv * hd]), profile)?;
            self.proj(&cbuf, &l.wv, &v3.reshaped(&[t, nkv * hd]), profile)?;
            self.d2_qk(d2, &q3(0), &l.q_norm, &q3(1), &q3(0), t, p0, profile)?;
            self.d2_qk(d2, &k3(0), &l.k_norm, &k3(1), &k3(0), t, p0, profile)?;
            // the block's own K/V go beyond the cache length (scratch, overwritten by the next ingest)
            self.d2_append(slot, &k3(0), &v3, p0, nkv, hd, profile)?;
            let shape = [nkv, slot.cap, hd];
            let kc = self.view(&slot.k, slot.dtype, &shape);
            let vc = self.view(&slot.v, slot.dtype, &shape);
            let last = p0 + t - 1;
            for r in 0..t {
                // row r: every block key, and the context keys within its window (the last `window - 1 - r` of them)
                let win = if d2.window == 0 { 0 } else { d2.window + t - 1 - r };
                let win = if win == 0 || last + 1 <= win { 0 } else { win };
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
                self.exec(Op::AttnDecode, vec![q3(0).rows(r, 1), kc.clone(), vc.clone()], vec![ao.rows(r, 1)], attrs, profile)?;
            }
            self.proj(&ao.reshaped(&[t, nh * hd]), &l.wo, &o, profile)?;
            self.d2_conv(d2, &o, &dyn1, &l.aconv.base[1], &cbuf, profile)?;
            add(&x, &cbuf, &xn)?;
            std::mem::swap(&mut x, &mut xn);
            // feed-forward sub-block
            self.d2_norm(d2, &x, &l.ln2, &n, profile)?;
            self.proj(&n, &l.mconv.kp[0], &dyn0, profile)?;
            self.proj(&n, &l.mconv.kp[1], &dyn1, profile)?;
            self.d2_conv(d2, &n, &dyn0, &l.mconv.base[0], &cbuf, profile)?;
            self.proj(&cbuf, &l.gate, &g, profile)?;
            self.proj(&cbuf, &l.up, &u, profile)?;
            self.exec(Op::SiluGlu, vec![g.clone(), u.clone()], vec![a.clone()], OpAttrs::default(), profile)?;
            self.proj(&a, &l.down, &o, profile)?;
            self.d2_conv(d2, &o, &dyn1, &l.mconv.base[1], &cbuf, profile)?;
            add(&x, &cbuf, &xn)?;
            std::mem::swap(&mut x, &mut xn);
        }
        Ok(x)
    }
}
