//! The DeepSeek-V4.1 model graph: the op sequence of one forward chunk (`t` rows at positions `pos0 ..`), built from the
//! config and the checkpoint's tensor shapes, plus the sequence-state layout it reads and writes.
//!
//! The graph is data: every [`Step`] names its op, attributes and the buffers it reads and writes ([`Buf`]: checkpoint
//! tensors, workspace activations, state-slot rows, engine constants, host-filled inputs) with dtype and shape. `plan
//! show` records it against a backend's registry (each step is a valid `OpCall` on shape-only tensors); the executing
//! lanes bind the same steps to real buffers. Mirrors `model.py` `Transformer.forward` / `Block.forward` /
//! `Attention.forward` / `MoE.forward` (official inference code, MIT, DeepSeek); file:line per step kind in
//! docs/design/dsv41-ops.md.
//!
//! Multi-stream residual: the residual is `hc_mult` (4) streams, `[t, 4, dim]` in the workspace ("x0" / "x1", ping-pong); every sub-block
//! collapses it with `HcPre` and expands back with `HcPost`, the mixes come from `HcMixes` of the stream before the
//! sub-block, and each sub-block's collapse uses the mix the previous sub-block computed (the first one a constant
//! one-hot, the final collapse before the head the last FFN's).

use super::config::{Dsv41Config, Dsv41Quant};
use super::ops::{compressed_rows, n_blocks, QdqFmt, QdqScale};
use crate::backend::{Buffer, BufferImpl, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Op, OpAttrs, OpCall};
use crate::state::{RingKind, SharedKind, SlotSpec};

/// Tensor shapes of a checkpoint (safetensors headers; no data).
pub trait WeightShapes {
    fn get(&self, name: &str) -> Option<(DType, Vec<usize>)>;
}

impl WeightShapes for std::collections::BTreeMap<String, (DType, Vec<usize>)> {
    fn get(&self, name: &str) -> Option<(DType, Vec<usize>)> {
        std::collections::BTreeMap::get(self, name).cloned()
    }
}

/// Where a step's tensor lives.
#[derive(Clone, Debug, PartialEq)]
pub enum Buf {
    /// A checkpoint tensor, or an engine part made from checkpoint tensors at load: `{p}.codebook` (EXL3 codebook marker
    /// `{p}.mul1` / `{p}.mcg` as `i32 [1]`), `{layer}.ffn.experts.*.w1.trellis` (the routed experts stacked, `[E, ..]`).
    Weight(String),
    /// A model workspace activation.
    Ws(String),
    /// Rows `row0 .. row0 + shape[0]` of plane `plane` of state slot `slot` (`Dsv41Graph::slots`).
    Slot { slot: usize, plane: usize, row0: usize },
    /// An engine constant (rope frequency table, the one-hot first pre-mix).
    Const(String),
    /// Filled by the host before the pass (token ids, Engram rows fetched from the `EngramTier`).
    Host(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    pub buf: Buf,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub op: Op,
    /// What the step computes, e.g. `attn.wq_a`.
    pub what: String,
    pub inputs: Vec<Arg>,
    pub outputs: Vec<Arg>,
    pub attrs: OpAttrs,
}

/// Embedding, one layer, or the head.
#[derive(Clone, Debug)]
pub struct Stage {
    pub name: String,
    pub layer: Option<usize>,
    /// Block summary for `plan show` (e.g. `["engram", "hc_attn:window+cmp(2)/idx", "hc_moe"]`).
    pub blocks: Vec<String>,
    pub steps: Vec<Step>,
}

/// A DSpark / MTP block found in the checkpoint: parsed, not executed.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpBlock {
    pub index: usize,
    pub prefix: String,
    pub experts: usize,
    pub top_k: usize,
    pub tensors: usize,
}

#[derive(Clone, Debug)]
pub struct Dsv41Graph {
    pub stages: Vec<Stage>,
    pub slots: Vec<SlotSpec>,
    pub residual_streams: usize,
    pub mtp: Vec<MtpBlock>,
    /// Token capacity the shared slots are sized for.
    pub cap: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct GraphOpts {
    /// Rows of the chunk.
    pub t: usize,
    /// Absolute position of the first row.
    pub pos0: usize,
    /// Activation dtype.
    pub act: DType,
    /// Token capacity (sizes the shared compressed-KV / index-key slots).
    pub cap: usize,
}

/// A linear layer as the checkpoint stores it.
enum Lin {
    Exl3 { p: String, k: usize, n: usize, trellis: Vec<usize> },
    Dense { p: String, dtype: DType, k: usize, n: usize },
    Fp8 { p: String, k: usize, n: usize, scale: Vec<usize> },
}

struct Builder<'a> {
    c: &'a Dsv41Config,
    w: &'a dyn WeightShapes,
    o: GraphOpts,
    steps: Vec<Step>,
}

fn arg(buf: Buf, dtype: DType, shape: &[usize]) -> Arg {
    Arg { buf, dtype, shape: shape.to_vec() }
}

fn ws(name: &str, dtype: DType, shape: &[usize]) -> Arg {
    arg(Buf::Ws(name.into()), dtype, shape)
}

impl Builder<'_> {
    fn weight(&self, name: &str) -> Result<Arg> {
        let (dt, shape) = self.w.get(name).ok_or_else(|| Error::model(format!("deepseek_v41: checkpoint has no tensor {name}")))?;
        Ok(arg(Buf::Weight(name.into()), dt, &shape))
    }

    fn has(&self, name: &str) -> bool {
        self.w.get(name).is_some()
    }

    fn lin(&self, p: &str) -> Result<Lin> {
        if let Some((_, s)) = self.w.get(&format!("{p}.trellis")) {
            if s.len() != 3 {
                return Err(Error::model(format!("deepseek_v41: {p}.trellis has shape {s:?}")));
            }
            return Ok(Lin::Exl3 { p: p.into(), k: 16 * s[0], n: 16 * s[1], trellis: s });
        }
        let (dt, s) = self.w.get(&format!("{p}.weight")).ok_or_else(|| Error::model(format!("deepseek_v41: checkpoint has no linear {p} (.trellis or .weight)")))?;
        if s.len() != 2 {
            return Err(Error::model(format!("deepseek_v41: {p}.weight has shape {s:?}")));
        }
        if dt == DType::Fp8E4m3 {
            let (_, scale) = self.w.get(&format!("{p}.scale")).ok_or_else(|| Error::model(format!("deepseek_v41: fp8 linear {p} has no .scale")))?;
            return Ok(Lin::Fp8 { p: p.into(), k: s[1], n: s[0], scale });
        }
        Ok(Lin::Dense { p: p.into(), dtype: dt, k: s[1], n: s[0] })
    }

    fn push(&mut self, op: Op, what: impl Into<String>, inputs: Vec<Arg>, outputs: Vec<Arg>, attrs: OpAttrs) {
        self.steps.push(Step { op, what: what.into(), inputs, outputs, attrs });
    }

    /// `out = x @ W^T` for the linear `p` (EXL3, fp8 block-scaled or dense), output `out` of dtype `odt`.
    fn matmul(&mut self, x: Arg, p: &str, out: Buf, odt: DType) -> Result<Arg> {
        let m = x.shape[0];
        let lin = self.lin(p)?;
        let (op, inputs, n, mut attrs) = match lin {
            Lin::Exl3 { p, k, n, trellis } => {
                if x.shape[1] != k {
                    return Err(Error::model(format!("deepseek_v41: {p} takes K={k}, input has {:?}", x.shape)));
                }
                let w = vec![
                    x,
                    arg(Buf::Weight(format!("{p}.trellis")), DType::I16, &trellis),
                    arg(Buf::Weight(format!("{p}.suh")), DType::F16, &[k]),
                    arg(Buf::Weight(format!("{p}.svh")), DType::F16, &[n]),
                    arg(Buf::Weight(format!("{p}.codebook")), DType::I32, &[1]),
                ];
                (Op::MatmulExl3, w, n, OpAttrs::default())
            }
            Lin::Fp8 { p, k, n, scale } => {
                if x.shape[1] != k {
                    return Err(Error::model(format!("deepseek_v41: {p} takes K={k}, input has {:?}", x.shape)));
                }
                let mut a = OpAttrs::default();
                a.dsv.block = match &self.c.quant {
                    Dsv41Quant::Fp8 { block, .. } => *block,
                    _ => 32,
                };
                let w = vec![x, self.weight(&format!("{p}.weight"))?, arg(Buf::Weight(format!("{p}.scale")), DType::U8, &scale)];
                (Op::MatmulFp8Block, w, n, a)
            }
            Lin::Dense { p, dtype, n, k } => {
                let w = vec![x, arg(Buf::Weight(format!("{p}.weight")), dtype, &[n, k])];
                (if odt == DType::F32 { Op::Logits } else { Op::Matmul }, w, n, OpAttrs::default())
            }
        };
        attrs.eps = self.c.norm_eps;
        let o = arg(out, odt, &[m, n]);
        self.push(op, p.to_string(), inputs, vec![o.clone()], attrs);
        Ok(o)
    }

    fn rmsnorm(&mut self, x: Arg, w: &str, out: &str) -> Result<Arg> {
        let wt = self.weight(w)?;
        let o = Arg { buf: Buf::Ws(out.into()), ..x.clone() };
        self.push(Op::RmsNorm, w.to_string(), vec![x, wt], vec![o.clone()], OpAttrs { eps: self.c.norm_eps, ..Default::default() });
        Ok(o)
    }

    fn rope(&mut self, x: Arg, layer: usize, pos0: usize, stride: usize, inverse: bool, out: Buf, what: &str) -> Arg {
        let mut a = OpAttrs { pos0, ..Default::default() };
        (a.dsv.pos_stride, a.dsv.inverse) = (stride, inverse);
        let table = if self.c.ratio(layer) > 0 { "rope.compressed_yarn" } else { "rope.base" };
        let f = arg(Buf::Const(table.into()), DType::F32, &[self.c.rope_head_dim / 2]);
        let o = Arg { buf: out, ..x.clone() };
        self.push(Op::RopeTail, what.to_string(), vec![x, f], vec![o.clone()], a);
        o
    }

    fn qdq(&mut self, x: Arg, fmt: QdqFmt, block: usize, scale: QdqScale, out: Buf, what: &str) -> Arg {
        let mut a = OpAttrs::default();
        (a.dsv.qdq_fmt, a.dsv.qdq_block, a.dsv.qdq_scale) = (fmt, block, scale);
        let o = Arg { buf: out, ..x.clone() };
        self.push(Op::ActQdq, what.to_string(), vec![x], vec![o.clone()], a);
        o
    }

    fn hc_mixes(&mut self, x: &Arg, p: &str, out: &str) -> Result<[Arg; 3]> {
        let (t, s) = (self.o.t, self.c.hc_mult);
        let inputs = vec![x.clone(), self.weight(&format!("{p}_fn"))?, self.weight(&format!("{p}_scale"))?, self.weight(&format!("{p}_base"))?];
        let outs = [ws(&format!("{out}.pre"), DType::F32, &[t, s]), ws(&format!("{out}.post"), DType::F32, &[t, s]), ws(&format!("{out}.comb"), DType::F32, &[t, s, s])];
        let mut a = OpAttrs { eps: self.c.norm_eps, ..Default::default() };
        (a.dsv.iters, a.dsv.hc_eps) = (self.c.hc_sinkhorn_iters, self.c.hc_eps);
        self.push(Op::HcMixes, p.to_string(), inputs, outs.to_vec(), a);
        Ok(outs)
    }
}

/// The other residual buffer (`x0` <-> `x1`).
fn flip(x: &Arg) -> Arg {
    let other = if x.buf == Buf::Ws("x0".into()) { "x1" } else { "x0" };
    Arg { buf: Buf::Ws(other.into()), ..x.clone() }
}

/// Fixed slot indices of a layer.
#[derive(Clone, Copy, Debug, Default)]
struct LayerSlots {
    window: usize,
    tail: Option<usize>,
    cmp: Option<usize>,
    index_k: Option<usize>,
}

/// The state layout: one sliding-window ring per layer; per KV source the compressor tail ring (ratio > 1), the shared
/// compressed KV and the shared index keys, each read by every layer whose source it is.
fn slots(c: &Dsv41Config, act: DType) -> (Vec<SlotSpec>, Vec<LayerSlots>) {
    let mut specs = Vec::new();
    let mut ls = vec![LayerSlots::default(); c.n_layers];
    for (l, s) in ls.iter_mut().enumerate() {
        s.window = specs.len();
        specs.push(SlotSpec::Ring { rows: c.window, width: c.head_dim, planes: 1, dtype: act, margin: 0, kind: RingKind::WindowKv });
        let _ = l;
    }
    for &src in &c.kv_source_layers {
        let r = c.ratio(src);
        let readers: Vec<usize> = (0..c.n_layers).filter(|&l| c.kv_source(l) == Some(src)).collect();
        if r > 1 {
            ls[src].tail = Some(specs.len());
            specs.push(SlotSpec::Ring { rows: r, width: c.head_dim, planes: 2, dtype: DType::F32, margin: 0, kind: RingKind::CompressTail });
        }
        let cmp = specs.len();
        specs.push(SlotSpec::Shared { ratio: r, width: c.head_dim, dtype: act, writer: src, readers: readers.clone(), kind: SharedKind::CompressedKv });
        let idx = specs.len();
        let idx_readers: Vec<usize> = readers.iter().copied().filter(|&l| c.index_source_layers.contains(&l)).collect();
        specs.push(SlotSpec::Shared { ratio: r, width: c.index_head_dim, dtype: act, writer: src, readers: idx_readers, kind: SharedKind::IndexK });
        for &l in &readers {
            ls[l].cmp = Some(cmp);
            ls[l].index_k = Some(idx);
        }
    }
    (specs, ls)
}

/// Build the graph of one chunk.
pub fn build(c: &Dsv41Config, w: &dyn WeightShapes, o: GraphOpts) -> Result<Dsv41Graph> {
    let (t, d, s, act) = (o.t, c.dim, c.hc_mult, o.act);
    if t == 0 {
        return Err(Error::invalid("deepseek_v41 graph: zero rows"));
    }
    let (slot_specs, lslots) = slots(c, act);
    let mut stages = Vec::new();
    let mut b = Builder { c, w, o, steps: Vec::new() };

    // embedding, expanded to hc_mult streams (Transformer.forward)
    let ids = arg(Buf::Host("token_ids".into()), DType::I32, &[t]);
    let emb = b.weight("embed.weight")?;
    let h = ws("h", act, &[t, d]);
    b.push(Op::Embedding, "embed", vec![ids, emb], vec![h.clone()], OpAttrs::default());
    // the residual ping-pongs between two buffers: HcPost mixes across streams and EngramMerge reads every stream, so no
    // step's output aliases its input (outputs never alias inputs in this graph, except the state rings marked in place)
    let mut x = ws("x0", act, &[t, s, d]);
    b.push(Op::HcExpand, "hc_expand", vec![h], vec![x.clone()], OpAttrs::default());
    stages.push(Stage { name: "embed".into(), layer: None, blocks: vec!["embed".into(), format!("hc_expand x{s}")], steps: std::mem::take(&mut b.steps) });

    let mut pre_mix = arg(Buf::Const("hc.one_hot_pre_mix".into()), DType::F32, &[t, s]);
    for l in 0..c.n_layers {
        let p = format!("layers.{l}");
        let mut blocks = Vec::new();
        // Engram (Transformer.forward, before the block): rows hashed on the host, fetched from the tier
        if let Some(e) = c.engram.as_ref().filter(|e| e.layer_ids.contains(&l)) {
            let cols = (e.max_ngram - 1) * e.n_heads;
            let rows = arg(Buf::Host(format!("engram.{l}.rows")), DType::Fp8E4m3, &[t, cols, e.head_dim]);
            let scales = arg(Buf::Host(format!("engram.{l}.scales")), DType::U8, &[t, cols, e.head_dim / 32]);
            let emb = ws("engram.emb", act, &[t, cols * e.head_dim]);
            b.push(Op::EngramGather, format!("{p}.engram.embed"), vec![rows, scales], vec![emb.clone()], OpAttrs::default());
            let kv = b.matmul(emb, &format!("{p}.engram.wkv"), Buf::Ws("engram.kv".into()), act)?;
            let (qw, kw) = (b.weight(&format!("{p}.engram.q_weight"))?, b.weight(&format!("{p}.engram.k_weight"))?);
            let y = flip(&x);
            b.push(Op::EngramMerge, format!("{p}.engram"), vec![x.clone(), kv, qw, kw], vec![y.clone()], OpAttrs { eps: c.norm_eps, ..Default::default() });
            x = y;
            blocks.push("engram".to_string());
        }

        // attention sub-block (Block.forward)
        let [attn_pre, attn_post, attn_comb] = b.hc_mixes(&x, &format!("{p}.hc_attn"), "hc_attn")?;
        let a_in = ws("a", act, &[t, d]);
        b.push(Op::HcPre, format!("{p}.hc_pre(attn)"), vec![x.clone(), pre_mix.clone()], vec![a_in.clone()], OpAttrs::default());
        let n = b.rmsnorm(a_in, &format!("{p}.attn_norm.weight"), "n")?;
        let attn_out = attention(&mut b, l, &lslots[l], &n, &mut blocks)?;
        let y = flip(&x);
        b.push(Op::HcPost, format!("{p}.hc_post(attn)"), vec![attn_out, x.clone(), attn_post, attn_comb], vec![y.clone()], OpAttrs::default());
        x = y;

        // MoE sub-block
        let [ffn_pre, ffn_post, ffn_comb] = b.hc_mixes(&x, &format!("{p}.hc_ffn"), "hc_ffn")?;
        let f_in = ws("a", act, &[t, d]);
        b.push(Op::HcPre, format!("{p}.hc_pre(ffn)"), vec![x.clone(), attn_pre], vec![f_in.clone()], OpAttrs::default());
        let n2 = b.rmsnorm(f_in, &format!("{p}.ffn_norm.weight"), "n")?;
        let ffn_out = moe(&mut b, &format!("{p}.ffn"), &n2, c.n_routed_experts, c.top_k)?;
        blocks.push(format!("hc_moe({}x top{} + shared)", c.n_routed_experts, c.top_k));
        let y = flip(&x);
        b.push(Op::HcPost, format!("{p}.hc_post(ffn)"), vec![ffn_out, x.clone(), ffn_post, ffn_comb], vec![y.clone()], OpAttrs::default());
        x = y;
        // the next layer's attention collapses with this FFN's pre-mix (written to "hc_ffn.pre", not overwritten before)
        pre_mix = ffn_pre;
        stages.push(Stage { name: format!("layer {l}"), layer: Some(l), blocks, steps: std::mem::take(&mut b.steps) });
    }

    // head: collapse with the last FFN's pre-mix, norm, logits of the last row (ParallelHead)
    let hc = ws("a", act, &[t, d]);
    b.push(Op::HcPre, "hc_pre(head)", vec![x.clone(), pre_mix], vec![hc.clone()], OpAttrs::default());
    b.rmsnorm(hc, "norm.weight", "n")?;
    let last = Arg { buf: Buf::Ws("n.last_row".into()), dtype: act, shape: vec![1, d] };
    b.matmul(last, "head", Buf::Ws("logits".into()), DType::F32)?;
    stages.push(Stage { name: "head".into(), layer: None, blocks: vec!["hc_pre".into(), "norm".into(), "head".into()], steps: std::mem::take(&mut b.steps) });

    Ok(Dsv41Graph { stages, slots: slot_specs, residual_streams: s, mtp: mtp_blocks(c, w), cap: o.cap })
}

/// The attention of layer `l` (`Attention.forward`); returns its `[t, dim]` output.
fn attention(b: &mut Builder, l: usize, ls: &LayerSlots, n: &Arg, blocks: &mut Vec<String>) -> Result<Arg> {
    let c = b.c;
    let (t, act, pos0) = (b.o.t, b.o.act, b.o.pos0);
    let (h, hd, win) = (c.n_heads, c.head_dim, c.window);
    let p = format!("layers.{l}.attn");
    let r = c.ratio(l);

    // q: low rank, normed, rotated tail
    let qa = b.matmul(n.clone(), &format!("{p}.wq_a"), Buf::Ws("qa".into()), act)?;
    let qr = b.rmsnorm(qa, &format!("{p}.q_norm.weight"), "qr")?;
    let q = b.matmul(qr.clone(), &format!("{p}.wq_b"), Buf::Ws("q".into()), act)?;
    let q = b.rope(Arg { shape: vec![t, h, hd], ..q }, l, pos0, 1, false, Buf::Ws("q_rope".into()), &format!("{p}.rope(q)"));

    // sliding-window KV: one latent, normed, rotated, fp8 round trip, into the ring
    let kv = b.matmul(n.clone(), &format!("{p}.wkv"), Buf::Ws("kv".into()), act)?;
    let kv = b.rmsnorm(kv, &format!("{p}.kv_norm.weight"), "kv_n")?;
    let kv = b.rope(Arg { shape: vec![t, 1, hd], ..kv }, l, pos0, 1, false, Buf::Ws("kv_rope".into()), &format!("{p}.rope(kv)"));
    let kv = b.qdq(Arg { shape: vec![t, hd], ..kv }, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0, Buf::Ws("kv_q".into()), &format!("{p}.act_quant(kv)"));
    let winbuf = ws("win", act, &[win + t, hd]);
    let ring = arg(Buf::Slot { slot: ls.window, plane: 0, row0: 0 }, act, &[win, hd]);
    b.push(Op::WindowKv, format!("{p}.window_kv"), vec![kv], vec![winbuf.clone(), ring], OpAttrs { window: win, pos0, ..Default::default() });

    let (nk, groups) = if r > 0 { compressed_rows(pos0, t, r) } else { (0, 0) };
    let mut desc = format!("hc_attn(mqa hd{hd} + sink, window {win}");
    let (cmp, idx) = if r > 0 {
        let src = c.kv_source(l).expect("validated");
        let isrc = c.index_source(l).expect("validated");
        let cmp_slot = ls.cmp.expect("kv readers have a slot");
        let ik_slot = ls.index_k.expect("kv readers have a slot");
        let row0 = pos0 / r;
        if src == l {
            // the compressor (Compressor.forward): ratio 1 is a plain projection, above it softmax-gated pooling in f32
            let latent = if r > 1 {
                let ckv = b.matmul(n.clone(), &format!("{p}.compressor.wkv"), Buf::Ws("cmp.kv".into()), DType::F32)?;
                let cs = b.matmul(n.clone(), &format!("{p}.compressor.wgate"), Buf::Ws("cmp.score".into()), DType::F32)?;
                let tail = ls.tail.expect("ratio > 1 sources have a tail ring");
                let outs = vec![
                    ws("latent", act, &[groups, hd]),
                    arg(Buf::Slot { slot: tail, plane: 0, row0: 0 }, DType::F32, &[r, hd]),
                    arg(Buf::Slot { slot: tail, plane: 1, row0: 0 }, DType::F32, &[r, hd]),
                ];
                let mut a = OpAttrs { pos0, ..Default::default() };
                a.dsv.ratio = r;
                b.push(Op::CompressPool, format!("{p}.compressor"), vec![ckv, cs], outs.clone(), a);
                outs[0].clone()
            } else {
                b.matmul(n.clone(), &format!("{p}.compressor.wkv"), Buf::Ws("latent".into()), act)?
            };
            if groups > 0 {
                let latent = b.rmsnorm(latent, &format!("{p}.compressor.norm.weight"), "latent_n")?;
                // index keys from the pre-rope latent (Indexer.forward, owns_k), then the latent itself
                let k = b.matmul(latent.clone(), &format!("{p}.indexer.wk"), Buf::Ws("index_k".into()), act)?;
                let k = b.rmsnorm(k, &format!("{p}.indexer.k_norm.weight"), "index_k_n")?;
                let k = b.rope(Arg { shape: vec![groups, 1, c.index_head_dim], ..k }, l, row0 * r, r, false, Buf::Ws("index_k_rope".into()), &format!("{p}.rope(index_k)"));
                b.qdq(
                    Arg { shape: vec![groups, c.index_head_dim], ..k },
                    QdqFmt::Fp4E2m1,
                    32,
                    QdqScale::Ue8m0,
                    Buf::Slot { slot: ik_slot, plane: 0, row0 },
                    &format!("{p}.act_quant(index_k)"),
                );
                let lat = b.rope(Arg { shape: vec![groups, 1, hd], ..latent }, l, row0 * r, r, false, Buf::Ws("latent_rope".into()), &format!("{p}.rope(latent)"));
                b.qdq(Arg { shape: vec![groups, hd], ..lat }, QdqFmt::Fp4E2m1, 16, QdqScale::E4m3, Buf::Slot { slot: cmp_slot, plane: 0, row0 }, &format!("{p}.act_quant(latent)"));
            }
            desc.push_str(&format!(" + compressor(ratio {r})"));
        }
        let kk = c.index_topk.min(nk);
        if isrc == l && nk > 0 {
            // the indexer (Indexer.forward): fp4 query heads against the shared index keys
            let iq = b.matmul(qr, &format!("{p}.indexer.wq_b"), Buf::Ws("index_q".into()), act)?;
            let iq = b.rope(Arg { shape: vec![t, c.index_n_heads, c.index_head_dim], ..iq }, l, pos0, 1, false, Buf::Ws("index_q_rope".into()), &format!("{p}.rope(index_q)"));
            let iq = b.qdq(iq, QdqFmt::Fp4E2m1, 32, QdqScale::Ue8m0, Buf::Ws("index_q_q".into()), &format!("{p}.act_quant(index_q)"));
            let iw = b.matmul(n.clone(), &format!("{p}.indexer.weights_proj"), Buf::Ws("index_w".into()), act)?;
            let keys = arg(Buf::Slot { slot: ik_slot, plane: 0, row0: 0 }, act, &[nk, c.index_head_dim]);
            let scale = (c.index_head_dim as f32).powf(-0.5) * (c.index_n_heads as f32).powf(-0.5);
            let mut a = OpAttrs { pos0, scale, ..Default::default() };
            (a.dsv.ratio, a.dsv.block) = (r, c.candidate_block_size.max(1));
            let nb = n_blocks(nk, a.dsv.block);
            if c.candidate_source_layer == Some(l) {
                let mask = ws("candidates", DType::U8, &[t, nb]);
                b.push(Op::IndexerBlocks, format!("{p}.indexer.candidates"), vec![iq.clone(), keys.clone(), iw.clone()], vec![mask], OpAttrs { top_k: c.candidate_topk_blocks, ..a.clone() });
                desc.push_str(&format!(" + candidate blocks top{}", c.candidate_topk_blocks));
            }
            let cand = if c.uses_candidates(l) { ws("candidates", DType::U8, &[t, nb]) } else { arg(Buf::Const("empty_u8".into()), DType::U8, &[0]) };
            let idx = ws(&format!("topk_idxs.{l}"), DType::I32, &[t, kk]);
            b.push(Op::IndexerTopk, format!("{p}.indexer"), vec![iq, keys, iw, cand], vec![idx], OpAttrs { top_k: c.index_topk, ..a });
            desc.push_str(&format!(" + indexer top{}{}", c.index_topk, if c.uses_candidates(l) { " in candidates" } else { "" }));
        }
        if src != l {
            desc.push_str(&format!(" + cmp(ratio {r}) of layer {src}"));
        }
        if isrc != l {
            desc.push_str(&format!(", idx of layer {isrc}"));
        }
        (arg(Buf::Slot { slot: cmp_slot, plane: 0, row0: 0 }, act, &[nk, hd]), ws(&format!("topk_idxs.{isrc}"), DType::I32, &[t, kk]))
    } else {
        (arg(Buf::Const("empty".into()), act, &[0, hd]), arg(Buf::Const("empty_i32".into()), DType::I32, &[t, 0]))
    };
    desc.push(')');
    blocks.push(desc);

    let sink = b.weight(&format!("{p}.attn_sink"))?;
    let o = ws("o", act, &[t, h, hd]);
    b.push(Op::AttnSparse, format!("{p}.sparse_attn"), vec![q, winbuf, cmp, idx, sink], vec![o.clone()], OpAttrs { window: win, pos0, scale: (hd as f32).powf(-0.5), ..Default::default() });
    let o = b.rope(o, l, pos0, 1, true, Buf::Ws("o_derot".into()), &format!("{p}.rope_inverse(o)"));

    // grouped low-rank output: fp8 round trip, per-group projection, then wo_b
    let g = c.o_groups;
    let gk = h * hd / g;
    let o = b.qdq(Arg { shape: vec![t, g, gk], ..o }, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0, Buf::Ws("o_q".into()), &format!("{p}.act_quant(o)"));
    let oa = if b.has(&format!("{p}.wo_a.slice.0.trellis")) {
        // EXL3 (Mia): one matmul per group slice on group-major rows
        let og = ws("o_groups", act, &[g, t, gk]);
        b.push(Op::Transpose01, format!("{p}.wo_a(group-major)"), vec![o], vec![og], OpAttrs::default());
        for gi in 0..g {
            let x = Arg { buf: Buf::Ws(format!("o_groups[{gi}]")), dtype: act, shape: vec![t, gk] };
            b.matmul(x, &format!("{p}.wo_a.slice.{gi}"), Buf::Ws(format!("oa_groups[{gi}]")), act)?;
        }
        let oa = ws("oa", act, &[t, g, c.o_lora_rank]);
        b.push(Op::Transpose01, format!("{p}.wo_a(token-major)"), vec![ws("oa_groups", act, &[g, t, c.o_lora_rank])], vec![oa.clone()], OpAttrs::default());
        Arg { shape: vec![t, g * c.o_lora_rank], ..oa }
    } else {
        // dense block-diagonal weight (the base after convert.py): one grouped matmul
        let (wdt, ws_) = b.w.get(&format!("{p}.wo_a.weight")).ok_or_else(|| Error::model(format!("deepseek_v41: no {p}.wo_a")))?;
        if ws_ != [g * c.o_lora_rank, gk] {
            return Err(Error::model(format!("deepseek_v41: {p}.wo_a.weight shape {ws_:?}")));
        }
        // an fp8 wo_a (with `.scale`) is dequantised to bf16 at load, as convert.py does
        let wdt = if wdt == DType::Fp8E4m3 { DType::Bf16 } else { wdt };
        let wv = arg(Buf::Weight(format!("{p}.wo_a.weight")), wdt, &[g, c.o_lora_rank, gk]);
        let oa = ws("oa", act, &[t, g * c.o_lora_rank]);
        b.push(Op::GroupedMatmul, format!("{p}.wo_a"), vec![Arg { shape: vec![t, g * gk], ..o }, wv], vec![oa.clone()], OpAttrs::default());
        oa
    };
    b.matmul(oa, &format!("{p}.wo_b"), Buf::Ws("blk".into()), act)
}

/// The MoE (`MoE.forward`): router, routed EXL3 experts, shared expert; returns `[t, dim]`.
fn moe(b: &mut Builder, p: &str, n: &Arg, experts: usize, top_k: usize) -> Result<Arg> {
    let c = b.c;
    let (t, d, act) = (b.o.t, c.dim, b.o.act);
    let logits = b.matmul(n.clone(), &format!("{p}.gate"), Buf::Ws("router_logits".into()), DType::F32)?;
    let bias = b.weight(&format!("{p}.gate.bias"))?;
    let bias = Arg { dtype: DType::F32, ..bias };
    let (rw, ids) = (ws("route_w", DType::F32, &[t, top_k]), ws("route_ids", DType::I32, &[t, top_k]));
    b.push(Op::MoeRoute, format!("{p}.gate"), vec![logits, bias], vec![rw.clone(), ids.clone()], OpAttrs { top_k, scale: c.route_scale, norm_topk: c.norm_topk, ..Default::default() });

    let mut inputs = vec![n.clone(), ids, rw];
    for wn in ["w1", "w3", "w2"] {
        let e0 = format!("{p}.experts.0.{wn}");
        let last = format!("{p}.experts.{}.{wn}", experts - 1);
        let (Some((_, s0)), Some((_, sl))) = (b.w.get(&format!("{e0}.trellis")), b.w.get(&format!("{last}.trellis"))) else {
            return Err(Error::Unsupported(format!(
                "deepseek_v41: routed experts {p}.experts.*.{wn} are not EXL3 (the base's FP4 experts run as MoeExperts with dsv.limit, lane 1b)"
            )));
        };
        if s0 != sl || s0.len() != 3 {
            return Err(Error::model(format!("deepseek_v41: {p} experts 0 and {} differ in shape ({s0:?} / {sl:?})", experts - 1)));
        }
        let (k, nn) = (16 * s0[0], 16 * s0[1]);
        inputs.push(arg(Buf::Weight(format!("{p}.experts.*.{wn}.trellis")), DType::I16, &[experts, s0[0], s0[1], s0[2]]));
        inputs.push(arg(Buf::Weight(format!("{p}.experts.*.{wn}.suh")), DType::F16, &[experts, k]));
        inputs.push(arg(Buf::Weight(format!("{p}.experts.*.{wn}.svh")), DType::F16, &[experts, nn]));
    }
    inputs.push(arg(Buf::Weight(format!("{p}.experts.*.codebook")), DType::I32, &[1]));
    let routed = ws("moe_routed", act, &[t, d]);
    let mut a = OpAttrs { top_k, ..Default::default() };
    a.dsv.limit = c.swiglu_limit;
    b.push(Op::MoeExpertsExl3, format!("{p}.experts"), inputs, vec![routed.clone()], a);

    // shared expert: clamped SwiGLU
    let g = b.matmul(n.clone(), &format!("{p}.shared_experts.w1"), Buf::Ws("g".into()), act)?;
    let u = b.matmul(n.clone(), &format!("{p}.shared_experts.w3"), Buf::Ws("u".into()), act)?;
    let hsh = ws("h_shared", act, &g.shape);
    let mut a = OpAttrs::default();
    a.dsv.limit = c.swiglu_limit;
    b.push(Op::SiluGluClamp, format!("{p}.shared_experts.act"), vec![g, u], vec![hsh.clone()], a);
    let sh = b.matmul(hsh, &format!("{p}.shared_experts.w2"), Buf::Ws("moe_shared".into()), act)?;
    let out = ws("blk", act, &[t, d]);
    b.push(Op::Add, format!("{p}.combine"), vec![routed, sh], vec![out.clone()], OpAttrs::default());
    Ok(out)
}

/// The MTP / DSpark blocks the checkpoint carries (`mtp.{i}.*`).
fn mtp_blocks(c: &Dsv41Config, w: &dyn WeightShapes) -> Vec<MtpBlock> {
    let Some(m) = &c.mtp else { return Vec::new() };
    (0..m.n_layers)
        .filter(|i| w.get(&format!("mtp.{i}.attn_norm.weight")).is_some())
        .map(|i| MtpBlock {
            index: i,
            prefix: format!("mtp.{i}"),
            experts: m.n_routed_experts,
            top_k: m.top_k,
            tensors: (0..m.n_routed_experts).filter(|e| w.get(&format!("mtp.{i}.ffn.experts.{e}.w1.trellis")).is_some() || w.get(&format!("mtp.{i}.ffn.experts.{e}.w1.weight")).is_some()).count(),
        })
        .collect()
}

#[derive(Debug)]
struct Phantom(usize);
impl BufferImpl for Phantom {
    fn bytes(&self) -> usize {
        self.0
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Arg {
    /// A tensor of this dtype and shape with no memory behind it (plan recording, admission shapes).
    pub fn phantom(&self) -> TensorRef {
        TensorRef::new(Buffer::new(Phantom(self.shape.iter().product::<usize>() * self.dtype.size_bytes())), self.dtype, &self.shape)
    }
}

impl Step {
    /// The step as an `OpCall` on shape-only tensors (checked like any call).
    pub fn phantom_call(&self) -> Result<OpCall> {
        let call = OpCall::new(self.op, self.inputs.iter().map(Arg::phantom).collect(), self.outputs.iter().map(Arg::phantom).collect(), self.attrs.clone());
        call.check().map_err(|e| Error::model(format!("deepseek_v41 graph step {}: {e}", self.what)))?;
        Ok(call)
    }
}
