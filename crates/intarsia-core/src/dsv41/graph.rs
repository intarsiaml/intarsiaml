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
    /// The step reads the pass's first position from the device (`OpCall::pos`, the runtime's `i32 [1]` position buffer)
    /// instead of `attrs.pos0`: a device-position graph ([`PassKind::dpos`]), replayable at any position.
    pub pos: bool,
    /// 0: the pass's own stream. 1..=4: a lane of a fork the runtime opens in a recording (`INTARSIA_DSV41_LANES`), where the
    /// steps of different lanes between two lane-0 steps may run side by side (they share no buffer). Ignored when not forking.
    pub lane: u8,
}

/// Embedding, one layer, or the head.
#[derive(Clone, Debug)]
pub struct Stage {
    pub name: String,
    pub layer: Option<usize>,
    /// Block summary for `plan show` (e.g. `["engram", "hc_attn:window+cmp(2)/idx", "hc_moe"]`).
    pub blocks: Vec<String>,
    pub steps: Vec<Step>,
    /// Host-side actions that are not ops: `(j, action)` runs right before `steps[j]` (`j == steps.len()`: after the
    /// last step). Collectives of a tensor-parallel rank and constant fills.
    pub extras: Vec<(usize, Extra)>,
}

/// A host-side action of a stage (not an `Op`: the op set is frozen).
#[derive(Clone, Debug, PartialEq)]
pub enum Extra {
    /// Sum `x` over the ranks in place (bf16 / f32).
    AllReduce { x: Arg },
    /// `out = round(sum over ranks of partial)`: f32 partials rounded once into `out` (`Comm::all_reduce_sum_into`).
    AllReduceInto { partial: Arg, out: Arg },
    /// `recv` (`[.., world * n]`) gets every rank's `send` (`[.., n]`) in rank order (one row only).
    AllGather { send: Arg, recv: Arg },
    /// Elements `offset .. offset + count` of workspace `buf` (dtype `dtype`) set to `value`.
    Fill { buf: String, dtype: DType, offset: usize, count: usize, value: f64 },
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
    /// The full-vocabulary f32 logits of the last row (`[1, vocab]`), after the head (and its gather under tensor
    /// parallelism).
    pub logits: Arg,
    /// The argmax of every row of `logits` (`[rows]` i32; first index of the strict maximum, as
    /// `sampler::argmax`), when the graph computes it: a verify pass ([`PassKind::all_rows`]). Lets the host download
    /// `rows` ints instead of `rows * vocab` floats.
    pub ids: Option<Arg>,
    /// Tensor-parallel group size the graph was built for (1: one rank).
    pub world: usize,
    /// A verify pass (`PassKind::all_rows`): its EXL3 linears may take split-K like the one-row decode pass.
    pub verify: bool,
}

/// Tensor parallelism and truncation of the graph ([`build_with`]).
#[derive(Clone, Copy, Debug)]
pub struct GraphTp {
    /// Ranks of the group. The weights the graph is built against are this rank's (a rank cache: column-parallel
    /// linears hold `N / world` outputs, row-parallel `K / world` inputs, `attn_sink` `n_heads / world`, the rank's
    /// `o_groups / world` `wo_a` slices renumbered from 0, the routed experts `I / world` wide).
    pub world: usize,
    /// Run only the first `layers` layers, then the head (few-layer parity runs on a truncated rank cache).
    pub layers: Option<usize>,
    /// Start at this layer (a layer-range test cache, `intarsia dsv41-cache --first-layer`): the embedding feeds layer
    /// `first` directly. Not a model; for exactness checks of the layers in the range.
    pub first: usize,
}

impl Default for GraphTp {
    fn default() -> Self {
        GraphTp { world: 1, layers: None, first: 0 }
    }
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
    tp: GraphTp,
    steps: Vec<Step>,
    extras: Vec<(usize, Extra)>,
    /// Device-position graph ([`PassKind::dpos`]).
    dpos: bool,
    /// The lane the steps pushed now get ([`Step::lane`]).
    lane: u8,
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
        self.steps.push(Step { op, what: what.into(), inputs, outputs, attrs, pos: false, lane: self.lane });
    }

    /// In a device-position graph, the last step pushed reads the position from the device; `ratio` is its
    /// `dsv.ratio` there (the compressed latents' rope and slot writes, the sparse attention's visible rows).
    fn positioned(&mut self, ratio: usize) {
        if self.dpos {
            let s = self.steps.last_mut().expect("a step");
            s.pos = true;
            s.attrs.dsv.ratio = ratio.max(1);
        }
    }

    /// The last step's (an `ActQdq` into a shared slot) output shape: `[rows, width]` (the whole slot in a device-position
    /// graph, the new groups' rows otherwise).
    fn slot_rows(&mut self, rows: usize, width: usize) {
        let s = self.steps.last_mut().expect("a step");
        s.outputs[0].shape = vec![rows, width];
    }

    /// A host action after the steps pushed so far.
    fn extra(&mut self, e: Extra) {
        self.extras.push((self.steps.len(), e));
    }

    fn stage(&mut self, name: String, layer: Option<usize>, blocks: Vec<String>) -> Stage {
        Stage { name, layer, blocks, steps: std::mem::take(&mut self.steps), extras: std::mem::take(&mut self.extras) }
    }

    /// A row-parallel linear's output: under tensor parallelism an f32 partial summed over the ranks into `out` (act).
    fn matmul_reduced(&mut self, x: Arg, p: &str, out: &str, act: DType) -> Result<Arg> {
        if self.tp.world == 1 {
            return self.matmul(x, p, Buf::Ws(out.into()), act);
        }
        let part = self.matmul(x, p, Buf::Ws(format!("{out}.partial")), DType::F32)?;
        let o = Arg { buf: Buf::Ws(out.into()), dtype: act, shape: part.shape.clone() };
        self.extra(Extra::AllReduceInto { partial: part, out: o.clone() });
        Ok(o)
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

    /// The fused hyper-connection sublayer (docs/design/dsv41-ops.md "Wiring the fused HcSublayer"): `HcMixes(h)` of the
    /// sublayer `p` (`{p}_fn` / `_scale` / `_base`) into `{out}.pre` / `.post` / `.comb`, and
    /// `n = RmsNorm(HcPre(h, pre_in), norm_w)`. With `fold = Some(pending HcPost)`, h = that `HcPost` of `x`, written to
    /// `h_out` (the other residual buffer); without it h = x. Returns (n, [pre, post, comb], the stream after the step).
    fn hc_sublayer(&mut self, x: &Arg, p: &str, out: &str, sub: &str, pre_in: Arg, norm_w: &str, fold: Option<PendingPost>) -> Result<(Arg, [Arg; 3], Arg)> {
        let (t, s, d, act) = (self.o.t, self.c.hc_mult, self.c.dim, self.o.act);
        if !hc_fused() {
            // the separate ops (INTARSIA_DSV41_HC_FUSED=0): HcPost, HcMixes, HcPre, RmsNorm, as before the fused op
            let mut fold = fold;
            let x = self.flush_post(x, &mut fold);
            let mixes = self.hc_mixes(&x, p, out)?;
            let a_in = ws("a", act, &[t, d]);
            self.push(Op::HcPre, format!("{}.hc_pre({sub})", p.rsplit_once('.').map_or(p, |(l, _)| l)), vec![x.clone(), pre_in], vec![a_in.clone()], OpAttrs::default());
            let n = self.rmsnorm(a_in, norm_w, "n")?;
            return Ok((n, mixes, x));
        }
        let outs = [ws(&format!("{out}.pre"), DType::F32, &[t, s]), ws(&format!("{out}.post"), DType::F32, &[t, s]), ws(&format!("{out}.comb"), DType::F32, &[t, s, s])];
        let n = ws("n", act, &[t, d]);
        let (post_x, post_mix, post_comb, h_out) = match &fold {
            Some(f) => (f.out.clone(), f.post.clone(), f.comb.clone(), flip(x)),
            None => {
                let e = |dt| arg(Buf::Const("empty".into()), dt, &[0]);
                // a distinct (also empty) constant for the absent output, so no step names one buffer as input and output
                (e(act), e(DType::F32), e(DType::F32), arg(Buf::Const("empty_out".into()), act, &[0]))
            }
        };
        let inputs = vec![
            x.clone(),
            self.weight(&format!("{p}_fn"))?,
            self.weight(&format!("{p}_scale"))?,
            self.weight(&format!("{p}_base"))?,
            pre_in,
            self.weight(norm_w)?,
            post_x,
            post_mix,
            post_comb,
        ];
        let outputs = vec![n.clone(), outs[0].clone(), outs[1].clone(), outs[2].clone(), h_out.clone()];
        // the op's contract: no output aliases an input
        for o in outputs.iter().filter(|o| !o.shape.contains(&0)) {
            if inputs.iter().any(|i| i.buf == o.buf && !i.shape.contains(&0)) {
                return Err(Error::invalid(format!("deepseek_v41 graph: {p} hc_sublayer output {:?} aliases an input", o.buf)));
            }
        }
        let mut a = OpAttrs { eps: self.c.norm_eps, ..Default::default() };
        (a.dsv.iters, a.dsv.hc_eps) = (self.c.hc_sinkhorn_iters, self.c.hc_eps);
        let what = match &fold {
            Some(f) => format!("{p}.hc_sublayer(+{})", f.what),
            None => format!("{p}.hc_sublayer"),
        };
        self.push(Op::HcSublayer, what, inputs, outputs, a);
        Ok((n, outs, if fold.is_some() { h_out } else { x.clone() }))
    }

    /// Emits a pending `HcPost` as its own step; returns the stream after it.
    fn flush_post(&mut self, x: &Arg, pending: &mut Option<PendingPost>) -> Arg {
        match pending.take() {
            Some(f) => {
                let y = flip(x);
                self.push(Op::HcPost, f.what, vec![f.out, x.clone(), f.post, f.comb], vec![y.clone()], OpAttrs::default());
                y
            }
            None => x.clone(),
        }
    }
}

/// A sublayer's `HcPost(out, x, post, comb)` not yet applied to the residual streams: the next `HcSublayer` folds it in
/// (decode / verify, `T <= HC_FOLD_MAX_ROWS`), or `Builder::flush_post` emits it as its own step.
struct PendingPost {
    out: Arg,
    post: Arg,
    comb: Arg,
    /// the step name it would have had on its own (`layers.{l}.hc_post(attn)` / `(ffn)`)
    what: String,
}

/// `INTARSIA_DSV41_HC_FUSED` (default on): each sublayer's HcMixes + HcPre + RmsNorm (+ the previous HcPost) as one
/// `HcSublayer`; `0` emits the separate ops (the graph before lane dsv41/tp2 wired the fused op), for A/B and diagnosis.
pub fn hc_fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_DSV41_HC_FUSED").map_or(true, |v| v != "0"))
}

/// Rows up to which an `HcSublayer` folds the previous sublayer's `HcPost` (docs/design/dsv41-ops.md: on a 4096-row
/// prefill chunk the fold is slower than the separate `HcPost`, at decode it saves ~6 us per sublayer on sm_121).
pub const HC_FOLD_MAX_ROWS: usize = 16;

/// The other residual buffer (`x0` <-> `x1`).
fn flip(x: &Arg) -> Arg {
    let other = if x.buf == Buf::Ws("x0".into()) { "x1" } else { "x0" };
    Arg { buf: Buf::Ws(other.into()), ..x.clone() }
}

/// Tokens a truncate / speculative rollback may cut while the rings still hold every row the window and the pending
/// compressor group need (`SlotSpec::Ring::margin`): DSpark verifies blocks of 5, n-gram drafts a few more.
pub const ROLLBACK_MARGIN: usize = 16;

/// Rows of a compressor tail ring of ratio `r`: a multiple of `r` holding the `r - 1` pending tokens plus
/// [`ROLLBACK_MARGIN`] (cutting `m` tokens re-opens a group whose first rows were written up to `m + r - 1` steps earlier).
pub fn tail_rows(r: usize) -> usize {
    r * (ROLLBACK_MARGIN + r).div_ceil(r)
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
/// compressed KV and the shared index keys, each read by every layer whose source it is; then one window ring per
/// DSpark stage ([`dspark_rings`]).
/// `dspark`: the layout of a sequence that drafts (the rings exist only then, so the plain graph's layout is unchanged).
fn slots(c: &Dsv41Config, w: &dyn WeightShapes, act: DType, dspark: bool) -> (Vec<SlotSpec>, Vec<LayerSlots>, Vec<usize>) {
    let (mut specs, ls) = model_slots(c, act);
    let mut rings = Vec::new();
    for _ in 0..if dspark { dspark_stages(c, w) } else { 0 } {
        rings.push(specs.len());
        let m = ROLLBACK_MARGIN;
        specs.push(SlotSpec::Ring { rows: c.window + m, width: c.head_dim, planes: 1, dtype: act, margin: m, kind: RingKind::WindowKv });
    }
    (specs, ls, rings)
}

/// DSpark stages the engine runs: the checkpoint's `mtp.{i}` blocks when all of them, the stage-0 `main_proj` and the
/// last stage's head parts are loaded (a rank cache without its `mtp` part has none), else 0.
pub fn dspark_stages(c: &Dsv41Config, w: &dyn WeightShapes) -> usize {
    let Some(m) = &c.mtp else { return 0 };
    if m.n_layers == 0 || m.block_size == 0 || m.target_layer_ids.is_empty() {
        return 0;
    }
    let last = m.n_layers - 1;
    let lin = |p: &str| w.get(&format!("{p}.trellis")).is_some() || w.get(&format!("{p}.weight")).is_some();
    let ok = lin("mtp.0.main_proj")
        && (0..m.n_layers).all(|i| lin(&format!("mtp.{i}.attn.wkv")) && w.get(&format!("mtp.{i}.attn_norm.weight")).is_some())
        && w.get(&format!("mtp.{last}.norm.weight")).is_some()
        && w.get(&format!("mtp.{last}.markov_head.embed.weight")).is_some()
        && w.get(&format!("mtp.{last}.markov_head.head.weight")).is_some();
    if ok {
        m.n_layers
    } else {
        0
    }
}

/// The DSpark taps and stage rings are fed by every pass that reaches the last target layer.
fn dspark_absorbs(c: &Dsv41Config, w: &dyn WeightShapes, first: usize, n_layers: usize) -> bool {
    dspark_stages(c, w) > 0 && c.mtp.as_ref().is_some_and(|m| m.target_layer_ids.iter().all(|&l| (first..n_layers).contains(&l)))
}

/// Rows up to which the head computes the logits of every row (a speculative verify pass; `t == 1` and longer chunks:
/// the last row only).
pub const VERIFY_MAX_ROWS: usize = 16;

/// The model's own slots (see [`slots`]).
fn model_slots(c: &Dsv41Config, act: DType) -> (Vec<SlotSpec>, Vec<LayerSlots>) {
    let mut specs = Vec::new();
    let mut ls = vec![LayerSlots::default(); c.n_layers];
    for (l, s) in ls.iter_mut().enumerate() {
        s.window = specs.len();
        let m = ROLLBACK_MARGIN;
        specs.push(SlotSpec::Ring { rows: c.window + m, width: c.head_dim, planes: 1, dtype: act, margin: m, kind: RingKind::WindowKv });
        let _ = l;
    }
    for &src in &c.kv_source_layers {
        let r = c.ratio(src);
        let readers: Vec<usize> = (0..c.n_layers).filter(|&l| c.kv_source(l) == Some(src)).collect();
        if r > 1 {
            ls[src].tail = Some(specs.len());
            let (rows, margin) = (tail_rows(r), ROLLBACK_MARGIN);
            specs.push(SlotSpec::Ring { rows, width: c.head_dim, planes: 2, dtype: DType::F32, margin, kind: RingKind::CompressTail });
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

/// Build the graph of one chunk (one rank, every layer).
pub fn build(c: &Dsv41Config, w: &dyn WeightShapes, o: GraphOpts) -> Result<Dsv41Graph> {
    build_with(c, w, o, GraphTp::default())
}

/// Build the graph of one chunk for one rank of a tensor-parallel group ([`GraphTp`]). Sharding (TensorFold's layout,
/// docs/research/2026-10-31-dsv41-porting-map.md 3.3): attention heads split (`wq_b` columns, sinks, `wo_a` groups,
/// `wo_b` rows), routed + shared experts split by intermediate width, head by vocab; everything else (embedding, norms,
/// `wq_a`, `wkv`, compressor, indexer, router, hc params, Engram `wkv`) replicated. Collectives: one f32 all-reduce
/// after `wo_b` and one after the MoE per layer, one of the gathered Engram rows per Engram layer (a rank's foreign rows
/// are exact zeros), one gather of the head's logits.
/// What a pass computes beyond the plain forward (both default off: the graph is the one without speculation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassKind {
    /// The head's logits of every row (a speculative verify pass, at most [`VERIFY_MAX_ROWS`] rows); else the last row's.
    pub all_rows: bool,
    /// Feed the DSpark stage rings from this pass's taps (every pass of a sequence that drafts; needs the stages loaded).
    pub absorb: bool,
    /// Device-position graph (CUDA graph capture): every positioned step reads the pass's first position from the device
    /// (`Step::pos`), and nothing depends on the host position: the shared compressed-KV / index-key views span the slot
    /// capacity, the top-k width is `min(index_topk, capacity rows)` (-1 past the visible count), the compressor branch
    /// is always emitted for `ceil(t / r)` groups, the slot writes take their row from the device. Bitwise the host
    /// graph's results (docs/design/dsv41-ops.md "Device position"); `o.pos0` only bounds the call.
    pub dpos: bool,
}

/// The plain forward graph ([`build_pass`] with the default [`PassKind`]).
pub fn build_with(c: &Dsv41Config, w: &dyn WeightShapes, o: GraphOpts, tp: GraphTp) -> Result<Dsv41Graph> {
    build_pass(c, w, o, tp, PassKind::default())
}

/// The forward graph of one chunk with the extras of `k`.
pub fn build_pass(c: &Dsv41Config, w: &dyn WeightShapes, o: GraphOpts, tp: GraphTp, k: PassKind) -> Result<Dsv41Graph> {
    if k.all_rows && !(1..=VERIFY_MAX_ROWS).contains(&o.t) {
        return Err(Error::invalid(format!("deepseek_v41: a verify pass of {} rows (1..={VERIFY_MAX_ROWS})", o.t)));
    }
    let (t, d, s, act) = (o.t, c.dim, c.hc_mult, o.act);
    if t == 0 {
        return Err(Error::invalid("deepseek_v41 graph: zero rows"));
    }
    if tp.world == 0 || c.n_heads % tp.world != 0 || c.o_groups % tp.world != 0 {
        return Err(Error::invalid(format!("deepseek_v41 graph: {} heads / {} groups do not split over {} ranks", c.n_heads, c.o_groups, tp.world)));
    }
    let n_layers = tp.layers.map_or(c.n_layers, |l| l.min(c.n_layers));
    let (slot_specs, lslots, rings) = slots(c, w, act, k.absorb);
    // a truncated model (`--layers` below the target layers: tests) keeps the drafting layout but has no taps to absorb
    if tp.first >= n_layers {
        return Err(Error::invalid(format!("deepseek_v41 graph: first layer {} of {n_layers}", tp.first)));
    }
    let absorb = k.absorb && dspark_absorbs(c, w, tp.first, n_layers);
    if k.absorb && dspark_stages(c, w) == 0 {
        return Err(Error::Unsupported("deepseek_v41: DSpark absorb without the DSpark stages".into()));
    }
    let n_taps = c.mtp.as_ref().map_or(0, |m| m.target_layer_ids.len());
    let mut stages = Vec::new();
    let mut b = Builder { c, w, o, tp, steps: Vec::new(), extras: Vec::new(), dpos: k.dpos, lane: 0 };

    // embedding, expanded to hc_mult streams (Transformer.forward)
    let ids = arg(Buf::Host("token_ids".into()), DType::I32, &[t]);
    let emb = b.weight("embed.weight")?;
    let h = ws("h", act, &[t, d]);
    b.push(Op::Embedding, "embed", vec![ids, emb], vec![h.clone()], OpAttrs::default());
    // the residual ping-pongs between two buffers: HcPost mixes across streams and EngramMerge reads every stream, so no
    // step's output aliases its input (outputs never alias inputs in this graph, except the state rings marked in place)
    let mut x = ws("x0", act, &[t, s, d]);
    b.push(Op::HcExpand, "hc_expand", vec![h], vec![x.clone()], OpAttrs::default());
    stages.push(b.stage("embed".into(), None, vec!["embed".into(), format!("hc_expand x{s}")]));

    let mut pre_mix = arg(Buf::Const("hc.one_hot_pre_mix".into()), DType::F32, &[t, s]);
    // HcMixes + HcPre + RmsNorm of every sublayer run as one HcSublayer; at T <= HC_FOLD_MAX_ROWS it also folds the
    // previous sublayer's HcPost, unless something reads the streams in between (an Engram merge or the DSpark tap of the
    // next layer, the head): then that HcPost is emitted on its own at the end of its layer.
    let fold = hc_fused() && t <= HC_FOLD_MAX_ROWS;
    let reads_streams = |l: usize| {
        l >= n_layers
            || c.engram.as_ref().is_some_and(|e| e.layer_ids.contains(&l))
            || c.mtp.as_ref().is_some_and(|m| m.target_layer_ids.contains(&l))
    };
    let mut pending: Option<PendingPost> = None;
    for l in tp.first..n_layers {
        let p = format!("layers.{l}");
        let mut blocks = Vec::new();
        // Engram (Transformer.forward, before the block): rows hashed on the host, fetched from the tier
        if let Some(e) = c.engram.as_ref().filter(|e| e.layer_ids.contains(&l)) {
            let cols = (e.max_ngram - 1) * e.n_heads;
            let rec = super::engram::record_bytes(e.head_dim);
            let records = arg(Buf::Host(format!("engram.{l}.records")), DType::U8, &[t, cols, rec]);
            let emb = ws("engram.emb", act, &[t, cols * e.head_dim]);
            b.push(Op::EngramGather, format!("{p}.engram.embed"), vec![records], vec![emb.clone()], OpAttrs::default());
            if b.tp.world > 1 {
                // a rank's tier returns another rank's rows as zero payload with unit scales: exact zeros, so the sum is
                // the full gather (and `engram.wkv` stays replicated)
                b.extra(Extra::AllReduce { x: emb.clone() });
            }
            let kv = b.matmul(emb, &format!("{p}.engram.wkv"), Buf::Ws("engram.kv".into()), act)?;
            let (qw, kw) = (b.weight(&format!("{p}.engram.q_weight"))?, b.weight(&format!("{p}.engram.k_weight"))?);
            let y = flip(&x);
            b.push(Op::EngramMerge, format!("{p}.engram"), vec![x.clone(), kv, qw, kw], vec![y.clone()], OpAttrs { eps: c.norm_eps, ..Default::default() });
            x = y;
            blocks.push("engram".to_string());
        }

        // attention sub-block (Block.forward)
        // DSpark tap (Transformer.forward, model.py:1269): the MTP head reads the mean over streams of the attention input of
        // its target layers; a collapse with the constant mix 1/S. Recorded so the drafting lane only consumes it.
        if let Some(j) = c.mtp.as_ref().and_then(|m| m.target_layer_ids.iter().position(|&tl| tl == l)) {
            let mean = arg(Buf::Const("hc.mean_mix".into()), DType::F32, &[t, s]);
            // tap j is slice j of a [taps, t, dim] buffer; the absorb stage reads it as [t, taps * dim]
            b.push(Op::HcPre, format!("{p}.dspark_tap"), vec![x.clone(), mean], vec![ws(&format!("dspark.taps[{j}]"), act, &[t, d])], OpAttrs::default());
            blocks.push("dspark_tap".to_string());
        }
        let (n, [attn_pre, attn_post, attn_comb], x1) = b.hc_sublayer(&x, &format!("{p}.hc_attn"), "hc_attn", "attn", pre_mix.clone(), &format!("{p}.attn_norm.weight"), pending.take())?;
        x = x1;
        let attn_out = attention(&mut b, l, &lslots[l], &n, &mut blocks)?;
        pending = Some(PendingPost { out: attn_out, post: attn_post, comb: attn_comb, what: format!("{p}.hc_post(attn)") });
        if !fold {
            x = b.flush_post(&x, &mut pending);
        }

        // MoE sub-block
        let (n2, [ffn_pre, ffn_post, ffn_comb], x2) = b.hc_sublayer(&x, &format!("{p}.hc_ffn"), "hc_ffn", "ffn", attn_pre, &format!("{p}.ffn_norm.weight"), pending.take())?;
        x = x2;
        let ffn_out = moe(&mut b, &format!("{p}.ffn"), &n2, c.n_routed_experts, c.top_k)?;
        blocks.push(format!("hc_moe({}x top{} + shared)", c.n_routed_experts, c.top_k));
        pending = Some(PendingPost { out: ffn_out, post: ffn_post, comb: ffn_comb, what: format!("{p}.hc_post(ffn)") });
        if !fold || reads_streams(l + 1) {
            x = b.flush_post(&x, &mut pending);
        }
        // the next layer's attention collapses with this FFN's pre-mix (written to "hc_ffn.pre", not overwritten before)
        pre_mix = ffn_pre;
        stages.push(b.stage(format!("layer {l}"), Some(l), blocks));
    }
    debug_assert!(pending.is_none());

    // DSpark absorb: this chunk's taps into every stage's window ring (DSparkAttention.forward's main_kv; the official
    // driver writes one position per drafting round, here every pass writes its rows, verify rows included: a rejected
    // row sits at a position the next drafting round does not read and the next pass rewrites)
    if absorb {
        let taps = ws("dspark.taps", act, &[n_taps, t, d]);
        let mh = ws("dspark.mh", act, &[t, n_taps, d]);
        b.push(Op::Transpose01, "dspark.taps(token-major)", vec![taps], vec![mh.clone()], OpAttrs::default());
        dspark_absorb(&mut b, Arg { shape: vec![t, n_taps * d], ..mh }, &rings)?;
        stages.push(b.stage("dspark absorb".into(), None, vec![format!("main_proj + {} stage rings", rings.len())]));
    }

    // head: collapse with the last FFN's pre-mix, norm, logits of the last row (ParallelHead); of every row in a
    // verify-sized pass (2..=VERIFY_MAX_ROWS rows)
    let hc = ws("a", act, &[t, d]);
    b.push(Op::HcPre, "hc_pre(head)", vec![x.clone(), pre_mix], vec![hc.clone()], OpAttrs::default());
    b.rmsnorm(hc, "norm.weight", "n")?;
    let rows = if k.all_rows { t } else { 1 };
    let src = if rows == 1 { Arg { buf: Buf::Ws("n.last_row".into()), dtype: act, shape: vec![1, d] } } else { ws("n", act, &[t, d]) };
    let logits = head_logits(&mut b, src, "logits")?;
    let ids = if k.all_rows {
        let ids = ws("verify.ids", DType::I32, &[rows]);
        b.push(Op::Argmax, "argmax(verify)", vec![logits.clone()], vec![ids.clone()], OpAttrs::default());
        Some(ids)
    } else {
        None
    };
    let head = b.stage("head".into(), None, vec!["hc_pre".into(), "norm".into(), if rows == 1 { "head".into() } else { format!("head x{rows}") }]);
    stages.push(head);

    Ok(Dsv41Graph { stages, slots: slot_specs, residual_streams: s, mtp: mtp_blocks(c, w), cap: o.cap, logits, ids, world: tp.world, verify: k.all_rows })
}

/// The full-vocabulary f32 logits of the rows of `x` (`[rows, dim]`) through `head`, gathered over the ranks
/// (`ParallelHead`): one row gathers straight into `{name}.full`; more rows gather rank-major and transpose.
fn head_logits(b: &mut Builder, x: Arg, name: &str) -> Result<Arg> {
    let rows = x.shape[0];
    let local = b.matmul(x, "head", Buf::Ws(name.into()), DType::F32)?;
    let w = b.tp.world;
    if w == 1 {
        return Ok(local);
    }
    let v = local.shape[1];
    let full = ws(&format!("{name}.full"), DType::F32, &[rows, v * w]);
    if rows == 1 {
        b.extra(Extra::AllGather { send: local, recv: full.clone() });
        return Ok(full);
    }
    let g = ws(&format!("{name}.gather"), DType::F32, &[w, rows, v]);
    b.extra(Extra::AllGather { send: local, recv: g.clone() });
    b.push(Op::Transpose01, format!("{name}.gather(row-major)"), vec![g], vec![Arg { shape: vec![rows, w, v], ..full.clone() }], OpAttrs::default());
    Ok(full)
}

/// The DSpark absorb of `t` rows on its own (tests and diagnosis): host input `dspark.mh` (`[t, taps * dim]`, the
/// concatenated taps) instead of the target layers' taps; writes the stage rings at positions `pos0 ..`.
pub fn build_absorb(c: &Dsv41Config, w: &dyn WeightShapes, o: GraphOpts, tp: GraphTp) -> Result<Dsv41Graph> {
    let n_taps = c.mtp.as_ref().map_or(0, |m| m.target_layer_ids.len());
    if dspark_stages(c, w) == 0 {
        return Err(Error::Unsupported("deepseek_v41: the DSpark stages are not loaded".into()));
    }
    let (slot_specs, _, rings) = slots(c, w, o.act, true);
    let mut b = Builder { c, w, o, tp, steps: Vec::new(), extras: Vec::new(), dpos: false, lane: 0 };
    let mh = arg(Buf::Host("dspark.mh".into()), o.act, &[o.t, n_taps * c.dim]);
    dspark_absorb(&mut b, mh, &rings)?;
    let st = b.stage("dspark absorb".into(), None, vec![format!("main_proj + {} stage rings", rings.len())]);
    let logits = ws("dspark.mxn", o.act, &[o.t, c.dim]);
    Ok(Dsv41Graph { stages: vec![st], slots: slot_specs, residual_streams: c.hc_mult, mtp: mtp_blocks(c, w), cap: o.cap, logits, ids: None, world: tp.world, verify: false })
}

/// `main_x = main_norm(main_proj(main_hidden))` of the chunk's rows, then per stage `i` the window KV of every row
/// (`kv_norm(wkv(main_x))`, rotated at the row's position, fp8 round trip) into stage `i`'s ring (`DSparkAttention.forward`,
/// model.py:1043-1047). `mh`: `[t, taps * dim]`.
fn dspark_absorb(b: &mut Builder, mh: Arg, rings: &[usize]) -> Result<()> {
    let c = b.c;
    let (t, act, pos0) = (b.o.t, b.o.act, b.o.pos0);
    let (hd, win) = (c.head_dim, c.window);
    let mx = b.matmul(mh, "mtp.0.main_proj", Buf::Ws("dspark.mx".into()), act)?;
    let mx = b.rmsnorm(mx, "mtp.0.main_norm.weight", "dspark.mxn")?;
    for (i, &slot) in rings.iter().enumerate() {
        let p = format!("mtp.{i}.attn");
        let l = c.n_layers + i;
        let kv = b.matmul(mx.clone(), &format!("{p}.wkv"), Buf::Ws("kv".into()), act)?;
        let kv = b.rmsnorm(kv, &format!("{p}.kv_norm.weight"), "kv_n")?;
        let kv = b.rope(Arg { shape: vec![t, 1, hd], ..kv }, l, pos0, 1, false, Buf::Ws("kv_rope".into()), &format!("{p}.rope(main_kv)"));
        b.positioned(1);
        let kv = b.qdq(Arg { shape: vec![t, hd], ..kv }, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0, Buf::Ws("kv_q".into()), &format!("{p}.act_quant(main_kv)"));
        let ring = arg(Buf::Slot { slot, plane: 0, row0: 0 }, act, &[win + ROLLBACK_MARGIN, hd]);
        b.push(Op::WindowKv, format!("{p}.absorb"), vec![kv], vec![ws("win", act, &[win + t, hd]), ring], OpAttrs { window: win, pos0, ..Default::default() });
        b.positioned(1);
    }
    Ok(())
}

/// The `[t, t]` i32 key table of a DSpark block at `pos0` (host input `dspark.idx`): row `r` lists the window rows its
/// causal window misses (`0..=r`, valid ones only) and the block rows after it (`W + r + 1 .. W + t`), -1 padded.
/// With the `AttnSparse` window part (window rows `r + 1 ..= r + W` of `[W + t]`, positions `pos0 - W + j >= 0`) every
/// block row sees every valid window position and every block row once, as `get_dspark_topk_idxs` (model.py:1023).
pub fn dspark_idx(window: usize, t: usize, pos0: usize) -> Vec<i32> {
    let mut v = vec![-1i32; t * t];
    for r in 0..t {
        let mut k = 0;
        for j in 0..=r.min(window - 1) {
            if pos0 + j >= window {
                v[r * t + k] = j as i32;
                k += 1;
            }
        }
        for j in r + 1..t {
            v[r * t + k] = (window + j) as i32;
            k += 1;
        }
    }
    v
}

/// The graph of one DSpark drafting round (`Transformer.forward_spec`, start_pos > 0): the block of `t = block_size`
/// rows (`[first id, noise, ..]`, host `token_ids`) at positions `pos0 = start_pos + 1 ..`, the stages, the head's
/// logits of every row (`logits`, `[t, vocab]`), the head input `dspark.hx` (`[t, dim]`, for the confidence head).
/// With `steps > 0` the Markov loop follows the head in the same graph (`steps` steps of `DSparkMarkovHead.forward`,
/// greedy: the chosen ids come out as `ids`, the embeddings of the ids fed in as `dspark.markov_e`); the confidence runs
/// on the host after it. Reads (and harmlessly writes positions `pos0 ..`, see [`dspark_absorb`]) the stage rings;
/// appends nothing to the sequence.
pub fn build_draft(c: &Dsv41Config, w: &dyn WeightShapes, o: GraphOpts, tp: GraphTp, steps: usize) -> Result<Dsv41Graph> {
    draft_graph(c, w, o, tp, false, steps)
}

/// [`build_draft`] as a device-position graph (CUDA graph replay of the drafting round): the ropes and the window write
/// read the block's first position from the device; the attention is built at position `window`, which serves every
/// round at a position of at least `window` (its window rows are all valid there, and [`dspark_idx`] no longer depends
/// on the position). Rounds at earlier positions run the host graph.
pub fn build_draft_dpos(c: &Dsv41Config, w: &dyn WeightShapes, o: GraphOpts, tp: GraphTp, steps: usize) -> Result<Dsv41Graph> {
    draft_graph(c, w, GraphOpts { pos0: c.window, ..o }, tp, true, steps)
}

fn draft_graph(c: &Dsv41Config, w: &dyn WeightShapes, o: GraphOpts, tp: GraphTp, dpos: bool, steps: usize) -> Result<Dsv41Graph> {
    let m = c.mtp.as_ref().ok_or_else(|| Error::model("deepseek_v41: no DSpark config"))?;
    let n_st = dspark_stages(c, w);
    if n_st == 0 {
        return Err(Error::Unsupported("deepseek_v41: the DSpark stages are not loaded (rank cache without its mtp part)".into()));
    }
    let (t, d, s, act) = (o.t, c.dim, c.hc_mult, o.act);
    if t != m.block_size || o.pos0 == 0 {
        return Err(Error::invalid(format!("deepseek_v41 draft graph: {t} rows at {} (a block is {} rows after position 0)", o.pos0, m.block_size)));
    }
    if tp.world == 0 || c.n_heads % tp.world != 0 || c.o_groups % tp.world != 0 {
        return Err(Error::invalid(format!("deepseek_v41 draft graph: {} heads / {} groups do not split over {} ranks", c.n_heads, c.o_groups, tp.world)));
    }
    let (slot_specs, _, rings) = slots(c, w, act, true);
    let mut stages = Vec::new();
    let mut b = Builder { c, w, o, tp, steps: Vec::new(), extras: Vec::new(), dpos, lane: 0 };
    let ids = arg(Buf::Host("token_ids".into()), DType::I32, &[t]);
    let emb = b.weight("embed.weight")?;
    let h = ws("h", act, &[t, d]);
    b.push(Op::Embedding, "dspark.embed", vec![ids, emb], vec![h.clone()], OpAttrs::default());
    let mut x = ws("x0", act, &[t, s, d]);
    b.push(Op::HcExpand, "dspark.hc_expand", vec![h], vec![x.clone()], OpAttrs::default());
    stages.push(b.stage("dspark embed".into(), None, vec!["embed".into(), format!("hc_expand x{s}")]));
    let mut pre_mix = arg(Buf::Const("hc.one_hot_pre_mix".into()), DType::F32, &[t, s]);
    let fold = hc_fused() && t <= HC_FOLD_MAX_ROWS;
    let mut pending: Option<PendingPost> = None;
    for (i, &ring) in rings.iter().enumerate() {
        let p = format!("mtp.{i}");
        let mut blocks = Vec::new();
        let (n, [attn_pre, attn_post, attn_comb], x1) = b.hc_sublayer(&x, &format!("{p}.hc_attn"), "hc_attn", "attn", pre_mix.clone(), &format!("{p}.attn_norm.weight"), pending.take())?;
        x = x1;
        let attn_out = dspark_attention(&mut b, i, ring, &n)?;
        blocks.push(format!("dspark attn (window {} + block {t})", c.window));
        pending = Some(PendingPost { out: attn_out, post: attn_post, comb: attn_comb, what: format!("{p}.hc_post(attn)") });
        if !fold {
            x = b.flush_post(&x, &mut pending);
        }
        let (n2, [ffn_pre, ffn_post, ffn_comb], x2) = b.hc_sublayer(&x, &format!("{p}.hc_ffn"), "hc_ffn", "ffn", attn_pre, &format!("{p}.ffn_norm.weight"), pending.take())?;
        x = x2;
        let ffn_out = moe(&mut b, &format!("{p}.ffn"), &n2, m.n_routed_experts, m.top_k)?;
        blocks.push(format!("hc_moe({}x top{} + shared)", m.n_routed_experts, m.top_k));
        pending = Some(PendingPost { out: ffn_out, post: ffn_post, comb: ffn_comb, what: format!("{p}.hc_post(ffn)") });
        if !fold || i + 1 == n_st {
            x = b.flush_post(&x, &mut pending);
        }
        pre_mix = ffn_pre;
        stages.push(b.stage(format!("dspark stage {i}"), None, blocks));
    }
    // head (DSparkBlock.forward_head): collapse, the last stage's norm, the shared head over every row
    let hx = ws("dspark.hx", act, &[t, d]);
    b.push(Op::HcPre, "dspark.hc_pre(head)", vec![x, pre_mix], vec![hx.clone()], OpAttrs::default());
    let last = n_st - 1;
    let nh = b.rmsnorm(hx, &format!("mtp.{last}.norm.weight"), "n")?;
    let logits = head_logits(&mut b, nh, "dspark.logits")?;
    stages.push(b.stage("dspark head".into(), None, vec!["hc_pre".into(), "norm".into(), format!("head x{t}")]));
    // the Markov loop (DSparkMarkovHead.forward, greedy), step i: the embedding of the previous id (step 0: the block's
    // first id), its head logits `bias`, the argmax of row i of the block logits plus the bias (an f32 add, then the first
    // strict maximum, as the host loop did). Replicated weights: the rank that reads the drafts runs it alone.
    let ids = if steps > 0 {
        let Buf::Ws(lname) = &logits.buf else { return Err(Error::model("deepseek_v41 draft graph: logits are not a workspace")) };
        let (p, v) = (format!("mtp.{last}.markov_head"), logits.shape[1]);
        let emb = b.weight(&format!("{p}.embed.weight"))?;
        let rank = emb.shape[1];
        for i in 0..steps.min(t) {
            let prev = if i == 0 { arg(Buf::Host("token_ids".into()), DType::I32, &[1]) } else { ws(&format!("dspark.markov_ids[{}]", i - 1), DType::I32, &[1]) };
            let e = ws(&format!("dspark.markov_e[{i}]"), act, &[1, rank]);
            b.push(Op::Embedding, format!("{p}.embed"), vec![prev, emb.clone()], vec![e.clone()], OpAttrs::default());
            let bias = b.matmul(e, &format!("{p}.head"), Buf::Ws("dspark.markov_bias".into()), DType::F32)?;
            let row = ws(&format!("{lname}[{i}]"), DType::F32, &[1, v]);
            let sum = ws("dspark.markov_sum", DType::F32, &[1, v]);
            b.push(Op::Add, format!("{p}.logits+bias"), vec![row, bias], vec![sum.clone()], OpAttrs::default());
            b.push(Op::Argmax, format!("{p}.argmax"), vec![sum], vec![ws(&format!("dspark.markov_ids[{i}]"), DType::I32, &[1])], OpAttrs::default());
        }
        stages.push(b.stage("dspark markov".into(), None, vec![format!("embed + head + argmax x{steps}")]));
        Some(ws("dspark.markov_ids", DType::I32, &[steps.min(t)]))
    } else {
        None
    };
    Ok(Dsv41Graph { stages, slots: slot_specs, residual_streams: s, mtp: mtp_blocks(c, w), cap: o.cap, logits, ids, world: tp.world, verify: false })
}

/// `DSparkAttention.forward` (start_pos > 0) of stage `i` on the block's normed rows `n`: q and the block's KV rotated at
/// positions `pos0 ..`, attention over the stage ring's window and the whole block (no causal mask inside it, host
/// `dspark.idx` from [`dspark_idx`]), the grouped output projection.
fn dspark_attention(b: &mut Builder, i: usize, ring: usize, n: &Arg) -> Result<Arg> {
    let c = b.c;
    let (t, act, pos0) = (b.o.t, b.o.act, b.o.pos0);
    let (h, hd, win) = (c.n_heads / b.tp.world, c.head_dim, c.window);
    let l = c.n_layers + i;
    let p = format!("mtp.{i}.attn");
    let qa = b.matmul(n.clone(), &format!("{p}.wq_a"), Buf::Ws("qa".into()), act)?;
    let qr = b.rmsnorm(qa, &format!("{p}.q_norm.weight"), "qr")?;
    let q = b.matmul(qr, &format!("{p}.wq_b"), Buf::Ws("q".into()), act)?;
    let q = b.rope(Arg { shape: vec![t, h, hd], ..q }, l, pos0, 1, false, Buf::Ws("q_rope".into()), &format!("{p}.rope(q)"));
    b.positioned(1);
    let kv = b.matmul(n.clone(), &format!("{p}.wkv"), Buf::Ws("kv".into()), act)?;
    let kv = b.rmsnorm(kv, &format!("{p}.kv_norm.weight"), "kv_n")?;
    let kv = b.rope(Arg { shape: vec![t, 1, hd], ..kv }, l, pos0, 1, false, Buf::Ws("kv_rope".into()), &format!("{p}.rope(kv)"));
    b.positioned(1);
    let kv = b.qdq(Arg { shape: vec![t, hd], ..kv }, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0, Buf::Ws("kv_q".into()), &format!("{p}.act_quant(kv)"));
    let winbuf = ws("win", act, &[win + t, hd]);
    let ringa = arg(Buf::Slot { slot: ring, plane: 0, row0: 0 }, act, &[win + ROLLBACK_MARGIN, hd]);
    b.push(Op::WindowKv, format!("{p}.window_kv"), vec![kv], vec![winbuf.clone(), ringa], OpAttrs { window: win, pos0, ..Default::default() });
    b.positioned(1);
    let idx = arg(Buf::Host("dspark.idx".into()), DType::I32, &[t, t]);
    let sink = b.weight(&format!("{p}.attn_sink"))?;
    let o = ws("o", act, &[t, h, hd]);
    // host position even in the device-position graph (built at `window`, see `build_draft_dpos`)
    b.push(Op::AttnSparse, format!("{p}.sparse_attn"), vec![q, winbuf.clone(), winbuf, idx, sink], vec![o.clone()], OpAttrs { window: win, pos0, scale: (hd as f32).powf(-0.5), ..Default::default() });
    let o = b.rope(o, l, pos0, 1, true, Buf::Ws("o_derot".into()), &format!("{p}.rope_inverse(o)"));
    b.positioned(1);
    // the fp8 round trip before wo_a only in the FP8 checkpoint's mode, as in the model's attention
    let o = if matches!(c.quant, Dsv41Quant::Fp8 { .. }) {
        b.qdq(Arg { shape: vec![t, c.o_groups / b.tp.world, h * hd * b.tp.world / c.o_groups], ..o }, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0, Buf::Ws("o_q".into()), &format!("{p}.act_quant(o)"))
    } else {
        o
    };
    let oa = wo_a(b, &p, o)?;
    b.matmul_reduced(oa, &format!("{p}.wo_b"), "blk", act)
}

/// One step of the DSpark Markov head (`DSparkMarkovHead.forward`) on the host id `dspark.markov_id`: the embedding row
/// (`dspark.markov_e`, `[1, rank]`) and its logits bias (`dspark.markov_bias`, `[1, vocab]` f32). Replicated weights:
/// any rank runs it alone (rank 0 does, between the draft pass and the verify pass).
pub fn build_markov(c: &Dsv41Config, w: &dyn WeightShapes, act: DType) -> Result<Dsv41Graph> {
    let n_st = dspark_stages(c, w);
    if n_st == 0 {
        return Err(Error::Unsupported("deepseek_v41: the DSpark stages are not loaded".into()));
    }
    let p = format!("mtp.{}.markov_head", n_st - 1);
    let o = GraphOpts { t: 1, pos0: 0, act, cap: 1 };
    let mut b = Builder { c, w, o, tp: GraphTp::default(), steps: Vec::new(), extras: Vec::new(), dpos: false, lane: 0 };
    let emb = b.weight(&format!("{p}.embed.weight"))?;
    let rank = emb.shape[1];
    let e = ws("dspark.markov_e", act, &[1, rank]);
    b.push(Op::Embedding, format!("{p}.embed"), vec![arg(Buf::Host("dspark.markov_id".into()), DType::I32, &[1]), emb], vec![e.clone()], OpAttrs::default());
    let logits = b.matmul(e, &format!("{p}.head"), Buf::Ws("dspark.markov_bias".into()), DType::F32)?;
    let st = b.stage("dspark markov".into(), None, vec!["embed".into(), "head".into()]);
    Ok(Dsv41Graph { stages: vec![st], slots: Vec::new(), residual_streams: c.hc_mult, mtp: Vec::new(), cap: 1, logits, ids: None, world: 1, verify: false })
}

/// The attention of layer `l` (`Attention.forward`); returns its `[t, dim]` output.
fn attention(b: &mut Builder, l: usize, ls: &LayerSlots, n: &Arg, blocks: &mut Vec<String>) -> Result<Arg> {
    let c = b.c;
    let (t, act, pos0) = (b.o.t, b.o.act, b.o.pos0);
    let (h, hd, win) = (c.n_heads / b.tp.world, c.head_dim, c.window);
    let p = format!("layers.{l}.attn");
    let r = c.ratio(l);

    // q: low rank, normed, rotated tail
    let qa = b.matmul(n.clone(), &format!("{p}.wq_a"), Buf::Ws("qa".into()), act)?;
    let qr = b.rmsnorm(qa, &format!("{p}.q_norm.weight"), "qr")?;
    let q = b.matmul(qr.clone(), &format!("{p}.wq_b"), Buf::Ws("q".into()), act)?;
    let q = b.rope(Arg { shape: vec![t, h, hd], ..q }, l, pos0, 1, false, Buf::Ws("q_rope".into()), &format!("{p}.rope(q)"));
    b.positioned(1);

    // sliding-window KV: one latent, normed, rotated, fp8 round trip, into the ring
    let kv = b.matmul(n.clone(), &format!("{p}.wkv"), Buf::Ws("kv".into()), act)?;
    let kv = b.rmsnorm(kv, &format!("{p}.kv_norm.weight"), "kv_n")?;
    let kv = b.rope(Arg { shape: vec![t, 1, hd], ..kv }, l, pos0, 1, false, Buf::Ws("kv_rope".into()), &format!("{p}.rope(kv)"));
    b.positioned(1);
    let kv = b.qdq(Arg { shape: vec![t, hd], ..kv }, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0, Buf::Ws("kv_q".into()), &format!("{p}.act_quant(kv)"));
    let winbuf = ws("win", act, &[win + t, hd]);
    let ring = arg(Buf::Slot { slot: ls.window, plane: 0, row0: 0 }, act, &[win + ROLLBACK_MARGIN, hd]);
    b.push(Op::WindowKv, format!("{p}.window_kv"), vec![kv], vec![winbuf.clone(), ring], OpAttrs { window: win, pos0, ..Default::default() });
    b.positioned(1);

    // device position: the shared views span the slot capacity (`nk` = its rows), `ceil(t / r)` groups always
    let dpos = b.dpos;
    let (nk, groups) = match r {
        0 => (0, 0),
        _ if dpos => (b.o.cap.div_ceil(r), t.div_ceil(r)),
        _ => compressed_rows(pos0, t, r),
    };
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
                    arg(Buf::Slot { slot: tail, plane: 0, row0: 0 }, DType::F32, &[tail_rows(r), hd]),
                    arg(Buf::Slot { slot: tail, plane: 1, row0: 0 }, DType::F32, &[tail_rows(r), hd]),
                ];
                let mut a = OpAttrs { pos0, ..Default::default() };
                a.dsv.ratio = r;
                b.push(Op::CompressPool, format!("{p}.compressor"), vec![ckv, cs], outs.clone(), a);
                b.positioned(r);
                outs[0].clone()
            } else {
                b.matmul(n.clone(), &format!("{p}.compressor.wkv"), Buf::Ws("latent".into()), act)?
            };
            if groups > 0 || dpos {
                let latent = b.rmsnorm(latent, &format!("{p}.compressor.norm.weight"), "latent_n")?;
                // index keys from the pre-rope latent (Indexer.forward, owns_k), then the latent itself
                let k = b.matmul(latent.clone(), &format!("{p}.indexer.wk"), Buf::Ws("index_k".into()), act)?;
                let k = b.rmsnorm(k, &format!("{p}.indexer.k_norm.weight"), "index_k_n")?;
                let k = b.rope(Arg { shape: vec![groups, 1, c.index_head_dim], ..k }, l, row0 * r, r, false, Buf::Ws("index_k_rope".into()), &format!("{p}.rope(index_k)"));
                b.positioned(r);
                // device position: the slot write takes the whole slot and its row `p / r` from the device
                let slot_out = |slot: usize, w: usize| if dpos { (Buf::Slot { slot, plane: 0, row0: 0 }, nk, w) } else { (Buf::Slot { slot, plane: 0, row0 }, groups, w) };
                let (ik_buf, ik_rows, ik_w) = slot_out(ik_slot, c.index_head_dim);
                b.qdq(Arg { shape: vec![groups, c.index_head_dim], ..k }, QdqFmt::Fp4E2m1, 32, QdqScale::Ue8m0, ik_buf, &format!("{p}.act_quant(index_k)"));
                b.slot_rows(ik_rows, ik_w);
                b.positioned(r);
                let lat = b.rope(Arg { shape: vec![groups, 1, hd], ..latent }, l, row0 * r, r, false, Buf::Ws("latent_rope".into()), &format!("{p}.rope(latent)"));
                b.positioned(r);
                let (cb, c_rows, c_w) = slot_out(cmp_slot, hd);
                b.qdq(Arg { shape: vec![groups, hd], ..lat }, QdqFmt::Fp4E2m1, 16, QdqScale::E4m3, cb, &format!("{p}.act_quant(latent)"));
                b.slot_rows(c_rows, c_w);
                b.positioned(r);
            }
            desc.push_str(&format!(" + compressor(ratio {r})"));
        }
        let kk = c.index_topk.min(nk);
        if isrc == l && (nk > 0 || dpos) {
            // the indexer (Indexer.forward): fp4 query heads against the shared index keys
            let iq = b.matmul(qr, &format!("{p}.indexer.wq_b"), Buf::Ws("index_q".into()), act)?;
            let iq = b.rope(Arg { shape: vec![t, c.index_n_heads, c.index_head_dim], ..iq }, l, pos0, 1, false, Buf::Ws("index_q_rope".into()), &format!("{p}.rope(index_q)"));
            b.positioned(1);
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
                b.positioned(r);
                desc.push_str(&format!(" + candidate blocks top{}", c.candidate_topk_blocks));
            }
            let cand = if c.uses_candidates(l) { ws("candidates", DType::U8, &[t, nb]) } else { arg(Buf::Const("empty_u8".into()), DType::U8, &[0]) };
            let idx = ws(&format!("topk_idxs.{l}"), DType::I32, &[t, kk]);
            b.push(Op::IndexerTopk, format!("{p}.indexer"), vec![iq, keys, iw, cand], vec![idx], OpAttrs { top_k: c.index_topk, ..a });
            b.positioned(r);
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
    b.positioned(r);
    let o = b.rope(o, l, pos0, 1, true, Buf::Ws("o_derot".into()), &format!("{p}.rope_inverse(o)"));
    b.positioned(1);

    // grouped low-rank output: fp8 round trip (FP8 base checkpoint only), per-group projection, then wo_b. The official code
    // quantises `o` before `wo_a` only in its FP8 mode (`model.py:785`); the EXL3 checkpoint's reference engine (TensorFold)
    // and the committed fixtures (`tools/dsv41/fixtures.py --quant tf`) do not, and the fp32 layer-0 `o_lora` then
    // disagrees with the fixture by 2.5e-2 instead of 2.5e-6 (`crates/intarsia-models/tests/dsv41_layers.rs`).
    let g = c.o_groups / b.tp.world;
    let gk = h * hd / g;
    let o = Arg { shape: vec![t, g, gk], ..o };
    let o = if matches!(c.quant, Dsv41Quant::Fp8 { .. }) {
        b.qdq(o, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0, Buf::Ws("o_q".into()), &format!("{p}.act_quant(o)"))
    } else {
        o
    };
    let oa = wo_a(b, &p, o)?;
    b.matmul_reduced(oa, &format!("{p}.wo_b"), "blk", act)
}

/// The grouped low-rank output projection of attention `p` on `o` (`[t, h, hd]`, de-rotated): per-group EXL3 slices (Mia)
/// or the dense block-diagonal `wo_a`; returns `[t, groups * o_lora_rank]`.
fn wo_a(b: &mut Builder, p: &str, o: Arg) -> Result<Arg> {
    let c = b.c;
    let (t, act) = (b.o.t, b.o.act);
    let (h, hd) = (c.n_heads / b.tp.world, c.head_dim);
    let g = c.o_groups / b.tp.world;
    let gk = h * hd / g;
    let o = Arg { shape: vec![t, g, gk], ..o };
    let oa = if b.has(&format!("{p}.wo_a.slice.0.trellis")) {
        // EXL3 (Mia): one matmul per group slice on group-major rows
        let og = ws("o_groups", act, &[g, t, gk]);
        b.push(Op::Transpose01, format!("{p}.wo_a(group-major)"), vec![o], vec![og], OpAttrs::default());
        // the slices read and write buffers of their own: lanes 1..=4 (side by side when the runtime forks)
        for gi in 0..g {
            let x = Arg { buf: Buf::Ws(format!("o_groups[{gi}]")), dtype: act, shape: vec![t, gk] };
            b.lane = (gi % 4 + 1) as u8;
            let r = b.matmul(x, &format!("{p}.wo_a.slice.{gi}"), Buf::Ws(format!("oa_groups[{gi}]")), act);
            b.lane = 0;
            r?;
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
    Ok(oa)
}

/// The MoE (`MoE.forward`): router, then routed EXL3 experts and the shared expert in ONE `MoeExperts` launch (the
/// shared expert rides as expert `E` of an `E + 1` arena, picked by an extra slot of weight 1 in every row, as
/// TensorFold's served decode does; `quant::exl3_moe` docs). Returns `[t, dim]` (summed over the ranks under tensor
/// parallelism).
///
/// The router writes `[t, k]`; the experts read `[t, k + 1]` whose last column is `(E, 1.0)`. One row (decode): the router
/// writes the first `k` entries of the `[1, k + 1]` table and a fill sets the last. More rows: the router's `[t, k]` is
/// transposed to the first `k` rows of a `[k + 1, t]` table, a fill sets its last row, and a transpose turns it back.
///
/// The arena / scales / meta are built at load from the per-expert tensors (`{p}.experts.{e}.w1` ..., a full
/// checkpoint) or from a rank cache's routed arena (`{p}.experts.arena`, `.scales`, `.meta`, `.codebook`), with the
/// shared expert's tensors (`{p}.shared_experts.w1/w3/w2`) appended (`dsv41::graph::fused_expert_shapes`).
fn moe(b: &mut Builder, p: &str, n: &Arg, experts: usize, top_k: usize) -> Result<Arg> {
    let c = b.c;
    let (t, d, act) = (b.o.t, c.dim, b.o.act);
    let logits = b.matmul(n.clone(), &format!("{p}.gate"), Buf::Ws("router_logits".into()), DType::F32)?;
    let bias = b.weight(&format!("{p}.gate.bias"))?;
    let bias = Arg { dtype: DType::F32, ..bias };
    let f = fused_expert_shapes(b.w, p, d, experts)?;
    let k1 = top_k + 1;
    let route_attrs = OpAttrs { top_k, scale: c.route_scale, norm_topk: c.norm_topk, ..Default::default() };
    let (ids7, w7) = (ws("route_ids7", DType::I32, &[t, k1]), ws("route_w7", DType::F32, &[t, k1]));
    if t == 1 {
        let (rw, ids) = (ws("route_w7", DType::F32, &[1, top_k]), ws("route_ids7", DType::I32, &[1, top_k]));
        b.push(Op::MoeRoute, format!("{p}.gate"), vec![logits, bias], vec![rw, ids], route_attrs);
        b.extra(Extra::Fill { buf: "route_ids7".into(), dtype: DType::I32, offset: top_k, count: 1, value: experts as f64 });
        b.extra(Extra::Fill { buf: "route_w7".into(), dtype: DType::F32, offset: top_k, count: 1, value: 1.0 });
    } else {
        let (rw, ids) = (ws("route_w", DType::F32, &[t, top_k]), ws("route_ids", DType::I32, &[t, top_k]));
        b.push(Op::MoeRoute, format!("{p}.gate"), vec![logits, bias], vec![rw.clone(), ids.clone()], route_attrs);
        for (src, tbl, dst, dt, v) in [(ids, "route_ids_t", ids7.clone(), DType::I32, experts as f64), (rw, "route_w_t", w7.clone(), DType::F32, 1.0)] {
            b.push(Op::Transpose01, format!("{p}.shared_slot({tbl})"), vec![Arg { shape: vec![t, top_k, 1], ..src }], vec![ws(tbl, dt, &[top_k, t, 1])], OpAttrs::default());
            b.extra(Extra::Fill { buf: tbl.into(), dtype: dt, offset: top_k * t, count: t, value: v });
            b.push(Op::Transpose01, format!("{p}.shared_slot({tbl})^T"), vec![ws(tbl, dt, &[k1, t, 1])], vec![Arg { shape: vec![t, k1, 1], ..dst }], OpAttrs::default());
        }
    }
    let inputs = vec![
        n.clone(),
        ids7,
        w7,
        arg(Buf::Weight(format!("{p}.experts.arena")), DType::I16, &[f.words]),
        arg(Buf::Weight(format!("{p}.experts.scales")), DType::F16, &[experts + 1, 3 * d + 3 * f.inter]),
        arg(Buf::Weight(format!("{p}.experts.meta")), DType::I32, &[experts + 1, crate::quant::exl3_moe::META_COLS]),
        arg(Buf::Weight(format!("{p}.experts.codebook")), DType::I32, &[1]),
        arg(Buf::Const("swiglu_limit".into()), DType::F32, &[1]),
    ];
    // routed + shared in one launch; under tensor parallelism each rank's output is its partial sum (f32, rounded once)
    let odt = if b.tp.world > 1 { DType::F32 } else { act };
    let out_name = if b.tp.world > 1 { "blk.partial" } else { "blk" };
    let routed = ws(out_name, odt, &[t, d]);
    b.push(Op::MoeExperts, format!("{p}.experts"), inputs, vec![routed.clone()], OpAttrs { top_k: k1, ..Default::default() });
    if b.tp.world == 1 {
        return Ok(routed);
    }
    let out = ws("blk", act, &[t, d]);
    b.extra(Extra::AllReduceInto { partial: routed, out: out.clone() });
    Ok(out)
}

/// Sizes of the fused routed + shared expert tensors of MoE `p` (`layers.{l}.ffn`) as the graph reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedExperts {
    /// i16 words of the arena (routed experts, then the shared expert's gate, up, down tiles).
    pub words: usize,
    /// i16 words of the routed part (the shared expert's tiles start there).
    pub routed_words: usize,
    /// The (rank's) intermediate width.
    pub inter: usize,
}

/// [`FusedExperts`] from the checkpoint's per-expert trellis shapes or a rank cache's routed arena, plus the shared
/// expert's trellis shapes; checks the shared expert has the routed experts' width.
pub fn fused_expert_shapes(w: &dyn WeightShapes, p: &str, d: usize, experts: usize) -> Result<FusedExperts> {
    let tiles = |name: &str| -> Result<(usize, usize, usize)> {
        let (_, s) = w.get(&format!("{name}.trellis")).ok_or_else(|| Error::model(format!("deepseek_v41: no EXL3 tensor {name}.trellis")))?;
        if s.len() != 3 {
            return Err(Error::model(format!("deepseek_v41: {name}.trellis has shape {s:?}")));
        }
        Ok((16 * s[0], 16 * s[1], s.iter().product()))
    };
    let (routed_words, inter) = if let Some((_, a)) = w.get(&format!("{p}.experts.arena")) {
        // a rank cache: the routed arena is built, its width is in the scales row
        let (_, sc) = w.get(&format!("{p}.experts.scales")).ok_or_else(|| Error::model(format!("deepseek_v41: {p}.experts.arena without .scales")))?;
        if sc.len() != 2 || sc[0] != experts || sc[1] < 3 * d || (sc[1] - 3 * d) % 3 != 0 {
            return Err(Error::model(format!("deepseek_v41: {p}.experts.scales has shape {sc:?} for {experts} experts of dim {d}")));
        }
        (a.iter().product::<usize>(), (sc[1] - 3 * d) / 3)
    } else {
        let mut words = 0usize;
        let mut inter = 0usize;
        for wn in ["w1", "w3", "w2"] {
            let e0 = format!("{p}.experts.0.{wn}");
            let last = format!("{p}.experts.{}.{wn}", experts - 1);
            if w.get(&format!("{e0}.trellis")).is_none() {
                return Err(Error::Unsupported(format!(
                    "deepseek_v41: routed experts {p}.experts.*.{wn} are not EXL3 (the base's FP4 experts run as the MXFP4 MoeExperts with a clamp, lane 1b)"
                )));
            }
            let (k, nn, n0) = tiles(&e0)?;
            let (kl, nl, nlast) = tiles(&last)?;
            if (k, nn, n0) != (kl, nl, nlast) {
                return Err(Error::model(format!("deepseek_v41: {p} experts 0 and {} differ in shape", experts - 1)));
            }
            let want = if wn == "w2" { (inter, d) } else { (d, nn) };
            if wn != "w2" {
                inter = nn;
            }
            if (k, nn) != want {
                return Err(Error::model(format!("deepseek_v41: {p}.experts.*.{wn} is [{k} -> {nn}], expected [{} -> {}]", want.0, want.1)));
            }
            // one width per (layer, matrix) stack in the Mia checkpoint, so the arena is E times expert 0's tiles
            words += experts * n0;
        }
        (words, inter)
    };
    let mut shared = 0usize;
    for wn in ["w1", "w3", "w2"] {
        let (k, nn, n) = tiles(&format!("{p}.shared_experts.{wn}"))?;
        let want = if wn == "w2" { (inter, d) } else { (d, inter) };
        if (k, nn) != want {
            return Err(Error::model(format!("deepseek_v41: {p}.shared_experts.{wn} is [{k} -> {nn}], the fused launch needs [{} -> {}]", want.0, want.1)));
        }
        shared += n;
    }
    Ok(FusedExperts { words: routed_words + shared, routed_words, inter })
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
