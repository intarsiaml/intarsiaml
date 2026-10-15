//! The generic decoder (ARCHITECTURE 3.5 / 3.6): walks the recipe's layer
//! pattern and calls ops through the backend. Nothing here is model-family
//! specific: shapes, tensor names, norms, qk-norm, tied head and rope theta all
//! come from the recipe + `config.json`.
//!
//! Dataflow per block (pre-norm, residual stream `h`):
//! ```text
//!   n = norm(h)                      first block: RmsNorm
//!   n, h = rmsnorm_add(prev_out, h)  later blocks: fused residual add + norm
//!   prev_out = block(n)
//! ```
//! and one final `rmsnorm_add` with the model's final norm before the head.

use crate::backend::{Backend, Buffer, GraphHandle, MemKind, TensorRef};
use crate::comm::Comm;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::memplan::MemoryPlan;
use crate::ops::{Op, OpAttrs, OpCall, ShapeInfo};
use crate::quant::moe::{MoeScratch, MoeWeights, OpRunner};
use crate::quant::{Fp8Act, QTensor, QuantLoadOpts};
use crate::recipe::{LinearDims, ModelDims, Recipe};
use crate::registry::Registry;
use crate::state::{PeerState, SequenceState, SlotSpec, StateSlot};
use crate::telemetry::Telemetry;
use crate::weights::{load_tensor, load_tensor_shard, Shard, ShardAxis, WeightSource};
use std::cell::Cell;
use std::sync::{Arc, Mutex};
use std::time::Instant;

mod batch;
mod gdn_block;
mod graph_step;
use gdn_block::{ExtraWs, GdnBlock};
use graph_step::GraphState;

/// Rows of logits computed/downloaded at a time when all positions are requested.
const LOGIT_ROWS: usize = 16;

#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Upper bound on context length (tokens); clamped by measured free memory.
    pub max_seq: usize,
    /// Fraction of measured free memory the KV cache may use.
    pub kv_fraction: f32,
    /// Tokens per forward pass (prefill chunk / activation workspace rows).
    pub prefill_chunk: usize,
    pub act_dtype: Option<DType>,
    pub kv_dtype: Option<DType>,
    /// Time every Nth step per op (synchronises the device); 1 = every step; 0 = never.
    pub profile_every: u32,
    /// Bytes this process may use in total (weights + workspace + KV); the KV cache is
    /// clamped to what is left, so a fleet lease sizes the KV, not the box's free memory.
    pub mem_budget: Option<u64>,
    /// FP8 projections: activations quantised with the checkpoint's static scale (`static`, W8A8) or kept
    /// 16-bit (`none`, W8A16). `None` = the recipe's `[defaults] fp8_act`.
    pub fp8_act: Option<Fp8Act>,
    /// Replay the single-token decode step as a captured graph when the backend supports it (`--no-graphs` turns
    /// this off; profiled steps always run eagerly). Same kernels, same results as the eager path.
    pub graphs: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            max_seq: 4096,
            kv_fraction: 0.5,
            prefill_chunk: 256,
            act_dtype: None,
            kv_dtype: None,
            profile_every: 1,
            mem_budget: None,
            fp8_act: None,
            graphs: true,
        }
    }
}

struct AttnBlock {
    pre_norm: TensorRef,
    /// `[n_heads * head_dim * (1 + gated), hidden]`: with `gated` each head's rows are `q` then its output gate.
    wq: QTensor,
    wk: QTensor,
    wv: QTensor,
    wo: QTensor,
    q_norm: Option<TensorRef>,
    k_norm: Option<TensorRef>,
    slot: usize,
    /// `q` carries a per-head output gate (`attn_out * sigmoid(gate)` before `wo`).
    gated: bool,
    /// Leading values of each head that rotate (`partial_rotary_factor * head_dim`); 0 = the whole head.
    rotary_dim: usize,
}

struct FfnBlock {
    pre_norm: TensorRef,
    gate: TensorRef,
    up: TensorRef,
    down: TensorRef,
}

/// Sparse MoE FFN (router + routed experts + gated shared expert); the activation scratch is shared in the workspace.
struct MoeBlock {
    pre_norm: TensorRef,
    w: MoeWeights,
}

enum Block {
    Attn(AttnBlock),
    Gdn(GdnBlock),
    Ffn(FfnBlock),
    Moe(MoeBlock),
}

struct Workspace {
    max_tokens: usize,
    ids: Buffer,
    h: [Buffer; 2],
    n: Buffer,
    blk: Buffer,
    q: [Buffer; 2],
    k: [Buffer; 2],
    v: Buffer,
    g: Buffer,
    u: Buffer,
    a: Buffer,
    logits: Buffer,
    tok: Buffer,
    /// Tensor parallelism: f32 `[max_tokens, hidden]` partial results of the row-parallel projections (o, down),
    /// reduced into the bf16 activation buffer with a single rounding. None: bf16 partials reduced in place.
    part: Option<Buffer>,
    /// `i32 [1]`: position of the token a captured decode step processes (rewritten before every replay).
    pos: Buffer,
    /// Buffers of the gated-attention and GatedDeltaNet blocks (empty when the model has none).
    x: ExtraWs,
    /// Activations of the MoE blocks (one set, reused by every layer).
    moe: Option<MoeScratch>,
}

/// Runs the MoE block's ops through `Model::exec` (profiling, plan recording, kernel scratch).
struct ExecRunner<'a> {
    m: &'a Model,
    profile: bool,
}

impl OpRunner for ExecRunner<'_> {
    fn run(&self, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<()> {
        self.m.exec(op, inputs, outputs, attrs, self.profile)
    }
    fn run_fused(&self, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<bool> {
        self.m.try_fused(op, inputs, outputs, attrs, self.profile)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadMode {
    /// Fill the KV cache only.
    None,
    /// Logits of the last position.
    LastLogits,
    /// Greedy token of the last position (argmax on device, 4 bytes back).
    LastArgmax,
    /// Logits of every position, streamed through a callback.
    AllLogits,
}

#[derive(Debug)]
pub enum StepOut {
    None,
    Logits(Vec<f32>),
    Token(u32),
}

pub type RowCallback<'a> = &'a mut dyn FnMut(usize, &[f32]) -> Result<()>;

/// Views of the attention scratch for some rows: the two ping-pong q buffers `[t, n_heads, head_dim]`, the two k
/// buffers and the v buffer `[t, n_kv_heads, head_dim]` (see [`Model::attn_rows`]).
struct AttnBufs {
    q: [TensorRef; 2],
    k: [TensorRef; 2],
    v: TensorRef,
}

/// What a layer pass runs on: one sequence's next `t` tokens, or one decode token of each of several sequences.
#[derive(Clone, Copy)]
enum Seqs<'a> {
    One(&'a SequenceState),
    Batch(&'a [&'a SequenceState]),
}

impl AttnBufs {
    /// Row `r` of every buffer, as a one-row set.
    fn row(&self, r: usize) -> AttnBufs {
        AttnBufs {
            q: [self.q[0].rows(r, 1), self.q[1].rows(r, 1)],
            k: [self.k[0].rows(r, 1), self.k[1].rows(r, 1)],
            v: self.v.rows(r, 1),
        }
    }
}

pub struct Model {
    pub backend: Arc<dyn Backend>,
    pub registry: Registry,
    /// The model's dimensions (the checkpoint's, not one rank's).
    pub dims: ModelDims,
    /// What this rank computes with: heads, KV heads and the FFN width divided by the tensor-parallel degree
    /// (equal to `dims` without tensor parallelism). Plan shapes and the workspace follow these.
    pub ldims: ModelDims,
    pub recipe: Recipe,
    pub act: DType,
    pub kv: DType,
    pub plan: MemoryPlan,
    pub weights_bytes: u64,
    embed: TensorRef,
    final_norm: TensorRef,
    /// Only rank 0 holds the head (and runs the final norm's consumer): the other ranks stop after the last layer.
    lm_head: Option<QTensor>,
    layers: Vec<Vec<Block>>,
    slot_specs: Vec<SlotSpec>,
    ws: Workspace,
    telemetry: Arc<Telemetry>,
    profile_every: u32,
    /// Kernel scratch workspace shared by all ops (sized by plan resolution; ops run in stream order).
    scratch: Option<Buffer>,
    /// Installed by [`Model::dry_run`]: `exec` records `(op, shape, kernel)` instead of launching.
    recorder: Mutex<Option<crate::plan::Recorder>>,
    /// Result of plan resolution (kernel of every op/shape the model needs).
    pub resolved: crate::plan::ResolvedPlan,
    /// Captured decode step (CUDA graph); see `model/graph_step.rs`.
    graphs: GraphState,
    /// While a graph is being captured: the ops it records (replays count them in telemetry).
    capture_ops: Mutex<Option<Vec<Op>>>,
    /// Tensor parallelism: this rank's place in its group (rank 0 / world 1 without it).
    tp: TpRank,
    /// `INTARSIA_TP_NO_COMM=1`: skip the collectives, to time a decode step without them (the output is garbage).
    no_comm: bool,
    /// `INTARSIA_TP_TEST_BUG=scale:F[@RANK]` (test only, accuracy-gate sensitivity): before every all-reduce, rank `RANK`
    /// (default 1) multiplies its partial by `F`. The gate must fail on it (docs/accuracy.md, "TP self-golden budget").
    test_bug: Option<(usize, f32)>,
    /// Ranks 1.. of the group, each a full shard with its own backend, weights, workspace and KV (rank 0 only).
    peers: Vec<Model>,
    /// Set for the duration of a speculative verify pass ([`Model::forward_verify`]): every op call is marked
    /// `OpCall::exact_rows`, so the registry gives it a row-exact kernel (or fails). Never set otherwise.
    exact_rows: std::sync::atomic::AtomicBool,
    /// `i32 [max rows]` argmax output of a verify pass; allocated by [`Model::enable_verify`] (spec on only).
    verify_tok: Option<Buffer>,
    /// Most rows of a verify pass; 0 = speculative verify off (the default: plan and kernels as without it).
    verify_rows: usize,
    /// Most sequences of a batched decode step ([`Model::forward_batch`]); 0 = batching off (the default).
    batch_rows: usize,
    /// `i32 [BATCH_MAX_ROWS]` argmax output of a batched step; allocated by [`Model::enable_batch`].
    batch_tok: Option<Buffer>,
}

/// A rank's place in a tensor-parallel group and its handle for collectives.
#[derive(Clone)]
pub struct TpRank {
    pub rank: usize,
    pub world: usize,
    pub comm: Option<Arc<dyn Comm>>,
}

impl TpRank {
    pub fn single() -> TpRank {
        TpRank { rank: 0, world: 1, comm: None }
    }
}

/// `INTARSIA_TP_TEST_BUG=scale:F[@RANK]` -> (rank, factor). Unset or empty: no bug.
fn parse_tp_test_bug(v: Option<&str>) -> Result<Option<(usize, f32)>> {
    let Some(v) = v.map(str::trim).filter(|v| !v.is_empty()) else { return Ok(None) };
    let bad = || Error::invalid(format!("INTARSIA_TP_TEST_BUG={v} (expected scale:FACTOR[@RANK])"));
    let spec = v.strip_prefix("scale:").ok_or_else(bad)?;
    let (f, r) = match spec.split_once('@') {
        Some((f, r)) => (f, r.parse::<usize>().map_err(|_| bad())?),
        None => (spec, 1),
    };
    let f: f32 = f.parse().map_err(|_| bad())?;
    if !f.is_finite() {
        return Err(bad());
    }
    Ok(Some((r, f)))
}

fn resolve(src: &dyn WeightSource, name: &str) -> Result<String> {
    if src.info(name).is_some() {
        return Ok(name.to_string());
    }
    let w = format!("{name}.weight");
    if src.info(&w).is_some() {
        return Ok(w);
    }
    Err(Error::model(format!("tensor '{name}' (or '{name}.weight') not found in checkpoint")))
}

impl Model {
    /// Tokens per forward pass (the prefill chunk); callers split long prompts on it to stay responsive.
    pub fn prefill_chunk(&self) -> usize {
        self.ws.max_tokens
    }

    pub fn load(
        backend: Arc<dyn Backend>,
        recipe: &Recipe,
        dims: &ModelDims,
        src: &dyn WeightSource,
        opts: &LoadOptions,
        telemetry: Arc<Telemetry>,
    ) -> Result<Model> {
        Self::load_shard(backend, recipe, dims, src, opts, telemetry, TpRank::single())
    }

    /// Load rank `tp.rank` of a `tp.world`-way tensor-parallel model onto `backend`: column-parallel projections
    /// (q, k, v, gate, up) keep their rows for this rank's heads / FFN slice, row-parallel ones (o, down) keep
    /// their columns, and only that slice is uploaded. `dims` are the checkpoint's. With `TpRank::single()` this is
    /// the plain model.
    pub fn load_shard(
        backend: Arc<dyn Backend>,
        recipe: &Recipe,
        dims: &ModelDims,
        src: &dyn WeightSource,
        opts: &LoadOptions,
        telemetry: Arc<Telemetry>,
        tp: TpRank,
    ) -> Result<Model> {
        recipe.validate()?;
        let global = dims;
        let local_dims = dims.sharded(tp.world)?;
        let dims = &local_dims;
        let b: &dyn Backend = &*backend;
        let weights_bytes = Cell::new(0u64);
        let load = |tmpl: &str, layer: usize, shape: Option<&[usize]>| -> Result<TensorRef> {
            let name = resolve(src, &Recipe::expand(tmpl, layer))?;
            let t = load_tensor(b, src, &name, shape)?;
            weights_bytes.set(weights_bytes.get() + t.nbytes() as u64);
            Ok(t)
        };
        // a linear layer in whatever format the checkpoint holds it (bf16, fp8, ...), checked against `[n, k]`
        let qopts = QuantLoadOpts { fp8_act: opts.fp8_act.unwrap_or(recipe.defaults.fp8_act), shard: None };
        // `n`, `k`: the shape this rank holds (for a sharded weight, its slice of the checkpoint's `[n, k]`)
        let load_qs = |tmpl: &str, layer: usize, n: usize, k: usize, axis: Option<ShardAxis>| -> Result<QTensor> {
            let prefix = Recipe::expand(tmpl, layer);
            let shard = axis.filter(|_| tp.world > 1).map(|axis| Shard { axis, rank: tp.rank, world: tp.world });
            let q = crate::quant::load_linear(src, &prefix, b, &QuantLoadOpts { shard, ..qopts })?;
            // what the checkpoint declares (hf_quant_config.json) must match what was stored and what we implement
            if let Some(scheme) = src.quant_scheme() {
                scheme.check_linear(&prefix, q.format)?;
            }
            if (q.n, q.k) != (n, k) {
                return Err(Error::model(format!("tensor '{prefix}' is [{}, {}], expected [{n}, {k}]", q.n, q.k)));
            }
            weights_bytes.set(weights_bytes.get() + q.weight_bytes() as u64);
            Ok(q)
        };
        let load_q = |tmpl: &str, layer: usize, n: usize, k: usize| load_qs(tmpl, layer, n, k, None);
        // a dense weight `[n, k]` (this rank's shape) in whatever slice the rank keeps
        let load_sharded = |tmpl: &str, layer: usize, axis: ShardAxis, expect: [usize; 2]| -> Result<TensorRef> {
            if tp.world == 1 {
                return load(tmpl, layer, Some(&expect));
            }
            let name = resolve(src, &Recipe::expand(tmpl, layer))?;
            let t = load_tensor_shard(b, src, &name, Shard { axis, rank: tp.rank, world: tp.world })?;
            if t.shape != expect {
                return Err(Error::model(format!("tensor '{name}' shard is {:?}, expected {expect:?}", t.shape)));
            }
            weights_bytes.set(weights_bytes.get() + t.nbytes() as u64);
            Ok(t)
        };

        let embed = load(&recipe.top.embed, 0, Some(&[dims.vocab, dims.hidden]))?;
        let act = match opts.act_dtype {
            Some(d) => d,
            None if embed.dtype.is_float() && embed.dtype != DType::Fp8E4m3 && embed.dtype != DType::Fp8E5m2 => {
                embed.dtype
            }
            None => DType::Bf16,
        };
        let kv = match opts.kv_dtype {
            Some(d) => d,
            None => recipe.kv_dtype()?,
        };
        let final_norm = load(&recipe.top.final_norm, 0, Some(&[dims.hidden]))?;

        // lm_head: own tensor, or tied to the embedding table.
        // (the head may be quantised: Qwen3.6 NVFP4 ships `lm_head.{weight,weight_scale,weight_scale_2,input_scale}`)
        let lm_head = if tp.rank != 0 {
            None
        } else {
            let lh = &recipe.top.lm_head;
            let own = resolve(src, &lh.tensor).ok();
            let dense = |t: TensorRef| QTensor {
                format: crate::quant::BF16,
                n: dims.vocab,
                k: dims.hidden,
                parts: vec![t],
                act_quant: false,
            };
            Some(match (own, &lh.tied_to) {
                (Some(_), _) if !dims.tie_embeddings => load_q(&lh.tensor, 0, dims.vocab, dims.hidden)?,
                // the config says the head is its own tensor but it is not in the checkpoint (renamed, re-exported):
                // falling back to the embedding table would serve garbage logits without any error
                (None, _) if !dims.tie_embeddings => {
                    return Err(Error::model(format!(
                        "lm_head tensor '{}' (or '{}.weight') is missing from the checkpoint and the config does not tie the \
                         embeddings (tie_word_embeddings is false or absent); if the checkpoint really ties them, say so in config.json",
                        lh.tensor, lh.tensor
                    )))
                }
                (_, Some(_)) => dense(embed.clone()),
                (Some(_), None) => load_q(&lh.tensor, 0, dims.vocab, dims.hidden)?,
                (None, None) => {
                    return Err(Error::model(format!(
                        "lm_head tensor '{}' missing and the recipe has no tied_to",
                        lh.tensor
                    )))
                }
            })
        };

        let qdim = dims.n_heads * dims.head_dim;
        let kvdim = dims.n_kv_heads * dims.head_dim;
        let mut layers = Vec::new();
        let mut slot_specs = Vec::new();
        for li in 0..dims.n_layers {
            let mut blocks = Vec::new();
            for name in recipe.layer.pattern_for(li, &dims.layer_types)? {
                let pre = recipe
                    .layer
                    .norms
                    .get(&format!("pre_{name}"))
                    .ok_or_else(|| Error::model(format!("recipe layer.norms.pre_{name} missing")))?;
                match name.as_str() {
                    "attn" => {
                        let a = recipe.layer.attn.as_ref().unwrap();
                        if a.kind != "gqa" {
                            return Err(Error::Unsupported(format!(
                                "attention kind '{}' is not implemented in this build",
                                a.kind
                            )));
                        }
                        if a.rope != "default" {
                            return Err(Error::Unsupported(format!("rope variant '{}'", a.rope)));
                        }
                        // partial rotary embeddings: only the first `rotary_dim` values of each head rotate
                        let rotary_dim = if (dims.partial_rotary_factor - 1.0).abs() < 1e-6 {
                            0
                        } else {
                            let rd = (dims.head_dim as f64 * dims.partial_rotary_factor as f64).round() as usize;
                            if rd == 0 || rd % 2 != 0 || rd > dims.head_dim {
                                return Err(Error::Unsupported(format!(
                                    "partial_rotary_factor {} of head_dim {} gives rotary dim {rd} (must be even)",
                                    dims.partial_rotary_factor, dims.head_dim
                                )));
                            }
                            rd
                        };
                        let gate = usize::from(a.gated);
                        let slot = slot_specs.len();
                        slot_specs.push(SlotSpec::AttnKv {
                            n_kv_heads: dims.n_kv_heads,
                            head_dim: dims.head_dim,
                            dtype: kv,
                        });
                        blocks.push(Block::Attn(AttnBlock {
                            pre_norm: load(pre, li, Some(&[dims.hidden]))?,
                            wq: load_qs(&a.q, li, qdim * (1 + gate), dims.hidden, Some(ShardAxis::Rows))?,
                            wk: load_qs(&a.k, li, kvdim, dims.hidden, Some(ShardAxis::Rows))?,
                            wv: load_qs(&a.v, li, kvdim, dims.hidden, Some(ShardAxis::Rows))?,
                            wo: load_qs(&a.o, li, dims.hidden, qdim, Some(ShardAxis::Cols))?,
                            q_norm: a.q_norm.as_ref().map(|n| load(n, li, Some(&[dims.head_dim]))).transpose()?,
                            k_norm: a.k_norm.as_ref().map(|n| load(n, li, Some(&[dims.head_dim]))).transpose()?,
                            slot,
                            gated: a.gated,
                            rotary_dim,
                        }));
                    }
                    "gdn" => {
                        if tp.world > 1 {
                            return Err(Error::Unsupported(
                                "tensor parallelism: GatedDeltaNet blocks are not sharded in this build".into(),
                            ));
                        }
                        let g = recipe.layer.gdn.as_ref().unwrap();
                        if g.kind != "gated_delta_net" {
                            return Err(Error::Unsupported(format!(
                                "linear attention kind '{}' is not implemented in this build",
                                g.kind
                            )));
                        }
                        let lin: LinearDims = dims.linear.ok_or_else(|| {
                            Error::model("the recipe has a gdn block but config.json has no linear attention dimensions")
                        })?;
                        let (c, vdim, hv) = (lin.conv_dim(), lin.v_heads * lin.v_head_dim, lin.v_heads);
                        let slot = slot_specs.len();
                        slot_specs.push(SlotSpec::Recurrent { bytes: gdn_block::state_bytes(&lin) });
                        let conv = load(&g.conv, li, None)?;
                        if conv.numel() != c * lin.conv_kernel {
                            return Err(Error::model(format!(
                                "conv taps {:?} do not match [{c}, {}]",
                                conv.shape, lin.conv_kernel
                            )));
                        }
                        blocks.push(Block::Gdn(GdnBlock {
                            pre_norm: load(pre, li, Some(&[dims.hidden]))?,
                            in_qkv: load_q(&g.in_qkv, li, c, dims.hidden)?,
                            in_z: load_q(&g.in_z, li, vdim, dims.hidden)?,
                            in_b: load_q(&g.in_b, li, hv, dims.hidden)?,
                            in_a: load_q(&g.in_a, li, hv, dims.hidden)?,
                            conv: conv.reshaped(&[c, lin.conv_kernel]),
                            a_log: load(&g.a_log, li, Some(&[hv]))?,
                            dt_bias: load(&g.dt_bias, li, Some(&[hv]))?,
                            norm: load(&g.norm, li, Some(&[lin.v_head_dim]))?,
                            out: load_q(&g.out, li, dims.hidden, vdim)?,
                            slot,
                            lin,
                        }));
                    }
                    "moe" => {
                        if tp.world > 1 {
                            return Err(Error::Unsupported(
                                "tensor parallelism: MoE blocks are not sharded in this build".into(),
                            ));
                        }
                        let mc = dims.moe.as_ref().ok_or_else(|| {
                            Error::model("the recipe has a moe block but config.json has no MoE dimensions")
                        })?;
                        let m = recipe.layer.other.get("moe").ok_or_else(|| Error::model("recipe [layer.moe] missing"))?;
                        if m.get("kind").and_then(|v| v.as_str()) != Some("qwen3_5_moe") {
                            return Err(Error::Unsupported(format!("moe kind {:?} is not implemented in this build", m.get("kind"))));
                        }
                        let tmpl = m
                            .get("prefix")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| Error::model("recipe [layer.moe] needs a prefix"))?;
                        let w = MoeWeights::load(b, src, &Recipe::expand(tmpl, li), mc, &qopts)?;
                        weights_bytes.set(weights_bytes.get() + w.weight_bytes() as u64);
                        blocks.push(Block::Moe(MoeBlock { pre_norm: load(pre, li, Some(&[dims.hidden]))?, w }));
                    }
                    "ffn" => {
                        let f = recipe.layer.ffn.as_ref().unwrap();
                        if f.kind != "dense" || f.act != "silu_glu" {
                            return Err(Error::Unsupported(format!(
                                "ffn kind '{}' / act '{}' is not implemented in this build",
                                f.kind, f.act
                            )));
                        }
                        blocks.push(Block::Ffn(FfnBlock {
                            pre_norm: load(pre, li, Some(&[dims.hidden]))?,
                            gate: load_sharded(&f.gate, li, ShardAxis::Rows, [dims.intermediate, dims.hidden])?,
                            up: load_sharded(&f.up, li, ShardAxis::Rows, [dims.intermediate, dims.hidden])?,
                            down: load_sharded(&f.down, li, ShardAxis::Cols, [dims.hidden, dims.intermediate])?,
                        }));
                    }
                    other => {
                        return Err(Error::Unsupported(format!(
                            "block kind '{other}' is not implemented in this build"
                        )))
                    }
                }
            }
            layers.push(blocks);
        }

        // activation workspace
        // tensor parallelism: INTARSIA_TP_PARTIAL=bf16 reduces bf16 partials in place (two roundings per reduce, like
        // vLLM); the default keeps the partials in f32 and rounds once, as one unsharded matmul does
        let part_f32 = match std::env::var("INTARSIA_TP_PARTIAL").ok().as_deref() {
            None | Some("") | Some("f32") => true,
            Some("bf16") => false,
            Some(v) => return Err(Error::invalid(format!("INTARSIA_TP_PARTIAL={v} (f32 or bf16)"))),
        };
        let mt = opts.prefill_chunk.max(1);
        let esz = act.size_bytes();
        let mut ws_bytes = 0u64;
        let mut wb = |bytes: usize| -> Result<Buffer> {
            ws_bytes += bytes as u64;
            b.alloc(bytes.max(16), MemKind::Device)
        };
        let (has_gdn, has_gated) = layers.iter().flatten().fold((false, false), |(g, a), blk| match blk {
            Block::Gdn(_) => (true, a),
            Block::Attn(x) => (g, a || x.gated),
            _ => (g, a),
        });
        let x = gdn_block::alloc_extra(&mut wb, mt, esz, dims, has_gdn, has_gated)?;
        let mut moe_bytes = 0u64;
        let moe = match dims.moe.as_ref().filter(|_| layers.iter().flatten().any(|bl| matches!(bl, Block::Moe(_)))) {
            Some(mc) => {
                let si = mc.shared_inter.max(1);
                moe_bytes = (mt * (mc.experts * esz + mc.top_k * 8 + 2 * mc.hidden * esz + 3 * si * esz + 2 * esz)) as u64;
                Some(MoeScratch::new(b, mc, mt, act)?)
            }
            None => None,
        };
        let ws = Workspace {
            max_tokens: mt,
            ids: wb(mt * 4)?,
            h: [wb(mt * dims.hidden * esz)?, wb(mt * dims.hidden * esz)?],
            n: wb(mt * dims.hidden * esz)?,
            blk: wb(mt * dims.hidden * esz)?,
            q: [wb(mt * qdim * esz)?, wb(mt * qdim * esz)?],
            k: [wb(mt * kvdim * esz)?, wb(mt * kvdim * esz)?],
            v: wb(mt * kvdim * esz)?,
            g: wb(mt * dims.intermediate * esz)?,
            u: wb(mt * dims.intermediate * esz)?,
            a: wb(mt * dims.intermediate * esz)?,
            logits: wb(LOGIT_ROWS * dims.vocab * 4)?,
            tok: wb(16)?,
            part: if tp.world > 1 && part_f32 { Some(wb(mt * dims.hidden * 4)?) } else { None },
            pos: wb(16)?,
            x,
            moe,
        };
        // recurrent state is fixed-size: the live copy plus the prefix-reuse snapshots count against the budget
        let rec_bytes: u64 = slot_specs
            .iter()
            .map(|s| match s {
                SlotSpec::Recurrent { bytes } => *bytes as u64,
                _ => 0,
            })
            .sum();
        ws_bytes += moe_bytes;
        ws_bytes += rec_bytes * (1 + crate::state::RECURRENT_SNAPSHOTS as u64);

        let kv_per_tok: u64 = slot_specs
            .iter()
            .map(|s| match s {
                SlotSpec::AttnKv { n_kv_heads, head_dim, dtype } => {
                    (2 * n_kv_heads * head_dim * dtype.size_bytes()) as u64
                }
                _ => 0,
            })
            .sum();
        let plan = MemoryPlan::compute_budgeted(
            &b.info(),
            weights_bytes.get(),
            ws_bytes,
            kv_per_tok,
            opts.max_seq,
            dims.max_positions,
            opts.kv_fraction,
            opts.mem_budget,
        )?;
        let registry = Registry::new(b.name(), &b.info(), b.kernels());

        let mut model = Model {
            backend,
            registry,
            dims: global.clone(),
            ldims: dims.clone(),
            recipe: recipe.clone(),
            act,
            kv,
            plan,
            weights_bytes: weights_bytes.get(),
            embed,
            final_norm,
            lm_head,
            layers,
            slot_specs,
            ws,
            telemetry,
            profile_every: opts.profile_every,
            scratch: None,
            recorder: Mutex::new(None),
            resolved: Default::default(),
            graphs: GraphState::new(opts.graphs && tp.world == 1),
            capture_ops: Mutex::new(None),
            tp,
            no_comm: std::env::var_os("INTARSIA_TP_NO_COMM").is_some_and(|v| v != "0"),
            test_bug: parse_tp_test_bug(std::env::var("INTARSIA_TP_TEST_BUG").ok().as_deref())?,
            peers: Vec::new(),
            exact_rows: std::sync::atomic::AtomicBool::new(false),
            verify_tok: None,
            verify_rows: 0,
            batch_rows: 0,
            batch_tok: None,
        };
        // resolve the kernel of every (op, shape) now: a missing kernel is a load error
        crate::plan::resolve(&mut model)?;
        Ok(model)
    }

    /// Apply plan options (`--reference`, admission, `require_admitted`) and re-resolve. `reference`
    /// is the oracle backend for admission (the cpu backend); required when `opts.admit` is set.
    pub fn configure_plan(
        &mut self,
        opts: &crate::plan::PlanOptions,
        reference: Option<&dyn Backend>,
    ) -> Result<crate::plan::AdmitSummary> {
        self.graphs.invalidate(); // kernel choices change
        self.registry.reference_only = opts.force_reference;
        self.registry.require_admitted = false;
        let mut summary = Default::default();
        if opts.admit {
            let r = reference.ok_or_else(|| Error::invalid("plan admission needs a reference backend"))?;
            summary = crate::plan::admit(self, r, opts)?;
        }
        self.registry.require_admitted = opts.require_admitted;
        crate::plan::resolve(self)?;
        for i in 0..self.peers.len() {
            self.peers[i].registry.adopt_status(&self.registry);
            self.peers[i].graphs.invalidate();
            crate::plan::resolve(&mut self.peers[i])?;
        }
        Ok(summary)
    }

    /// Run the real forward path for `t` tokens at `pos0` with a recorder installed: every `exec`
    /// selects its kernel and is logged, nothing launches, nothing is uploaded or downloaded.
    pub fn dry_run(&self, phase: crate::plan::Phase, t: usize, pos0: usize) -> Result<crate::plan::Recorder> {
        let st = crate::plan::phantom_state(&self.slot_specs, self.plan.kv_cap_tokens, pos0);
        *self.recorder.lock().unwrap() = Some(crate::plan::Recorder { phase: Some(phase), t, pos0, ..Default::default() });
        let ids = vec![0u32; t];
        let verify = phase == crate::plan::Phase::Verify;
        let batch = phase == crate::plan::Phase::Batch;
        if verify || batch {
            self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let run = (|| -> Result<()> {
            if batch {
                // `t` sequences, each `pos0` tokens long: the real batched pass, recorded
                return self.batch_probe(t, pos0);
            }
            let hn = self.run_chunk(&st, &ids, false)?;
            if verify {
                return self.verify_head(&hn, t, false);
            }
            // the heads generation and accuracy use: argmax of the last row, logits of up to LOGIT_ROWS rows
            if self.lm_head.is_some() {
                self.head_argmax(&hn, t - 1, false)?;
                if let Some(r) = self.recorder.lock().unwrap().as_mut() {
                    r.extra = true;
                }
                self.head_logits(&hn, 0, t.min(LOGIT_ROWS), false)?;
            }
            Ok(())
        })();
        self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        let rec = self.recorder.lock().unwrap().take().unwrap_or_default();
        run?;
        Ok(rec)
    }

    /// Most rows a verify pass ([`Model::forward_verify`]) takes: one logits buffer of rows.
    pub const VERIFY_MAX_ROWS: usize = LOGIT_ROWS;

    /// Enable speculative verify passes of up to `max_rows` rows (2..=[`Model::VERIFY_MAX_ROWS`]). Call before
    /// [`Model::configure_plan`]: plan resolution then also dry-runs [`crate::plan::Phase::Verify`] passes (every row
    /// count, at the start and the end of the context), so a verify op without a row-exact kernel fails the load,
    /// admission covers the verify kernels and the scratch workspace is sized for them. Single-rank models only.
    pub fn enable_verify(&mut self, max_rows: usize) -> Result<()> {
        if !self.peers.is_empty() || self.tp.world > 1 || self.lm_head.is_none() {
            return Err(Error::Unsupported("speculative verify needs a single-rank model".into()));
        }
        let most = Self::VERIFY_MAX_ROWS.min(self.ws.max_tokens);
        if !(2..=most).contains(&max_rows) {
            return Err(Error::invalid(format!("verify rows {max_rows} outside 2..={most}")));
        }
        if self.verify_tok.is_none() {
            self.verify_tok = Some(self.backend.alloc(Self::VERIFY_MAX_ROWS * 4, MemKind::Device)?);
        }
        self.verify_rows = max_rows;
        Ok(())
    }

    /// Most rows a verify pass may have (0: speculative verify not enabled).
    pub fn verify_rows(&self) -> usize {
        self.verify_rows
    }

    /// Speculative verify pass: run `ids` (`[last token, drafts...]`, 2..=rows given to
    /// [`Model::enable_verify`]) as one forward with every op on its row-exact kernel, and return the greedy
    /// token of every row. Row `t`'s logits are bitwise those a single-token decode step at position
    /// `st.len + t` would compute, so its argmax is the token sequential greedy decode would pick there. All rows
    /// are committed; the caller truncates the rejected ones (attention reads keys `<= st.len + t` only, so rows
    /// past the accepted length are never seen and are overwritten by the next append).
    pub fn forward_verify(&self, st: &mut SequenceState, ids: &[u32]) -> Result<Vec<u32>> {
        Self::check_forward(st, ids, HeadMode::LastArgmax, false)?;
        let q = ids.len();
        let Some(tok) = &self.verify_tok else {
            return Err(Error::invalid("forward_verify before enable_verify"));
        };
        if !(2..=self.verify_rows).contains(&q) {
            return Err(Error::invalid(format!("forward_verify of {q} rows (enabled for 2..={})", self.verify_rows)));
        }
        let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
        self.exact_rows.store(true, std::sync::atomic::Ordering::Relaxed);
        let run = (|| -> Result<()> {
            let hn = self.run_chunk(st, ids, profile)?;
            self.verify_head(&hn, q, profile)
        })();
        self.exact_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        run?;
        let mut b = vec![0u8; q * 4];
        self.backend.download(tok, 0, &mut b, 0)?;
        st.commit(ids);
        self.telemetry.step();
        Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect())
    }

    /// lm_head of rows `[0, q)` into `ws.logits` (`[q, V]` f32) and their argmax into `verify_tok` (`i32 [q]`).
    fn verify_head(&self, hn: &TensorRef, q: usize, profile: bool) -> Result<()> {
        let v = self.dims.vocab;
        let tok = self.verify_tok.as_ref().ok_or_else(|| Error::invalid("verify head without its buffer"))?;
        let logits = self.view(&self.ws.logits, DType::F32, &[q, v]);
        self.proj(&hn.rows(0, q), self.head()?, &logits, profile)?;
        self.exec(Op::Argmax, vec![logits], vec![self.view(tok, DType::I32, &[q])], OpAttrs::default(), profile)
    }

    /// Make sure the scratch workspace holds at least `bytes` (0 = none needed).
    pub fn reserve_scratch(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 || self.scratch.as_ref().is_some_and(|b| b.bytes() >= bytes) {
            return Ok(());
        }
        self.graphs.invalidate(); // a captured graph holds the old scratch address
        self.scratch = None; // free the old one first
        self.scratch = Some(self.backend.alloc(bytes.max(16), MemKind::Device)?);
        self.plan.workspace_bytes += bytes as u64;
        Ok(())
    }

    fn scratch_view(&self, need: usize) -> Result<TensorRef> {
        match &self.scratch {
            Some(b) if b.bytes() >= need => Ok(TensorRef::new(b.clone(), DType::U8, &[need])),
            Some(b) => Err(Error::Memory(format!("kernel needs {need} scratch bytes but the plan reserved {}", b.bytes()))),
            None => Err(Error::Memory(format!("kernel needs {need} scratch bytes but the plan reserved none"))),
        }
    }

    /// Block kinds of every layer in plan vocabulary ("attention", "dense_ffn", ...).
    pub fn layer_kinds(&self) -> Vec<Vec<&'static str>> {
        self.layers
            .iter()
            .map(|blocks| {
                blocks
                    .iter()
                    .map(|b| match b {
                        Block::Attn(_) => "attention",
                        Block::Gdn(_) => "gated_delta_net",
                        Block::Ffn(_) => "dense_ffn",
                        Block::Moe(_) => "moe",
                    })
                    .collect()
            })
            .collect()
    }

    /// Allocate a fresh sequence state with the planned KV capacity.
    pub fn new_state(&self) -> Result<SequenceState> {
        let mut st = SequenceState::new(&*self.backend, &self.slot_specs, self.plan.kv_cap_tokens)?;
        for p in &self.peers {
            st.peers.push(PeerState {
                backend: p.backend.clone(),
                st: SequenceState::new(&*p.backend, &p.slot_specs, self.plan.kv_cap_tokens)?,
            });
        }
        Ok(st)
    }

    /// Dtype of the row-parallel partial results the collective sums ("f32" or "bf16"); None without tensor parallelism.
    pub fn tp_partials(&self) -> Option<&'static str> {
        (self.tp.world > 1).then(|| if self.ws.part.is_some() { "f32" } else { "bf16" })
    }

    /// Tensor-parallel degree (1 without it).
    pub fn tp_world(&self) -> usize {
        self.tp.world
    }

    /// Every rank's backend, rank 0 first.
    pub fn rank_backends(&self) -> Vec<Arc<dyn Backend>> {
        std::iter::once(self.backend.clone()).chain(self.peers.iter().map(|p| p.backend.clone())).collect()
    }

    /// Rank `r`'s shard (rank 0 is `self`).
    pub fn rank_model(&self, r: usize) -> Option<&Model> {
        if r == 0 {
            Some(self)
        } else {
            self.peers.get(r - 1)
        }
    }

    /// Collective counters of rank 0 (None without tensor parallelism).
    pub fn comm_stats(&self) -> Option<crate::comm::CommStats> {
        self.tp.comm.as_ref().map(|c| c.stats())
    }

    /// Name of the collective implementation ("host", "hip-p2p", ...), None without tensor parallelism.
    pub fn comm_name(&self) -> Option<String> {
        self.tp.comm.as_ref().map(|c| c.name().to_string())
    }

    /// Assemble a tensor-parallel model from its loaded shards (`shards[r]` is rank `r`): the KV capacity becomes
    /// the smallest any rank can hold (every rank must keep the same number of tokens), the peers follow rank 0's
    /// kernel selection, and the plan is re-resolved for the common capacity.
    pub fn from_shards(mut shards: Vec<Model>) -> Result<Model> {
        if shards.is_empty() {
            return Err(Error::invalid("a tensor-parallel group needs at least one shard"));
        }
        let world = shards[0].tp.world;
        if shards.len() != world || shards.iter().enumerate().any(|(i, m)| m.tp.rank != i || m.tp.world != world) {
            return Err(Error::invalid(format!("expected {world} shards in rank order")));
        }
        let cap = shards.iter().map(|m| m.plan.kv_cap_tokens).min().unwrap();
        for m in shards.iter_mut() {
            if m.plan.kv_cap_tokens > cap {
                m.plan.notes.push(format!("context clamped from {} to {cap} tokens: the smallest rank decides", m.plan.kv_cap_tokens));
                m.plan.kv_cap_tokens = cap;
                m.plan.kv_total_bytes = cap as u64 * m.plan.kv_bytes_per_token;
            }
        }
        let mut it = shards.into_iter();
        let mut primary = it.next().unwrap();
        primary.peers = it.collect();
        for p in primary.peers.iter_mut() {
            p.registry.adopt_status(&primary.registry);
        }
        crate::plan::resolve(&mut primary)?;
        for i in 0..primary.peers.len() {
            let p = &mut primary.peers[i];
            crate::plan::resolve(p)?;
        }
        Ok(primary)
    }

    /// Load a `backends.len()`-way tensor-parallel model: rank `r` on `backends[r]`, talking through `comms[r]`.
    /// Shards load one after the other (they share the admission cache and read the same mmap).
    pub fn load_tp(
        backends: Vec<Arc<dyn Backend>>,
        comms: Vec<Arc<dyn Comm>>,
        recipe: &Recipe,
        dims: &ModelDims,
        src: &dyn WeightSource,
        opts: &LoadOptions,
        telemetry: Arc<Telemetry>,
    ) -> Result<Model> {
        let world = backends.len();
        if world < 2 || comms.len() != world {
            return Err(Error::invalid(format!("tensor parallelism needs >= 2 backends and one comm each (got {world} / {})", comms.len())));
        }
        let mut shards = Vec::new();
        for (rank, (backend, comm)) in backends.into_iter().zip(comms).enumerate() {
            if comm.rank() != rank || comm.world() != world {
                return Err(Error::invalid(format!("comm {rank} is rank {} of {}", comm.rank(), comm.world())));
            }
            // peers keep their own counters: the primary's telemetry is the one the server reports
            let tel = if rank == 0 { telemetry.clone() } else { Arc::new(Telemetry::new()) };
            shards.push(Self::load_shard(backend, recipe, dims, src, opts, tel, TpRank { rank, world, comm: Some(comm) })?);
        }
        Self::from_shards(shards)
    }

    /// Run `ids` through the model starting at `st.len`, appending to the state.
    ///
    /// `on_row` is required for [`HeadMode::AllLogits`]: it receives
    /// `(absolute position, logits row)` in order.
    ///
    /// A failed forward on a model with recurrent state resets the sequence: the recurrent slots are
    /// updated in place chunk by chunk, so a pass that died halfway leaves state that matches no token prefix.
    pub fn forward(
        &self,
        st: &mut SequenceState,
        ids: &[u32],
        mode: HeadMode,
        on_row: Option<RowCallback<'_>>,
    ) -> Result<StepOut> {
        let r = if self.peers.is_empty() {
            self.forward_inner(st, ids, mode, on_row)
        } else {
            self.forward_tp(st, ids, mode, on_row)
        };
        if r.is_err() && st.has_recurrent() {
            let _ = st.reset(&*self.backend);
        }
        r
    }

    /// Rejections that must happen before any rank starts (a rank that stops early would leave the others waiting
    /// in a collective).
    fn check_forward(st: &SequenceState, ids: &[u32], mode: HeadMode, has_row_cb: bool) -> Result<()> {
        if ids.is_empty() {
            return Err(Error::BadRequest("empty token list".into()));
        }
        if st.len + ids.len() > st.cap {
            return Err(Error::BadRequest(format!(
                "sequence of {} tokens exceeds the planned context of {}",
                st.len + ids.len(),
                st.cap
            )));
        }
        if mode == HeadMode::AllLogits && !has_row_cb {
            return Err(Error::invalid("AllLogits needs a row callback"));
        }
        Ok(())
    }

    /// Tensor-parallel forward: ranks 1.. run the layers on their own threads (no head), rank 0 runs the whole
    /// pass here and returns the result. A rank that fails aborts the communicator so no other rank is left
    /// waiting.
    fn forward_tp(
        &self,
        st: &mut SequenceState,
        ids: &[u32],
        mode: HeadMode,
        on_row: Option<RowCallback<'_>>,
    ) -> Result<StepOut> {
        if st.peers.len() != self.peers.len() {
            return Err(Error::invalid("this sequence state does not belong to a tensor-parallel model (use Model::new_state)"));
        }
        Self::check_forward(st, ids, mode, on_row.is_some())?;
        // peers follow the primary's length: a pass that failed halfway can leave them ahead of it
        for p in st.peers.iter_mut() {
            if p.st.len < st.len {
                return Err(Error::backend(format!("tensor-parallel ranks diverged ({} vs {} tokens)", p.st.len, st.len)));
            }
            if p.st.len > st.len {
                p.st.truncate(st.len, &*p.backend)?;
            }
        }
        let comm0 = self.tp.comm.clone().ok_or_else(|| Error::invalid("tensor-parallel model without a communicator"))?;
        let mut peers = std::mem::take(&mut st.peers);
        let out = std::thread::scope(|sc| {
            let handles: Vec<_> = self
                .peers
                .iter()
                .zip(peers.iter_mut())
                .map(|(m, ps)| {
                    sc.spawn(move || {
                        let comm = m.tp.comm.clone();
                        // a panic must release the other ranks too
                        struct Guard(Option<Arc<dyn Comm>>, bool);
                        impl Drop for Guard {
                            fn drop(&mut self) {
                                if !self.1 {
                                    if let Some(c) = &self.0 {
                                        c.abort();
                                    }
                                }
                            }
                        }
                        let mut guard = Guard(comm, false);
                        let r = m.run_peer(&mut ps.st, ids);
                        guard.1 = r.is_ok();
                        r
                    })
                })
                .collect();
            let r0 = self.forward_inner(st, ids, mode, on_row);
            if r0.is_err() {
                comm0.abort();
            }
            let mut res = r0;
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        if res.is_ok() {
                            res = Err(e);
                        }
                    }
                    Err(_) => {
                        if res.is_ok() {
                            res = Err(Error::backend("a tensor-parallel rank panicked"));
                        }
                    }
                }
            }
            res
        });
        st.peers = peers;
        out
    }

    /// A peer rank's share of a forward pass: the layers only (rank 0 owns the head).
    fn run_peer(&self, ps: &mut SequenceState, ids: &[u32]) -> Result<()> {
        for chunk in ids.chunks(self.ws.max_tokens) {
            self.run_chunk(ps, chunk, false)?;
            ps.commit(chunk);
        }
        Ok(())
    }

    fn forward_inner(
        &self,
        st: &mut SequenceState,
        ids: &[u32],
        mode: HeadMode,
        mut on_row: Option<RowCallback<'_>>,
    ) -> Result<StepOut> {
        Self::check_forward(st, ids, mode, on_row.is_some())?;
        // single-token decode: replay the captured step (profiled steps and unsupported backends run eagerly)
        if ids.len() == 1 && matches!(mode, HeadMode::LastLogits | HeadMode::LastArgmax) && self.graphs.usable(self) {
            let profile = self.profile_every > 0
                && self.telemetry.steps_total() % (self.profile_every as u64 * graph_step::PROFILE_STRETCH) == 0;
            if !profile {
                if let Some(out) = self.graph_decode_step(st, ids[0], mode == HeadMode::LastLogits)? {
                    st.commit(ids);
                    self.telemetry.step();
                    return Ok(out);
                }
            }
        }
        let nchunks = ids.len().div_ceil(self.ws.max_tokens);
        let mut out = StepOut::None;
        for (ci, chunk) in ids.chunks(self.ws.max_tokens).enumerate() {
            let last = ci + 1 == nchunks;
            let base = st.len;
            let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
            if ids.len() == 1 && matches!(mode, HeadMode::LastLogits | HeadMode::LastArgmax) {
                self.telemetry.decode_eager_step(profile);
            }
            let hn = self.run_chunk(st, chunk, profile)?;
            match mode {
                HeadMode::None => {}
                HeadMode::LastLogits if last => {
                    out = StepOut::Logits(self.head_logits(&hn, chunk.len() - 1, 1, profile)?);
                }
                HeadMode::LastArgmax if last => {
                    out = StepOut::Token(self.head_argmax(&hn, chunk.len() - 1, profile)?);
                }
                HeadMode::AllLogits => {
                    let cb = on_row.as_mut().unwrap();
                    let mut r = 0;
                    while r < chunk.len() {
                        let c = LOGIT_ROWS.min(chunk.len() - r);
                        let flat = self.head_logits(&hn, r, c, profile)?;
                        for i in 0..c {
                            cb(base + r + i, &flat[i * self.dims.vocab..(i + 1) * self.dims.vocab])?;
                        }
                        r += c;
                    }
                }
                _ => {}
            }
            st.commit(chunk);
            self.telemetry.step();
        }
        Ok(out)
    }

    fn view(&self, b: &Buffer, dtype: DType, shape: &[usize]) -> TensorRef {
        TensorRef::new(b.clone(), dtype, shape)
    }

    fn exec(
        &self,
        op: Op,
        inputs: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
        attrs: OpAttrs,
        profile: bool,
    ) -> Result<()> {
        self.exec_pos(op, inputs, outputs, attrs, None, profile)
    }

    /// `exec` with an optional device-side position (`OpCall::pos`) for position-dependent decode ops.
    fn exec_pos(
        &self,
        op: Op,
        inputs: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
        attrs: OpAttrs,
        pos: Option<&TensorRef>,
        profile: bool,
    ) -> Result<()> {
        let mut call = OpCall::new(op, inputs, outputs, attrs);
        call.pos = pos.cloned();
        call.exact_rows = self.verifying();
        call.check()?;
        {
            let mut rec = self.recorder.lock().unwrap();
            if let Some(r) = rec.as_mut() {
                match self.registry.select(&call) {
                    Ok(k) => r.calls.push(crate::plan::PlannedCall {
                        phase: r.phase.unwrap_or(crate::plan::Phase::Decode),
                        t: r.t,
                        pos0: r.pos0,
                        kernel: k.name.clone(),
                        call,
                        extra: r.extra,
                    }),
                    Err(e) => {
                        let s = ShapeInfo::from_call(&call);
                        r.missing.push((
                            call.op,
                            format!("{e} (heads {}/{} head_dim {} kv_len {})", s.n_heads, s.n_kv_heads, s.head_dim, s.kv_len),
                        ))
                    }
                }
                return Ok(());
            }
        }
        let kernel = self.registry.select(&call)?;
        if kernel.scratch.is_some() {
            let need = kernel.scratch_bytes(&ShapeInfo::from_call(&call));
            if need > 0 {
                call.scratch = Some(self.scratch_view(need)?);
            }
        }
        if profile {
            // drain queued work first, or the previous op's (or an async upload's) time lands on this op
            self.backend.synchronize(0)?;
        }
        let t0 = Instant::now();
        self.backend.launch(&kernel.name, &call, 0)?;
        if profile {
            self.backend.synchronize(0)?;
            self.telemetry.record_op(op, Some(t0.elapsed()));
        } else if let Some(ops) = self.capture_ops.lock().unwrap().as_mut() {
            ops.push(op); // being captured: nothing ran, each replay counts these
        } else {
            self.telemetry.record_op(op, None);
        }
        Ok(())
    }

    /// Returns the normed final hidden states `[T, H]`.
    fn run_chunk(&self, st: &SequenceState, ids: &[u32], profile: bool) -> Result<TensorRef> {
        self.run_chunk_at(st, ids, profile, None, true)
    }

    /// `run_chunk` with the decode-graph hooks: `dpos` is the device-side position (attention ops read it when
    /// they run, `attrs.pos0` is then 0) and `upload_ids = false` leaves `ws.ids` as the caller wrote it.
    fn run_chunk_at(
        &self,
        st: &SequenceState,
        ids: &[u32],
        profile: bool,
        dpos: Option<&TensorRef>,
        upload_ids: bool,
    ) -> Result<TensorRef> {
        self.run_layers(Seqs::One(st), ids, profile, dpos, upload_ids)
    }

    /// The layer loop behind [`Model::run_chunk_at`]: `ids` are the `t` tokens of one sequence (`Seqs::One`) or the
    /// next token of each of `t` sequences (`Seqs::Batch`, a batched decode step: attention runs per row on the row's
    /// own cache, see `model/batch.rs`).
    fn run_layers(
        &self,
        seqs: Seqs<'_>,
        ids: &[u32],
        profile: bool,
        dpos: Option<&TensorRef>,
        upload_ids: bool,
    ) -> Result<TensorRef> {
        let t = ids.len();
        let d = &self.dims;
        let act = self.act;
        let h = d.hidden;
        let pos0 = match seqs {
            Seqs::One(st) if dpos.is_none() => st.len,
            _ => 0,
        };
        let ws = &self.ws;

        if upload_ids && !self.recorder.lock().unwrap().is_some() {
            let ids_i32: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
            self.backend.upload(&ws.ids, 0, &ids_i32, 0)?;
            for &i in ids {
                if i as usize >= d.vocab {
                    return Err(Error::BadRequest(format!("token id {i} >= vocab {}", d.vocab)));
                }
            }
        }

        let mut cur = 0usize; // which h buffer holds the residual stream
        let hv = |i: usize| self.view(&ws.h[i], act, &[t, h]);
        let nv = self.view(&ws.n, act, &[t, h]);
        let blkv = self.view(&ws.blk, act, &[t, h]);

        self.exec(
            Op::Embedding,
            vec![self.view(&ws.ids, DType::I32, &[t]), self.embed.clone()],
            vec![hv(cur)],
            OpAttrs::default(),
            profile,
        )?;

        let eps_attrs = OpAttrs { eps: d.rms_eps, w_offset: self.recipe.model.norm_offset, ..Default::default() };
        let mut pending = false;
        let norm_in = |pre: &TensorRef, pending: bool, cur: &mut usize| -> Result<()> {
            if pending {
                let next = 1 - *cur;
                self.exec(
                    Op::RmsNormAdd,
                    vec![blkv.clone(), hv(*cur), pre.clone()],
                    vec![nv.clone(), hv(next)],
                    eps_attrs.clone(),
                    profile,
                )?;
                *cur = next;
            } else {
                self.exec(Op::RmsNorm, vec![hv(*cur), pre.clone()], vec![nv.clone()], eps_attrs.clone(), profile)?;
            }
            Ok(())
        };

        for blocks in &self.layers {
            for block in blocks {
                match block {
                    Block::Attn(a) => {
                        norm_in(&a.pre_norm, pending, &mut cur)?;
                        match seqs {
                            Seqs::One(st) => self.attn_block(a, st, &nv, &blkv, t, pos0, dpos, profile)?,
                            Seqs::Batch(sts) => self.attn_block_batch(a, sts, &nv, &blkv, profile)?,
                        }
                    }
                    Block::Gdn(g) => {
                        norm_in(&g.pre_norm, pending, &mut cur)?;
                        let Seqs::One(st) = seqs else {
                            return Err(Error::Unsupported("batched decode of a recurrent block".into()));
                        };
                        self.gdn_block(g, st, &nv, &blkv, t, profile)?;
                    }
                    Block::Ffn(f) => {
                        norm_in(&f.pre_norm, pending, &mut cur)?;
                        self.ffn_block(f, &nv, &blkv, t, profile)?;
                    }
                    Block::Moe(m) => {
                        norm_in(&m.pre_norm, pending, &mut cur)?;
                        if matches!(seqs, Seqs::Batch(_)) {
                            return Err(Error::Unsupported("batched decode of a mixture-of-experts block".into()));
                        }
                        let ws = self.ws.moe.as_ref().ok_or_else(|| Error::invalid("moe block without workspace"))?;
                        m.w.forward(&ExecRunner { m: self, profile }, ws, &nv, &blkv)?;
                    }
                }
                pending = true;
            }
        }
        // final: h = h + block_out ; n = norm(h)
        norm_in(&self.final_norm, pending, &mut cur)?;
        Ok(nv)
    }

    /// Run a fused decode op (`Op::MatmulPair` ... `Op::GatedRmsnorm`) when the registry has a kernel for this exact call,
    /// returning `true`; return `false` without running anything when it has none (another backend, an unsupported shape such
    /// as a prefill chunk, a kernel that failed admission, `INTARSIA_HIP_DISABLE`), and the caller runs the ops the fused one
    /// replaces. The probe goes through `Registry::select`, the lookup `exec` itself makes, so a plan recording sees only
    /// the calls that will run. Fusion is off under tensor parallelism and with `INTARSIA_NO_FUSION=1`.
    fn try_fused(
        &self,
        op: Op,
        inputs: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
        attrs: OpAttrs,
        profile: bool,
    ) -> Result<bool> {
        let call = OpCall::new(op, inputs, outputs, attrs);
        if !self.fused_ok(&call) {
            return Ok(false);
        }
        self.exec(call.op, call.inputs, call.outputs, call.attrs, profile)?;
        Ok(true)
    }

    /// Whether `try_fused` would run `call` (fusion enabled, call valid, a kernel selectable), without running it. For a
    /// fused op whose replaced ops are spread over the layer, so the caller must decide before it emits any of them.
    fn fused_ok(&self, call: &OpCall) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if self.tp.world > 1 || *OFF.get_or_init(|| std::env::var_os("INTARSIA_NO_FUSION").is_some_and(|v| v != "0")) {
            return false;
        }
        if self.verifying() {
            let mut c = call.clone();
            c.exact_rows = true;
            return c.check().is_ok() && self.registry.select(&c).is_ok();
        }
        call.check().is_ok() && self.registry.select(call).is_ok()
    }

    /// Inside [`Model::forward_verify`].
    fn verifying(&self) -> bool {
        self.exact_rows.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `out = x @ W^T` for a linear layer in any supported weight format.
    fn proj(&self, x: &TensorRef, w: &QTensor, out: &TensorRef, profile: bool) -> Result<()> {
        let call = w.call(x.clone(), out.clone());
        self.exec(call.op, call.inputs, call.outputs, call.attrs, profile)
    }

    /// k and v read the same activation: one launch for both when the backend has the fused fp8 / bf16 pair. Returns
    /// whether it ran (then `k_out` and `v_out` hold both projections); `false` leaves both to `proj`.
    fn kv_pair(&self, a: &AttnBlock, n: &TensorRef, k_out: &TensorRef, v_out: &TensorRef, profile: bool) -> Result<bool> {
        match (a.wk.format, a.wv.format) {
            (crate::quant::FP8_E4M3, crate::quant::FP8_E4M3) => self.try_fused(
                Op::MatmulFp8Pair,
                vec![n.clone(), a.wk.parts[0].clone(), a.wk.parts[1].clone(), a.wv.parts[0].clone(), a.wv.parts[1].clone()],
                vec![k_out.clone(), v_out.clone()],
                OpAttrs::default(),
                profile,
            ),
            (crate::quant::BF16, crate::quant::BF16) => self.try_fused(
                Op::MatmulPair,
                vec![n.clone(), a.wk.parts[0].clone(), a.wv.parts[0].clone()],
                vec![k_out.clone(), v_out.clone()],
                OpAttrs::default(),
                profile,
            ),
            _ => Ok(false),
        }
    }

    /// The attention prologue as one op (`Op::QkNormRopeKv`): q-norm, rope and the k / v cache appends together, on
    /// the projected q / k / v in `bufs`. The projected q is `bufs.q[*qc]`; on `true` the normed and roped q is in
    /// `bufs.q[1 - *qc]` (`qc` flipped once, as the unfused norm + rope pair leaves it) and both caches have the new
    /// rows. On `false` nothing was emitted, so the caller runs the unfused sequence unchanged: the backend has no
    /// kernel for this exact call (CUDA / HIP never register one; a cache that is not bf16 on Metal; admission), the
    /// block has no q or k norm, or fusion is off. The decision depends on the call's shape (`t` rows of `bufs`) only.
    #[allow(clippy::too_many_arguments)]
    fn qkv_fold(
        &self,
        a: &AttnBlock,
        bufs: &AttnBufs,
        qc: &mut usize,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
        kcache: &TensorRef,
        vcache: &TensorRef,
        cap: usize,
        profile: bool,
    ) -> Result<bool> {
        let (Some(qw), Some(kw)) = (&a.q_norm, &a.k_norm) else { return Ok(false) };
        let d = &self.ldims;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let attrs = OpAttrs {
            eps: d.rms_eps,
            w_offset: self.recipe.model.norm_offset,
            pos0,
            theta: d.rope_theta,
            head_dim: hd,
            rotary_dim: a.rotary_dim,
            n_heads: nh,
            n_kv_heads: nkv,
            kv_cap: cap,
            ..Default::default()
        };
        debug_assert_eq!(bufs.q[0].shape, [t, nh, hd]);
        let mut call = OpCall::new(
            Op::QkNormRopeKv,
            vec![bufs.q[*qc].clone(), bufs.k[0].clone(), bufs.v.clone(), qw.clone(), kw.clone()],
            vec![bufs.q[1 - *qc].clone(), kcache.clone(), vcache.clone()],
            attrs,
        );
        call.pos = dpos.cloned();
        if !self.fused_ok(&call) {
            return Ok(false);
        }
        self.exec_pos(call.op, call.inputs, call.outputs, call.attrs, dpos, profile)?;
        *qc = 1 - *qc;
        Ok(true)
    }

    /// The q, k and v projections of an attention block for the `t` rows of `n`: q into `ws.q[0]` (with the output
    /// gate split off when the block has one), k into `ws.k[0]`, v into `ws.v`. Returns the index of the buffer
    /// holding q (always 0) and the gate.
    fn attn_project(&self, a: &AttnBlock, n: &TensorRef, t: usize, profile: bool) -> Result<(usize, Option<TensorRef>)> {
        let d = &self.ldims;
        let act = self.act;
        let ws = &self.ws;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let qc = 0usize;
        let gate = if a.gated {
            let (qg, gate) = self.gated_q_buffers(t);
            self.proj(n, &a.wq, &qg, profile)?;
            self.exec(
                Op::GateSplit,
                vec![qg.reshaped(&[t, nh, 2 * hd])],
                vec![self.view(&ws.q[qc], act, &[t, nh, hd]), gate.clone()],
                OpAttrs::default(),
                profile,
            )?;
            Some(gate)
        } else {
            self.proj(n, &a.wq, &self.view(&ws.q[qc], act, &[t, nh * hd]), profile)?;
            None
        };
        // k and v read the same activation: one launch for both when the backend has the fused fp8 / bf16 pair
        let (k_out, v_out) = (self.view(&ws.k[0], act, &[t, nkv * hd]), self.view(&ws.v, act, &[t, nkv * hd]));
        if !self.kv_pair(a, n, &k_out, &v_out, profile)? {
            self.proj(n, &a.wk, &k_out, profile)?;
            self.proj(n, &a.wv, &v_out, profile)?;
        }
        Ok((qc, gate))
    }

    /// Views of the attention scratch for `t` rows.
    fn attn_bufs(&self, t: usize) -> AttnBufs {
        let (d, act, ws) = (&self.ldims, self.act, &self.ws);
        let q3 = [t, d.n_heads, d.head_dim];
        let k3 = [t, d.n_kv_heads, d.head_dim];
        AttnBufs {
            q: [self.view(&ws.q[0], act, &q3), self.view(&ws.q[1], act, &q3)],
            k: [self.view(&ws.k[0], act, &k3), self.view(&ws.k[1], act, &k3)],
            v: self.view(&ws.v, act, &k3),
        }
    }

    /// Everything between the projections and the output projection, for the `t` rows in `bufs` against one
    /// sequence's cache `slot` (rows at `pos0..pos0 + t`): q-norm, rope, k-norm, rope, KV append (one fused op when
    /// the backend has it), then attention (`AttnDecode` for one row, `AttnPrefill` for more). Returns the attention
    /// output (`bufs.q[1 - qc]`) and the final `qc`. Plain decode, verify and prefill run it on all `t` rows of the
    /// block; a batch step runs it once per row on that row's slice of the scratch and that row's own cache, which is
    /// why a batched row is the solo step's row.
    #[allow(clippy::too_many_arguments)]
    fn attn_rows(
        &self,
        a: &AttnBlock,
        slot: &crate::state::AttnKvState,
        bufs: &AttnBufs,
        mut qc: usize,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
        profile: bool,
    ) -> Result<(TensorRef, usize)> {
        let d = &self.ldims;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let eps = OpAttrs { eps: d.rms_eps, w_offset: self.recipe.model.norm_offset, ..Default::default() };
        let rope = OpAttrs { pos0, theta: d.rope_theta, head_dim: hd, rotary_dim: a.rotary_dim, ..Default::default() };
        let cache_shape = [nkv, slot.cap, hd];
        let kcache = self.view(&slot.k, slot.dtype, &cache_shape);
        let vcache = self.view(&slot.v, slot.dtype, &cache_shape);
        let append = OpAttrs { pos0, kv_cap: slot.cap, n_kv_heads: nkv, head_dim: hd, ..Default::default() };
        // qk-norm + rope + kv-append as one op when the backend has it (needs both norms); else the ops one by one
        let folded = self.qkv_fold(a, bufs, &mut qc, t, pos0, dpos, &kcache, &vcache, slot.cap, profile)?;
        if !folded {
            if let Some(w) = &a.q_norm {
                self.exec(
                    Op::QkNorm,
                    vec![bufs.q[qc].clone(), w.clone()],
                    vec![bufs.q[1 - qc].clone()],
                    eps.clone(),
                    profile,
                )?;
                qc = 1 - qc;
            }
            self.exec_pos(Op::Rope, vec![bufs.q[qc].clone()], vec![bufs.q[1 - qc].clone()], rope.clone(), dpos, profile)?;
            qc = 1 - qc;

            let mut kc = 0usize;
            if let Some(w) = &a.k_norm {
                self.exec(
                    Op::QkNorm,
                    vec![bufs.k[kc].clone(), w.clone()],
                    vec![bufs.k[1 - kc].clone()],
                    eps.clone(),
                    profile,
                )?;
                kc = 1 - kc;
            }
            self.exec_pos(Op::Rope, vec![bufs.k[kc].clone()], vec![bufs.k[1 - kc].clone()], rope.clone(), dpos, profile)?;
            kc = 1 - kc;

            self.exec_pos(Op::KvAppend, vec![bufs.k[kc].clone()], vec![kcache.clone()], append.clone(), dpos, profile)?;
            self.exec_pos(Op::KvAppend, vec![bufs.v.clone()], vec![vcache.clone()], append, dpos, profile)?;
        }

        let attn_out = bufs.q[1 - qc].clone();
        let attrs = OpAttrs {
            pos0,
            scale: 1.0 / (hd as f32).sqrt(),
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            kv_cap: slot.cap,
            ..Default::default()
        };
        self.exec_pos(
            if t == 1 { Op::AttnDecode } else { Op::AttnPrefill },
            vec![bufs.q[qc].clone(), kcache, vcache],
            vec![attn_out.clone()],
            attrs,
            if t == 1 { dpos } else { None },
            profile,
        )?;
        Ok((attn_out, qc))
    }

    /// The attention block's output projection for `t` rows: the output gate (when the block has one), `wo`, and the
    /// tensor-parallel sum.
    #[allow(clippy::too_many_arguments)]
    fn attn_out_proj(
        &self,
        a: &AttnBlock,
        attn_out: &TensorRef,
        gate: Option<TensorRef>,
        qc: usize,
        out: &TensorRef,
        t: usize,
        profile: bool,
    ) -> Result<()> {
        let d = &self.ldims;
        let qshape = [t, d.n_heads * d.head_dim];
        let o_in = match gate {
            // output gate: attn_out * sigmoid(gate), into the (now free) roped-q buffer
            Some(g) => {
                let gated = self.view(&self.ws.q[qc], self.act, &qshape);
                self.exec(
                    Op::SigmoidMul,
                    vec![attn_out.reshaped(&qshape), g.reshaped(&qshape)],
                    vec![gated.clone()],
                    OpAttrs::default(),
                    profile,
                )?;
                gated
            }
            None => attn_out.reshaped(&qshape),
        };
        match &self.ws.part {
            Some(part) => {
                let part = self.view(part, DType::F32, &[t, d.hidden]);
                self.proj(&o_in, &a.wo, &part, profile)?;
                self.all_reduce(&part, out, profile)
            }
            None => {
                self.proj(&o_in, &a.wo, out, profile)?;
                self.all_reduce(out, out, profile)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_block(
        &self,
        a: &AttnBlock,
        st: &SequenceState,
        n: &TensorRef,
        out: &TensorRef,
        t: usize,
        pos0: usize,
        dpos: Option<&TensorRef>,
        profile: bool,
    ) -> Result<()> {
        let (qc, gate) = self.attn_project(a, n, t, profile)?;
        let slot = Self::attn_slot(st, a)?;
        let bufs = self.attn_bufs(t);
        let (attn_out, qc) = self.attn_rows(a, slot, &bufs, qc, t, pos0, dpos, profile)?;
        self.attn_out_proj(a, &attn_out, gate, qc, out, t, profile)
    }

    /// The KV cache slot attention block `a` reads and writes in `st`.
    fn attn_slot<'s>(st: &'s SequenceState, a: &AttnBlock) -> Result<&'s crate::state::AttnKvState> {
        match &st.slots[a.slot] {
            StateSlot::AttnKv(s) => Ok(s),
            StateSlot::Recurrent(_) => Err(Error::invalid("attention block bound to a recurrent state slot")),
        }
    }

    fn ffn_block(&self, f: &FfnBlock, n: &TensorRef, out: &TensorRef, t: usize, profile: bool) -> Result<()> {
        let act = self.act;
        let ws = &self.ws;
        let i = self.ldims.intermediate;
        let g = self.view(&ws.g, act, &[t, i]);
        let u = self.view(&ws.u, act, &[t, i]);
        let a = self.view(&ws.a, act, &[t, i]);
        // gate and up read the same activation: one launch for both when the backend has the fused bf16 pair
        let gu_fused = self.try_fused(
            Op::MatmulPair,
            vec![n.clone(), f.gate.clone(), f.up.clone()],
            vec![g.clone(), u.clone()],
            OpAttrs::default(),
            profile,
        )?;
        if !gu_fused {
            self.exec(Op::Matmul, vec![n.clone(), f.gate.clone()], vec![g.clone()], OpAttrs::default(), profile)?;
            self.exec(Op::Matmul, vec![n.clone(), f.up.clone()], vec![u.clone()], OpAttrs::default(), profile)?;
        }
        self.exec(Op::SiluGlu, vec![g, u], vec![a.clone()], OpAttrs::default(), profile)?;
        match &self.ws.part {
            Some(part) => {
                let part = self.view(part, DType::F32, &[t, self.ldims.hidden]);
                self.exec(Op::Logits, vec![a, f.down.clone()], vec![part.clone()], OpAttrs::default(), profile)?;
                self.all_reduce(&part, out, profile)
            }
            None => {
                self.exec(Op::Matmul, vec![a, f.down.clone()], vec![out.clone()], OpAttrs::default(), profile)?;
                self.all_reduce(out, out, profile)
            }
        }
    }

    /// Sum a row-parallel projection's partial results over the tensor-parallel group into `out` (in place when
    /// `partial` and `out` are the same tensor). A no-op without tensor parallelism. While the plan is recorded
    /// nothing runs: the call is counted instead (it would otherwise wait for ranks that are not running).
    fn all_reduce(&self, partial: &TensorRef, out: &TensorRef, profile: bool) -> Result<()> {
        let Some(comm) = &self.tp.comm else { return Ok(()) };
        if let Some(r) = self.recorder.lock().unwrap().as_mut() {
            r.comm.push(crate::plan::PlannedComm {
                phase: r.phase.unwrap_or(crate::plan::Phase::Decode),
                t: r.t,
                pos0: r.pos0,
                bytes: partial.nbytes(),
                extra: r.extra,
            });
            return Ok(());
        }
        if self.no_comm {
            return Ok(()); // INTARSIA_TP_NO_COMM: timing experiment, the activations are garbage
        }
        if let Some((rank, f)) = self.test_bug {
            if rank == self.tp.rank {
                self.scale_partial(partial, f)?;
            }
        }
        if profile {
            self.backend.synchronize(0)?;
        }
        let t0 = Instant::now();
        comm.all_reduce_sum_into(partial, out, 0)?;
        if profile {
            self.backend.synchronize(0)?;
            let d = t0.elapsed();
            comm.note_timed(d);
            self.telemetry.record_comm(Some(d));
        } else {
            self.telemetry.record_comm(None);
        }
        Ok(())
    }

    /// `INTARSIA_TP_TEST_BUG`: multiply a partial by `f` through the host (slow; a deliberate bug for gate sensitivity runs).
    fn scale_partial(&self, partial: &TensorRef, f: f32) -> Result<()> {
        self.backend.synchronize(0)?;
        let mut bytes = vec![0u8; partial.nbytes()];
        self.backend.download(&partial.buf, partial.offset, &mut bytes, 0)?;
        match partial.dtype {
            DType::F32 => {
                for c in bytes.chunks_exact_mut(4) {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * f;
                    c.copy_from_slice(&v.to_le_bytes());
                }
            }
            DType::Bf16 => {
                for c in bytes.chunks_exact_mut(2) {
                    let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) * f;
                    let b = v.to_bits();
                    let r = (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16; // round to nearest even
                    c.copy_from_slice(&r.to_le_bytes());
                }
            }
            d => return Err(Error::invalid(format!("INTARSIA_TP_TEST_BUG: partial dtype {d:?} not supported"))),
        }
        self.backend.upload(&partial.buf, partial.offset, &bytes, 0)?;
        self.backend.synchronize(0)
    }

    /// Logits for rows `[r0, r0+c)` of the normed hidden states, row-major f32 `[c, vocab]`.
    fn head_logits(&self, hn: &TensorRef, r0: usize, c: usize, profile: bool) -> Result<Vec<f32>> {
        let v = self.dims.vocab;
        let x = hn.rows(r0, c);
        let out = self.view(&self.ws.logits, DType::F32, &[c, v]);
        self.proj(&x, self.head()?, &out, profile)?;
        if self.recorder.lock().unwrap().is_some() {
            return Ok(Vec::new());
        }
        let mut bytes = vec![0u8; c * v * 4];
        self.backend.download(&self.ws.logits, 0, &mut bytes, 0)?;
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    /// lm_head + argmax of one row, left in `ws.logits` (`[1, V]` f32) and `ws.tok`; no host traffic.
    fn head_to_device(&self, hn: &TensorRef, row: usize, profile: bool) -> Result<()> {
        let v = self.dims.vocab;
        let x = hn.rows(row, 1);
        let logits = self.view(&self.ws.logits, DType::F32, &[1, v]);
        self.proj(&x, self.head()?, &logits, profile)?;
        let tok = self.view(&self.ws.tok, DType::I32, &[1]);
        self.exec(Op::Argmax, vec![logits], vec![tok], OpAttrs::default(), profile)
    }

    fn head_argmax(&self, hn: &TensorRef, row: usize, profile: bool) -> Result<u32> {
        self.head_to_device(hn, row, profile)?;
        if self.recorder.lock().unwrap().is_some() {
            return Ok(0);
        }
        let mut b = [0u8; 4];
        self.backend.download(&self.ws.tok, 0, &mut b, 0)?;
        Ok(i32::from_le_bytes(b) as u32)
    }

    /// The output head (rank 0 only).
    fn head(&self) -> Result<&QTensor> {
        self.lm_head.as_ref().ok_or_else(|| Error::invalid(format!("rank {} of the tensor-parallel group has no lm_head", self.tp.rank)))
    }

    /// KV cache bytes one token occupies across all attention slots.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.plan.kv_bytes_per_token
    }
}
