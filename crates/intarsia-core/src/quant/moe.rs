//! The Qwen3.5/3.6 sparse MoE block (HF `Qwen3_5MoeSparseMoeBlock`):
//!
//! ```text
//! logits  = router(x)                          bf16 [T,E]        (Matmul, plain dense weight)
//! (w, id) = topk(softmax(logits)), w /= sum(w)                   (Op::MoeTopk; norm_topk_prob)
//! routed  = sum_k w[k] * down_e(silu(gate_e(x)) * up_e(x))       (Op::MoeExperts, NVFP4 W4A16 experts)
//! shared  = shared_down(silu(shared_gate(x)) * shared_up(x))     (QTensor matmuls + SiluGlu)
//! out     = routed + sigmoid(shared_expert_gate(x)) * shared     (Op::MoeCombine)
//! ```
//!
//! Checkpoint tensors under `<prefix>` (`model.language_model.layers.{i}.mlp`): `gate.weight`,
//! `experts.{e}.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2}`, `shared_expert.{gate,up,down}_proj.*`,
//! `shared_expert_gate.weight`. Experts are stacked at load time into one buffer per matrix kind
//! (`gate_up` rows = gate rows then up rows, the load-time "gate+up concat" rewrite of ARCHITECTURE 3.6).

use crate::backend::{upload_new, Backend, MemKind, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Op, OpAttrs};
use crate::quant::{load_linear, QTensor, QuantLoadOpts, BF16, NVFP4};
use crate::weights::{WeightInfo, WeightSource};

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct MoeConfig {
    pub hidden: usize,
    pub experts: usize,
    pub top_k: usize,
    /// Routed expert width (`moe_intermediate_size`).
    pub inter: usize,
    /// Shared expert width (`shared_expert_intermediate_size`); 0 = no shared expert.
    pub shared_inter: usize,
    /// Renormalise the selected probabilities (HF `norm_topk_prob`; the Qwen3.5 MoE block always does).
    pub norm_topk: bool,
}

impl MoeConfig {
    /// From the (text) config of a Qwen3.5/3.6 MoE checkpoint.
    pub fn from_hf_config(cfg: &serde_json::Value) -> Result<MoeConfig> {
        let tc = crate::recipe::text_config(cfg);
        let u = |k: &str| {
            tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error::model(format!("config.json missing '{k}'")))
        };
        Ok(MoeConfig {
            hidden: u("hidden_size")?,
            experts: u("num_experts")?,
            top_k: u("num_experts_per_tok")?,
            inter: u("moe_intermediate_size")?,
            shared_inter: tc.get("shared_expert_intermediate_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            norm_topk: tc.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(true),
        })
    }
}

/// The shared expert and its sigmoid gate.
#[derive(Clone, Debug)]
pub struct SharedExpert {
    pub gate: QTensor,
    pub up: QTensor,
    pub down: QTensor,
    /// `shared_expert_gate`: `[1, H]` dense.
    pub gate_logit: QTensor,
}

/// Stacked routed-expert weights (NVFP4, checkpoint layout per expert).
#[derive(Clone, Debug)]
pub struct Experts {
    /// u8 `[E, 2I, H/2]`: rows `0..I` gate, `I..2I` up.
    pub gate_up_w: TensorRef,
    /// e4m3 `[E, 2I, H/16]`.
    pub gate_up_s: TensorRef,
    /// u8 `[E, H, I/2]`.
    pub down_w: TensorRef,
    /// e4m3 `[E, H, I/16]`.
    pub down_s: TensorRef,
    /// f32 `[E, 3]` = (gate weight_scale_2, up weight_scale_2, down weight_scale_2) per expert.
    pub gscales: TensorRef,
}

#[derive(Clone, Debug)]
pub struct MoeWeights {
    pub cfg: MoeConfig,
    pub router: QTensor,
    pub experts: Experts,
    pub shared: Option<SharedExpert>,
}

/// Activation workspace for [`MoeWeights::forward`], allocated once for `max_tokens` rows.
pub struct MoeScratch {
    pub max_tokens: usize,
    act: DType,
    logits: TensorRef,
    topk_w: TensorRef,
    topk_id: TensorRef,
    routed: TensorRef,
    sh_g: TensorRef,
    sh_u: TensorRef,
    sh_a: TensorRef,
    sh_out: TensorRef,
    sh_logit: TensorRef,
}

fn alloc(backend: &dyn Backend, dtype: DType, shape: &[usize]) -> Result<TensorRef> {
    let n = shape.iter().product::<usize>() * dtype.size_bytes();
    let buf = backend.alloc(n.max(16), MemKind::Device)?;
    Ok(TensorRef::new(buf, dtype, shape))
}

impl MoeScratch {
    pub fn new(backend: &dyn Backend, cfg: &MoeConfig, max_tokens: usize, act: DType) -> Result<MoeScratch> {
        let (t, h, e, k) = (max_tokens, cfg.hidden, cfg.experts, cfg.top_k);
        let si = cfg.shared_inter.max(1);
        Ok(MoeScratch {
            max_tokens,
            act,
            logits: alloc(backend, act, &[t, e])?,
            topk_w: alloc(backend, DType::F32, &[t, k])?,
            topk_id: alloc(backend, DType::I32, &[t, k])?,
            routed: alloc(backend, act, &[t, h])?,
            sh_g: alloc(backend, act, &[t, si])?,
            sh_u: alloc(backend, act, &[t, si])?,
            sh_a: alloc(backend, act, &[t, si])?,
            sh_out: alloc(backend, act, &[t, h])?,
            sh_logit: alloc(backend, act, &[t, 1])?,
        })
    }
}

/// Runs one op for the block. `Model::exec` (profiling, plan recording, scratch) implements this in the engine;
/// tests implement it with a fixed kernel name.
pub trait OpRunner {
    fn run(&self, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<()>;
    /// Run a fused op if the runner's backend has a kernel for exactly this call and return `true`; otherwise run nothing
    /// and return `false` (the caller then runs the unfused ops). The default has no fused kernels.
    fn run_fused(&self, _op: Op, _inputs: Vec<TensorRef>, _outputs: Vec<TensorRef>, _attrs: OpAttrs) -> Result<bool> {
        Ok(false)
    }
}

/// The residual add and norm in front of a block (`RmsNormAdd(blk, res, w) -> (n, sum)`), handed to the block that reads `n`
/// instead of being run before it: the block folds them into its first GEMV pair when the backend has the fused kernel
/// (`Op::RmsNormAddMatmulPair`, which also stores `n` and `sum` for the block's other readers) and otherwise runs them
/// unfused first. Either way the norm runs exactly once, before the first read of `n`.
#[derive(Clone, Debug)]
pub struct NormFold {
    /// The previous block's output (the `x` of the add).
    pub blk: TensorRef,
    /// The residual stream before this block's add.
    pub res: TensorRef,
    /// The norm weight.
    pub w: TensorRef,
    /// The residual stream after the add: the other output of the norm.
    pub sum: TensorRef,
    pub attrs: OpAttrs,
}

impl NormFold {
    /// Inputs and outputs of the fused op for the projections `w0` / `w1` into `o0` / `o1`, with the normed row in `n`.
    pub fn pair(&self, n: &TensorRef, w0: &TensorRef, w1: &TensorRef, o0: &TensorRef, o1: &TensorRef) -> (Vec<TensorRef>, Vec<TensorRef>) {
        (
            vec![self.blk.clone(), self.res.clone(), self.w.clone(), w0.clone(), w1.clone()],
            vec![n.clone(), self.sum.clone(), o0.clone(), o1.clone()],
        )
    }

    /// The unfused norm: `n` and `sum` from one `RmsNormAdd`.
    pub fn unfused(&self, n: &TensorRef) -> (Vec<TensorRef>, Vec<TensorRef>) {
        (vec![self.blk.clone(), self.res.clone(), self.w.clone()], vec![n.clone(), self.sum.clone()])
    }
}

fn view(t: &TensorRef, rows: usize) -> TensorRef {
    let mut shape = t.shape.clone();
    shape[0] = rows;
    TensorRef { shape, ..t.clone() }
}

impl MoeWeights {
    /// Load one MoE layer. `prefix` is the `mlp` module name, e.g. `model.language_model.layers.3.mlp`.
    pub fn load(
        backend: &dyn Backend,
        src: &dyn WeightSource,
        prefix: &str,
        cfg: &MoeConfig,
        opts: &QuantLoadOpts,
    ) -> Result<MoeWeights> {
        Self::load_shard(backend, src, prefix, cfg, opts, 0, 1)
    }

    /// Rank `rank` of a `world`-way tensor-parallel MoE layer (docs/design/35b-tp2.md): every expert and the shared expert
    /// keep `1 / world` of their intermediate width (gate and up rows, down columns), the router and the shared-expert gate
    /// are replicated, so `forward` produces this rank's partial sum of the block output. `cfg` is the rank's config
    /// (`inter` and `shared_inter` already divided, [`crate::recipe::ModelDims::sharded`]); `world == 1` is [`MoeWeights::load`].
    #[allow(clippy::too_many_arguments)]
    pub fn load_shard(
        backend: &dyn Backend,
        src: &dyn WeightSource,
        prefix: &str,
        cfg: &MoeConfig,
        opts: &QuantLoadOpts,
        rank: usize,
        world: usize,
    ) -> Result<MoeWeights> {
        let repl = QuantLoadOpts { shard: None, ..opts.clone() };
        let part = |axis| QuantLoadOpts { shard: (world > 1).then_some(crate::weights::Shard { axis, rank, world }), ..opts.clone() };
        let (rows, cols) = (part(crate::weights::ShardAxis::Rows), part(crate::weights::ShardAxis::Cols));
        let router = load_linear(src, &format!("{prefix}.gate"), backend, &repl)?;
        if (router.n, router.k) != (cfg.experts, cfg.hidden) {
            return Err(Error::model(format!("{prefix}.gate is [{}, {}], expected [{}, {}]", router.n, router.k, cfg.experts, cfg.hidden)));
        }
        let experts = load_experts(backend, src, prefix, cfg, rank, world)?;
        let shared = if cfg.shared_inter > 0 {
            let sp = format!("{prefix}.shared_expert");
            let gate = load_linear(src, &format!("{sp}.gate_proj"), backend, &rows)?;
            let up = load_linear(src, &format!("{sp}.up_proj"), backend, &rows)?;
            let down = load_linear(src, &format!("{sp}.down_proj"), backend, &cols)?;
            let gate_logit = load_linear(src, &format!("{prefix}.shared_expert_gate"), backend, &repl)?;
            if (gate.n, gate.k) != (cfg.shared_inter, cfg.hidden)
                || (up.n, up.k) != (cfg.shared_inter, cfg.hidden)
                || (down.n, down.k) != (cfg.hidden, cfg.shared_inter)
                || (gate_logit.n, gate_logit.k) != (1, cfg.hidden)
            {
                return Err(Error::model(format!("{sp}: shared expert shapes do not match the config {cfg:?}")));
            }
            Some(SharedExpert { gate, up, down, gate_logit })
        } else {
            None
        };
        Ok(MoeWeights { cfg: cfg.clone(), router, experts, shared })
    }

    /// Bytes of weights on the device.
    pub fn weight_bytes(&self) -> usize {
        let e = &self.experts;
        let mut n = self.router.weight_bytes()
            + e.gate_up_w.nbytes()
            + e.gate_up_s.nbytes()
            + e.down_w.nbytes()
            + e.down_s.nbytes()
            + e.gscales.nbytes();
        if let Some(s) = &self.shared {
            n += s.gate.weight_bytes() + s.up.weight_bytes() + s.down.weight_bytes() + s.gate_logit.weight_bytes();
        }
        n
    }

    /// `out[t, :] = moe(n[t, :])` for the first `t = n.shape[0]` rows (`t <= ws.max_tokens`). `out` must not alias `n`.
    pub fn forward(&self, run: &dyn OpRunner, ws: &MoeScratch, n: &TensorRef, out: &TensorRef) -> Result<()> {
        self.forward_folded(run, ws, None, n, out)
    }

    /// [`MoeWeights::forward`] where `n` is not computed yet: `fold` is the norm that produces it (`n` and the new residual
    /// stream), which this runs first, as one launch with the router and shared-gate projections when the backend has it.
    pub fn forward_folded(&self, run: &dyn OpRunner, ws: &MoeScratch, fold: Option<&NormFold>, n: &TensorRef, out: &TensorRef) -> Result<()> {
        let c = &self.cfg;
        let t = n.shape[0];
        if t == 0 || t > ws.max_tokens || n.shape != [t, c.hidden] || out.shape != [t, c.hidden] || n.dtype != ws.act {
            return Err(Error::invalid(format!(
                "moe forward: n {:?} {} out {:?} (workspace {} rows, {})",
                n.shape, n.dtype, out.shape, ws.max_tokens, ws.act
            )));
        }
        let logits = view(&ws.logits, t);
        let (tw, ti) = (view(&ws.topk_w, t), view(&ws.topk_id, t));
        let routed = view(&ws.routed, t);
        let e = &self.experts;
        let attrs = || OpAttrs { top_k: c.top_k, norm_topk: c.norm_topk, ..OpAttrs::default() };
        // the shared-expert gate logit reads the same activation as the router: one launch for both (dense bf16 pair)
        let (so, sl) = (view(&ws.sh_out, t), view(&ws.sh_logit, t));
        let mut gate_logit_done = false;
        let mut router_done = false;
        let pair = self.shared.as_ref().filter(|sh| self.router.format == BF16 && sh.gate_logit.format == BF16);
        let mut norm = fold;
        if let (Some(sh), Some(f)) = (pair, norm) {
            // the norm is the first launch of the block: fold it into the router + shared-gate pair
            let (inputs, outputs) = f.pair(n, &self.router.parts[0], &sh.gate_logit.parts[0], &logits, &sl);
            router_done = run.run_fused(Op::RmsNormAddMatmulPair, inputs, outputs, f.attrs.clone())?;
            if router_done {
                norm = None;
                gate_logit_done = true;
            }
        }
        if let Some(f) = norm {
            let (inputs, outputs) = f.unfused(n);
            run.run(Op::RmsNormAdd, inputs, outputs, f.attrs.clone())?;
        }
        if let (Some(sh), false) = (pair, router_done) {
            router_done = run.run_fused(
                Op::MatmulPair,
                vec![n.clone(), self.router.parts[0].clone(), sh.gate_logit.parts[0].clone()],
                vec![logits.clone(), sl.clone()],
                OpAttrs::default(),
            )?;
            gate_logit_done = router_done;
        }
        if !router_done {
            let call = self.router.call(n.clone(), logits.clone());
            run.run(call.op, call.inputs, call.outputs, call.attrs)?;
        }
        // P2' (docs/results/2026-10-19-p2prime-moeblock.md): everything after the router logits as one op when the backend has
        // it. Only tried when the shared-gate logit is already computed (folded into the router pair above) and the shared
        // expert is NVFP4 throughout; otherwise, or when no kernel is selectable, the unfused ops below run unchanged.
        if let Some(sh) = self.shared.as_ref().filter(|sh| {
            gate_logit_done && sh.gate.format == NVFP4 && sh.up.format == NVFP4 && sh.down.format == NVFP4
        }) {
            let mut inputs = vec![n.clone(), logits.clone(), sl.clone()];
            inputs.extend([e.gate_up_w.clone(), e.gate_up_s.clone(), e.down_w.clone(), e.down_s.clone(), e.gscales.clone()]);
            inputs.extend(sh.gate.parts.iter().chain(&sh.up.parts).chain(&sh.down.parts).cloned());
            if run.run_fused(Op::MoeBlock, inputs, vec![out.clone()], attrs())? {
                return Ok(());
            }
        }
        run.run(Op::MoeTopk, vec![logits], vec![tw.clone(), ti.clone()], attrs())?;
        run.run(
            Op::MoeExperts,
            vec![
                n.clone(),
                ti,
                tw,
                e.gate_up_w.clone(),
                e.gate_up_s.clone(),
                e.down_w.clone(),
                e.down_s.clone(),
                e.gscales.clone(),
            ],
            vec![if self.shared.is_some() { routed.clone() } else { out.clone() }],
            attrs(),
        )?;
        let Some(sh) = &self.shared else { return Ok(()) };
        let (g, u, a) = (view(&ws.sh_g, t), view(&ws.sh_u, t), view(&ws.sh_a, t));
        // shared gate + up + silu_glu in one launch when the backend has the fused nvfp4 kernel
        let glu_done = if sh.gate.format == NVFP4 && sh.up.format == NVFP4 {
            let (pg, pu) = (&sh.gate.parts, &sh.up.parts);
            run.run_fused(
                Op::MatmulNvfp4Glu,
                vec![n.clone(), pg[0].clone(), pg[1].clone(), pg[2].clone(), pu[0].clone(), pu[1].clone(), pu[2].clone()],
                vec![a.clone()],
                OpAttrs::default(),
            )?
        } else {
            false
        };
        if !glu_done {
            for (q, o) in [(&sh.gate, &g), (&sh.up, &u)] {
                let call = q.call(n.clone(), o.clone());
                run.run(call.op, call.inputs, call.outputs, call.attrs)?;
            }
            run.run(Op::SiluGlu, vec![g, u], vec![a.clone()], OpAttrs::default())?;
        }
        // the shared-expert down projection with the combine as its epilogue (`SharedDownCombine`, row-local: no wait) when the
        // backend has it. Only tried when the gate logit is already computed (folded into the router pair above), so the unfused
        // fallback below keeps today's op order exactly; `so` is then not written (nothing else reads it).
        if gate_logit_done && sh.down.format == NVFP4 {
            let pd = &sh.down.parts;
            let folded = run.run_fused(
                Op::SharedDownCombine,
                vec![a.clone(), pd[0].clone(), pd[1].clone(), pd[2].clone(), routed.clone(), sl.clone()],
                vec![out.clone()],
                OpAttrs::default(),
            )?;
            if folded {
                return Ok(());
            }
        }
        let call = sh.down.call(a, so.clone());
        run.run(call.op, call.inputs, call.outputs, call.attrs)?;
        if !gate_logit_done {
            let call = sh.gate_logit.call(n.clone(), sl.clone());
            run.run(call.op, call.inputs, call.outputs, call.attrs)?;
        }
        run.run(Op::MoeCombine, vec![routed, so, sl], vec![out.clone()], OpAttrs::default())
    }
}

fn expert_info(src: &dyn WeightSource, name: &str) -> Result<WeightInfo> {
    src.info(name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))
}

/// Stack `cfg.experts` per-expert NVFP4 tensors into the kernel layout (one upload per tensor, no host staging).
fn load_experts(backend: &dyn Backend, src: &dyn WeightSource, prefix: &str, cfg: &MoeConfig, rank: usize, world: usize) -> Result<Experts> {
    // `i`: this rank's width (the config is the rank's); `ig`: the checkpoint's (`i * world`)
    let (e, h, i) = (cfg.experts, cfg.hidden, cfg.inter);
    let ig = i * world;
    if h % 16 != 0 || i % 16 != 0 || rank >= world {
        return Err(Error::Unsupported(format!("nvfp4 experts need H and I (per rank) multiples of 16 (H={h}, I={i}, rank {rank} of {world})")));
    }
    let gu_w = alloc(backend, DType::U8, &[e, 2 * i, h / 2])?;
    let gu_s = alloc(backend, DType::Fp8E4m3, &[e, 2 * i, h / 16])?;
    let dn_w = alloc(backend, DType::U8, &[e, h, i / 2])?;
    let dn_s = alloc(backend, DType::Fp8E4m3, &[e, h, i / 16])?;
    let mut gs = vec![0u8; e * 3 * 4];
    // the checkpoint tensor `name` (checked against `shape`), or this rank's `axis` slice of it, uploaded at `off`
    let put = |dst: &TensorRef, off: usize, name: &str, shape: &[usize], dt: DType, axis: crate::weights::ShardAxis| -> Result<()> {
        let info = expert_info(src, name)?;
        if info.dtype != dt || info.shape != shape {
            return Err(Error::model(format!("tensor '{name}' is {} {:?}, expected {dt} {shape:?}", info.dtype, info.shape)));
        }
        if world == 1 {
            return backend.upload(&dst.buf, dst.offset + off, src.bytes(name)?, 0);
        }
        let sh = crate::weights::Shard { axis, rank, world };
        let part = crate::weights::shard_bytes(src.bytes(name)?, shape[0], shape[1], 1, sh, name)?;
        backend.upload(&dst.buf, dst.offset + off, &part, 0)
    };
    let (rows, cols) = (crate::weights::ShardAxis::Rows, crate::weights::ShardAxis::Cols);
    for x in 0..e {
        let p = format!("{prefix}.experts.{x}");
        let gu_row = h / 2;
        let gu_srow = h / 16;
        // gate rows then up rows of the rank's slice: `[e, 2 * i, ..]` keeps the kernel layout at the rank's width
        for (j, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
            put(&gu_w, (x * 2 * i + j * i) * gu_row, &format!("{p}.{proj}.weight"), &[ig, h / 2], DType::U8, rows)?;
            put(&gu_s, (x * 2 * i + j * i) * gu_srow, &format!("{p}.{proj}.weight_scale"), &[ig, h / 16], DType::Fp8E4m3, rows)?;
        }
        put(&dn_w, x * h * (i / 2), &format!("{p}.down_proj.weight"), &[h, ig / 2], DType::U8, cols)?;
        put(&dn_s, x * h * (i / 16), &format!("{p}.down_proj.weight_scale"), &[h, ig / 16], DType::Fp8E4m3, cols)?;
        for (j, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let name = format!("{p}.{proj}.weight_scale_2");
            let info = expert_info(src, &name)?;
            if info.dtype != DType::F32 || info.shape.iter().product::<usize>() != 1 {
                return Err(Error::model(format!("tensor '{name}' must be a single f32")));
            }
            gs[(x * 3 + j) * 4..(x * 3 + j) * 4 + 4].copy_from_slice(&src.bytes(&name)?[..4]);
        }
    }
    let gst = upload_new(backend, &gs)?;
    Ok(Experts {
        gate_up_w: gu_w,
        gate_up_s: gu_s,
        down_w: dn_w,
        down_s: dn_s,
        gscales: TensorRef::new(gst, DType::F32, &[e, 3]),
    })
}
