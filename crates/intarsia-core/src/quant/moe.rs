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
use crate::quant::{load_linear, QTensor, QuantLoadOpts};
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
        let router = load_linear(src, &format!("{prefix}.gate"), backend, opts)?;
        if (router.n, router.k) != (cfg.experts, cfg.hidden) {
            return Err(Error::model(format!("{prefix}.gate is [{}, {}], expected [{}, {}]", router.n, router.k, cfg.experts, cfg.hidden)));
        }
        let experts = load_experts(backend, src, prefix, cfg)?;
        let shared = if cfg.shared_inter > 0 {
            let sp = format!("{prefix}.shared_expert");
            let gate = load_linear(src, &format!("{sp}.gate_proj"), backend, opts)?;
            let up = load_linear(src, &format!("{sp}.up_proj"), backend, opts)?;
            let down = load_linear(src, &format!("{sp}.down_proj"), backend, opts)?;
            let gate_logit = load_linear(src, &format!("{prefix}.shared_expert_gate"), backend, opts)?;
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
        let call = self.router.call(n.clone(), logits.clone());
        run.run(call.op, call.inputs, call.outputs, call.attrs)?;
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
        let (so, sl) = (view(&ws.sh_out, t), view(&ws.sh_logit, t));
        for (q, o) in [(&sh.gate, &g), (&sh.up, &u)] {
            let call = q.call(n.clone(), o.clone());
            run.run(call.op, call.inputs, call.outputs, call.attrs)?;
        }
        run.run(Op::SiluGlu, vec![g, u], vec![a.clone()], OpAttrs::default())?;
        let call = sh.down.call(a, so.clone());
        run.run(call.op, call.inputs, call.outputs, call.attrs)?;
        let call = sh.gate_logit.call(n.clone(), sl.clone());
        run.run(call.op, call.inputs, call.outputs, call.attrs)?;
        run.run(Op::MoeCombine, vec![routed, so, sl], vec![out.clone()], OpAttrs::default())
    }
}

fn expert_info(src: &dyn WeightSource, name: &str) -> Result<WeightInfo> {
    src.info(name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))
}

/// Stack `cfg.experts` per-expert NVFP4 tensors into the kernel layout (one upload per tensor, no host staging).
fn load_experts(backend: &dyn Backend, src: &dyn WeightSource, prefix: &str, cfg: &MoeConfig) -> Result<Experts> {
    let (e, h, i) = (cfg.experts, cfg.hidden, cfg.inter);
    if h % 16 != 0 || i % 16 != 0 {
        return Err(Error::Unsupported(format!("nvfp4 experts need H and I multiples of 16 (H={h}, I={i})")));
    }
    let gu_w = alloc(backend, DType::U8, &[e, 2 * i, h / 2])?;
    let gu_s = alloc(backend, DType::Fp8E4m3, &[e, 2 * i, h / 16])?;
    let dn_w = alloc(backend, DType::U8, &[e, h, i / 2])?;
    let dn_s = alloc(backend, DType::Fp8E4m3, &[e, h, i / 16])?;
    let mut gs = vec![0u8; e * 3 * 4];
    let put = |dst: &TensorRef, off: usize, name: &str, shape: &[usize], dt: DType| -> Result<()> {
        let info = expert_info(src, name)?;
        if info.dtype != dt || info.shape != shape {
            return Err(Error::model(format!("tensor '{name}' is {} {:?}, expected {dt} {shape:?}", info.dtype, info.shape)));
        }
        backend.upload(&dst.buf, dst.offset + off, src.bytes(name)?, 0)
    };
    for x in 0..e {
        let p = format!("{prefix}.experts.{x}");
        let gu_row = h / 2;
        let gu_srow = h / 16;
        for (j, proj) in ["gate_proj", "up_proj"].iter().enumerate() {
            put(&gu_w, (x * 2 * i + j * i) * gu_row, &format!("{p}.{proj}.weight"), &[i, h / 2], DType::U8)?;
            put(&gu_s, (x * 2 * i + j * i) * gu_srow, &format!("{p}.{proj}.weight_scale"), &[i, h / 16], DType::Fp8E4m3)?;
        }
        put(&dn_w, x * h * (i / 2), &format!("{p}.down_proj.weight"), &[h, i / 2], DType::U8)?;
        put(&dn_s, x * h * (i / 16), &format!("{p}.down_proj.weight_scale"), &[h, i / 16], DType::Fp8E4m3)?;
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
