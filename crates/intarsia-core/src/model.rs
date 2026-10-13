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

use crate::backend::{Backend, Buffer, MemKind, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::memplan::MemoryPlan;
use crate::ops::{Op, OpAttrs, OpCall};
use crate::recipe::{ModelDims, Recipe};
use crate::registry::Registry;
use crate::state::{SequenceState, SlotSpec, StateSlot};
use crate::telemetry::Telemetry;
use crate::weights::{load_tensor, WeightSource};
use std::sync::Arc;
use std::time::Instant;

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
        }
    }
}

struct AttnBlock {
    pre_norm: TensorRef,
    wq: TensorRef,
    wk: TensorRef,
    wv: TensorRef,
    wo: TensorRef,
    q_norm: Option<TensorRef>,
    k_norm: Option<TensorRef>,
    slot: usize,
}

struct FfnBlock {
    pre_norm: TensorRef,
    gate: TensorRef,
    up: TensorRef,
    down: TensorRef,
}

enum Block {
    Attn(AttnBlock),
    Ffn(FfnBlock),
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

pub struct Model {
    pub backend: Arc<dyn Backend>,
    pub registry: Registry,
    pub dims: ModelDims,
    pub recipe: Recipe,
    pub act: DType,
    pub kv: DType,
    pub plan: MemoryPlan,
    pub weights_bytes: u64,
    embed: TensorRef,
    final_norm: TensorRef,
    lm_head: TensorRef,
    layers: Vec<Vec<Block>>,
    slot_specs: Vec<SlotSpec>,
    ws: Workspace,
    telemetry: Arc<Telemetry>,
    profile_every: u32,
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
    pub fn load(
        backend: Arc<dyn Backend>,
        recipe: &Recipe,
        dims: &ModelDims,
        src: &dyn WeightSource,
        opts: &LoadOptions,
        telemetry: Arc<Telemetry>,
    ) -> Result<Model> {
        recipe.validate()?;
        let b: &dyn Backend = &*backend;
        let mut weights_bytes = 0u64;
        let mut load = |tmpl: &str, layer: usize, shape: Option<&[usize]>| -> Result<TensorRef> {
            let name = resolve(src, &Recipe::expand(tmpl, layer))?;
            let t = load_tensor(b, src, &name, shape)?;
            weights_bytes += t.nbytes() as u64;
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
        let lm_head = {
            let lh = &recipe.top.lm_head;
            let own = resolve(src, &lh.tensor).ok();
            match (own, &lh.tied_to) {
                (Some(_), _) if !dims.tie_embeddings => load(&lh.tensor, 0, Some(&[dims.vocab, dims.hidden]))?,
                (_, Some(_)) => embed.clone(),
                (Some(_), None) => load(&lh.tensor, 0, Some(&[dims.vocab, dims.hidden]))?,
                (None, None) => {
                    return Err(Error::model(format!(
                        "lm_head tensor '{}' missing and the recipe has no tied_to",
                        lh.tensor
                    )))
                }
            }
        };

        let qdim = dims.n_heads * dims.head_dim;
        let kvdim = dims.n_kv_heads * dims.head_dim;
        let mut layers = Vec::new();
        let mut slot_specs = Vec::new();
        for li in 0..dims.n_layers {
            let mut blocks = Vec::new();
            for name in &recipe.layer.pattern {
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
                        let slot = slot_specs.len();
                        slot_specs.push(SlotSpec::AttnKv {
                            n_kv_heads: dims.n_kv_heads,
                            head_dim: dims.head_dim,
                            dtype: kv,
                        });
                        blocks.push(Block::Attn(AttnBlock {
                            pre_norm: load(pre, li, Some(&[dims.hidden]))?,
                            wq: load(&a.q, li, Some(&[qdim, dims.hidden]))?,
                            wk: load(&a.k, li, Some(&[kvdim, dims.hidden]))?,
                            wv: load(&a.v, li, Some(&[kvdim, dims.hidden]))?,
                            wo: load(&a.o, li, Some(&[dims.hidden, qdim]))?,
                            q_norm: a.q_norm.as_ref().map(|n| load(n, li, Some(&[dims.head_dim]))).transpose()?,
                            k_norm: a.k_norm.as_ref().map(|n| load(n, li, Some(&[dims.head_dim]))).transpose()?,
                            slot,
                        }));
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
                            gate: load(&f.gate, li, Some(&[dims.intermediate, dims.hidden]))?,
                            up: load(&f.up, li, Some(&[dims.intermediate, dims.hidden]))?,
                            down: load(&f.down, li, Some(&[dims.hidden, dims.intermediate]))?,
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
        let mt = opts.prefill_chunk.max(1);
        let esz = act.size_bytes();
        let mut ws_bytes = 0u64;
        let mut wb = |bytes: usize| -> Result<Buffer> {
            ws_bytes += bytes as u64;
            b.alloc(bytes.max(16), MemKind::Device)
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
        };

        let kv_per_tok: u64 = slot_specs
            .iter()
            .map(|s| match s {
                SlotSpec::AttnKv { n_kv_heads, head_dim, dtype } => {
                    (2 * n_kv_heads * head_dim * dtype.size_bytes()) as u64
                }
                _ => 0,
            })
            .sum();
        let plan = MemoryPlan::compute(
            &b.info(),
            weights_bytes,
            ws_bytes,
            kv_per_tok,
            opts.max_seq,
            dims.max_positions,
            opts.kv_fraction,
        )?;
        let registry = Registry::new(b.name(), &b.info(), b.kernels());

        Ok(Model {
            backend,
            registry,
            dims: dims.clone(),
            recipe: recipe.clone(),
            act,
            kv,
            plan,
            weights_bytes,
            embed,
            final_norm,
            lm_head,
            layers,
            slot_specs,
            ws,
            telemetry,
            profile_every: opts.profile_every,
        })
    }

    /// Allocate a fresh sequence state with the planned KV capacity.
    pub fn new_state(&self) -> Result<SequenceState> {
        SequenceState::new(&*self.backend, &self.slot_specs, self.plan.kv_cap_tokens)
    }

    /// Run `ids` through the model starting at `st.len`, appending to the state.
    ///
    /// `on_row` is required for [`HeadMode::AllLogits`]: it receives
    /// `(absolute position, logits row)` in order.
    pub fn forward(
        &self,
        st: &mut SequenceState,
        ids: &[u32],
        mode: HeadMode,
        mut on_row: Option<RowCallback<'_>>,
    ) -> Result<StepOut> {
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
        if mode == HeadMode::AllLogits && on_row.is_none() {
            return Err(Error::invalid("AllLogits needs a row callback"));
        }
        let nchunks = ids.len().div_ceil(self.ws.max_tokens);
        let mut out = StepOut::None;
        for (ci, chunk) in ids.chunks(self.ws.max_tokens).enumerate() {
            let last = ci + 1 == nchunks;
            let base = st.len;
            let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
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
        let call = OpCall::new(op, inputs, outputs, attrs);
        call.check()?;
        let kernel = self.registry.select(&call)?;
        let t0 = Instant::now();
        self.backend.launch(&kernel.name, &call, 0)?;
        if profile {
            self.backend.synchronize(0)?;
            self.telemetry.record_op(op, Some(t0.elapsed()));
        } else {
            self.telemetry.record_op(op, None);
        }
        Ok(())
    }

    /// Returns the normed final hidden states `[T, H]`.
    fn run_chunk(&self, st: &SequenceState, ids: &[u32], profile: bool) -> Result<TensorRef> {
        let t = ids.len();
        let d = &self.dims;
        let act = self.act;
        let h = d.hidden;
        let pos0 = st.len;
        let ws = &self.ws;

        let ids_i32: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
        self.backend.upload(&ws.ids, 0, &ids_i32, 0)?;
        for &i in ids {
            if i as usize >= d.vocab {
                return Err(Error::BadRequest(format!("token id {i} >= vocab {}", d.vocab)));
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

        let eps_attrs = OpAttrs { eps: d.rms_eps, ..Default::default() };
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
                        self.attn_block(a, st, &nv, &blkv, t, pos0, profile)?;
                    }
                    Block::Ffn(f) => {
                        norm_in(&f.pre_norm, pending, &mut cur)?;
                        self.ffn_block(f, &nv, &blkv, t, profile)?;
                    }
                }
                pending = true;
            }
        }
        // final: h = h + block_out ; n = norm(h)
        norm_in(&self.final_norm, pending, &mut cur)?;
        Ok(nv)
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
        profile: bool,
    ) -> Result<()> {
        let d = &self.dims;
        let act = self.act;
        let ws = &self.ws;
        let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
        let qshape = [t, nh * hd];
        let kshape = [t, nkv * hd];
        let q3 = [t, nh, hd];
        let k3 = [t, nkv, hd];
        let eps = OpAttrs { eps: d.rms_eps, ..Default::default() };
        let rope = OpAttrs { pos0, theta: d.rope_theta, head_dim: hd, ..Default::default() };

        // q: matmul -> [norm] -> rope ; result tracked by `cur`
        let mut qc = 0usize;
        self.exec(
            Op::Matmul,
            vec![n.clone(), a.wq.clone()],
            vec![self.view(&ws.q[qc], act, &qshape)],
            OpAttrs::default(),
            profile,
        )?;
        if let Some(w) = &a.q_norm {
            self.exec(
                Op::QkNorm,
                vec![self.view(&ws.q[qc], act, &q3), w.clone()],
                vec![self.view(&ws.q[1 - qc], act, &q3)],
                eps.clone(),
                profile,
            )?;
            qc = 1 - qc;
        }
        self.exec(
            Op::Rope,
            vec![self.view(&ws.q[qc], act, &q3)],
            vec![self.view(&ws.q[1 - qc], act, &q3)],
            rope.clone(),
            profile,
        )?;
        qc = 1 - qc;

        let mut kc = 0usize;
        self.exec(
            Op::Matmul,
            vec![n.clone(), a.wk.clone()],
            vec![self.view(&ws.k[kc], act, &kshape)],
            OpAttrs::default(),
            profile,
        )?;
        if let Some(w) = &a.k_norm {
            self.exec(
                Op::QkNorm,
                vec![self.view(&ws.k[kc], act, &k3), w.clone()],
                vec![self.view(&ws.k[1 - kc], act, &k3)],
                eps.clone(),
                profile,
            )?;
            kc = 1 - kc;
        }
        self.exec(
            Op::Rope,
            vec![self.view(&ws.k[kc], act, &k3)],
            vec![self.view(&ws.k[1 - kc], act, &k3)],
            rope.clone(),
            profile,
        )?;
        kc = 1 - kc;

        self.exec(
            Op::Matmul,
            vec![n.clone(), a.wv.clone()],
            vec![self.view(&ws.v, act, &kshape)],
            OpAttrs::default(),
            profile,
        )?;

        let slot = match &st.slots[a.slot] {
            StateSlot::AttnKv(s) => s,
            StateSlot::Recurrent(_) => return Err(Error::invalid("attention block bound to a recurrent state slot")),
        };
        let cache_shape = [nkv, slot.cap, hd];
        let kcache = self.view(&slot.k, slot.dtype, &cache_shape);
        let vcache = self.view(&slot.v, slot.dtype, &cache_shape);
        let append = OpAttrs { pos0, kv_cap: slot.cap, n_kv_heads: nkv, head_dim: hd, ..Default::default() };
        self.exec(Op::KvAppend, vec![self.view(&ws.k[kc], act, &k3)], vec![kcache.clone()], append.clone(), profile)?;
        self.exec(Op::KvAppend, vec![self.view(&ws.v, act, &k3)], vec![vcache.clone()], append, profile)?;

        let attn_out = self.view(&ws.q[1 - qc], act, &q3);
        let attrs = OpAttrs {
            pos0,
            scale: 1.0 / (hd as f32).sqrt(),
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            kv_cap: slot.cap,
            ..Default::default()
        };
        self.exec(
            if t == 1 { Op::AttnDecode } else { Op::AttnPrefill },
            vec![self.view(&ws.q[qc], act, &q3), kcache, vcache],
            vec![attn_out.clone()],
            attrs,
            profile,
        )?;
        self.exec(
            Op::Matmul,
            vec![attn_out.reshaped(&qshape), a.wo.clone()],
            vec![out.clone()],
            OpAttrs::default(),
            profile,
        )
    }

    fn ffn_block(&self, f: &FfnBlock, n: &TensorRef, out: &TensorRef, t: usize, profile: bool) -> Result<()> {
        let act = self.act;
        let ws = &self.ws;
        let i = self.dims.intermediate;
        let g = self.view(&ws.g, act, &[t, i]);
        let u = self.view(&ws.u, act, &[t, i]);
        let a = self.view(&ws.a, act, &[t, i]);
        self.exec(Op::Matmul, vec![n.clone(), f.gate.clone()], vec![g.clone()], OpAttrs::default(), profile)?;
        self.exec(Op::Matmul, vec![n.clone(), f.up.clone()], vec![u.clone()], OpAttrs::default(), profile)?;
        self.exec(Op::SiluGlu, vec![g, u], vec![a.clone()], OpAttrs::default(), profile)?;
        self.exec(Op::Matmul, vec![a, f.down.clone()], vec![out.clone()], OpAttrs::default(), profile)
    }

    /// Logits for rows `[r0, r0+c)` of the normed hidden states, row-major f32 `[c, vocab]`.
    fn head_logits(&self, hn: &TensorRef, r0: usize, c: usize, profile: bool) -> Result<Vec<f32>> {
        let v = self.dims.vocab;
        let x = hn.rows(r0, c);
        let out = self.view(&self.ws.logits, DType::F32, &[c, v]);
        self.exec(Op::Logits, vec![x, self.lm_head.clone()], vec![out.clone()], OpAttrs::default(), profile)?;
        let mut bytes = vec![0u8; c * v * 4];
        self.backend.download(&self.ws.logits, 0, &mut bytes, 0)?;
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    fn head_argmax(&self, hn: &TensorRef, row: usize, profile: bool) -> Result<u32> {
        let v = self.dims.vocab;
        let x = hn.rows(row, 1);
        let logits = self.view(&self.ws.logits, DType::F32, &[1, v]);
        self.exec(Op::Logits, vec![x, self.lm_head.clone()], vec![logits.clone()], OpAttrs::default(), profile)?;
        let tok = self.view(&self.ws.tok, DType::I32, &[1]);
        self.exec(Op::Argmax, vec![logits], vec![tok], OpAttrs::default(), profile)?;
        let mut b = [0u8; 4];
        self.backend.download(&self.ws.tok, 0, &mut b, 0)?;
        Ok(i32::from_le_bytes(b) as u32)
    }

    /// KV cache bytes one token occupies across all attention slots.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.plan.kv_bytes_per_token
    }
}
