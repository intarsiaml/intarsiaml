//! The checkpoint's multi-token-prediction head as a speculative drafter (`--spec mtp`, recipe `[mtp]`), Qwen3.5 / 3.6
//! layout (vLLM `Qwen3NextMultiTokenPredictor`):
//!
//! ```text
//!   x   = fc([norm_e(embed(token_{p+1})) | norm_h(hidden_p)])      fc: [H, 2H]; hidden_p = the main model's final
//!                                                                  normed hidden state of position p
//!   r   = x ; n = input_layernorm(r)
//!   r  += attn(n)          gated full attention, rope position p, its own KV cache (`MtpState::kv`)
//!   n   = post_attention_layernorm(r) ; r += moe(n)
//!   g   = norm(r)          -> lm_head(g) is the draft for position p + 2; g is the hidden input of the next draft
//! ```
//!
//! `fc` runs as two `[H, H]` projections (embedding half, hidden half) and an add, so no concat buffer is needed. The
//! checkpoint stores the head's routed experts as fused bf16 tensors (`experts.gate_up_proj [E, 2I, H]`,
//! `experts.down_proj [E, H, I]`); they are quantised to NVFP4 at load (ModelOpt RTN, `quant::nvfp4::quantize_rows`) so
//! the head runs on the same MoE kernels as the main model. Drafts never change the output (every emitted token is a
//! verify row's argmax), so the head's numerics only move the acceptance rate.

use super::*;
use crate::quant::moe::{Experts, MoeConfig, SharedExpert};
use crate::quant::{nvfp4, BF16};
use crate::recipe::MtpSpec;
use crate::state::MtpState;

pub(super) struct MtpHead {
    /// `fc` columns `[0, H)` (normed embedding) and `[H, 2H)` (normed hidden state), each `[H, H]` bf16.
    fc_e: QTensor,
    fc_h: QTensor,
    norm_e: TensorRef,
    norm_h: TensorRef,
    /// The decoder layer: attention reads slot 0 of `MtpState::kv`; its `pre_norm` is `input_layernorm`.
    attn: AttnBlock,
    /// `post_attention_layernorm` and the MoE block.
    moe: MoeBlock,
    norm: TensorRef,
    /// `[1, H]`: the head's last normed output row, the hidden input of a chained draft.
    chain: Buffer,
    /// Rows of the pending buffer a sequence's [`MtpState`] gets.
    pending_rows: usize,
    /// Tests: replaces the head's drafts (after the head ran, so its state advances as usual), see [`Model::mtp_script`].
    script: Mutex<Option<DraftScript>>,
}

/// `(sequence tokens followed by the next token, k) -> drafts`.
pub type DraftScript = Box<dyn Fn(&[u32], usize) -> Vec<u32> + Send>;

/// bf16 bytes -> f32.
fn bf16_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

/// Weight bytes and device tensors of the head, for `Model::load_shard` (`opts.mtp`).
#[allow(clippy::too_many_arguments)]
pub(super) fn load_head(
    spec: &MtpSpec,
    b: &dyn Backend,
    src: &dyn WeightSource,
    dims: &ModelDims,
    recipe: &Recipe,
    main_attn: Option<&AttnBlock>,
    load: &dyn Fn(&str, Option<&[usize]>) -> Result<TensorRef>,
    load_q: &dyn Fn(&str, usize, usize) -> Result<QTensor>,
    add_bytes: &dyn Fn(u64),
    pending_rows: usize,
) -> Result<MtpHead> {
    if spec.kind != "qwen3_5_mtp" {
        return Err(Error::Unsupported(format!("[mtp] kind '{}' (this build has qwen3_5_mtp)", spec.kind)));
    }
    let a = recipe
        .layer
        .attn
        .as_ref()
        .ok_or_else(|| Error::model("[mtp] needs the recipe's [layer.attn] (the head is an attention layer)"))?;
    let main_attn = main_attn.ok_or_else(|| Error::model("[mtp] needs a model with full-attention layers"))?;
    let mc = dims.moe.as_ref().ok_or_else(|| Error::model("[mtp] needs the model's MoE dimensions"))?;
    let (h, p) = (dims.hidden, spec.prefix.as_str());
    let l = format!("{p}.layers.0");
    let qdim = dims.n_heads * dims.head_dim;
    let kvdim = dims.n_kv_heads * dims.head_dim;
    let gate = usize::from(a.gated);

    // fc [H, 2H] bf16, split by columns
    let fc_name = resolve(src, &format!("{p}.fc"))?;
    let info = src.info(&fc_name).ok_or_else(|| Error::model(format!("tensor '{fc_name}' missing")))?;
    if info.dtype != DType::Bf16 || info.shape != [h, 2 * h] {
        return Err(Error::model(format!(
            "'{fc_name}' is {} {:?}, expected bf16 [{h}, {}]",
            info.dtype,
            info.shape,
            2 * h
        )));
    }
    let raw = src.bytes(&fc_name)?;
    let (mut e, mut hh) = (Vec::with_capacity(h * h * 2), Vec::with_capacity(h * h * 2));
    for r in 0..h {
        let row = &raw[r * 4 * h..(r + 1) * 4 * h];
        e.extend_from_slice(&row[..2 * h]);
        hh.extend_from_slice(&row[2 * h..]);
    }
    let dense = |bytes: &[u8]| -> Result<QTensor> {
        let buf = crate::backend::upload_new(b, bytes)?;
        add_bytes(bytes.len() as u64);
        Ok(QTensor {
            format: BF16,
            n: h,
            k: h,
            parts: vec![TensorRef::new(buf, DType::Bf16, &[h, h])],
            act_quant: false,
        })
    };
    let (fc_e, fc_h) = (dense(&e)?, dense(&hh)?);
    drop((e, hh));

    let attn = AttnBlock {
        pre_norm: load(&format!("{l}.input_layernorm"), Some(&[h]))?,
        wq: load_q(&format!("{l}.self_attn.q_proj"), qdim * (1 + gate), h)?,
        wk: load_q(&format!("{l}.self_attn.k_proj"), kvdim, h)?,
        wv: load_q(&format!("{l}.self_attn.v_proj"), kvdim, h)?,
        wo: load_q(&format!("{l}.self_attn.o_proj"), h, qdim)?,
        q_norm: a
            .q_norm
            .as_ref()
            .map(|_| load(&format!("{l}.self_attn.q_norm"), Some(&[dims.head_dim])))
            .transpose()?,
        k_norm: a
            .k_norm
            .as_ref()
            .map(|_| load(&format!("{l}.self_attn.k_norm"), Some(&[dims.head_dim])))
            .transpose()?,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        slot: 0,
        gated: a.gated,
        rotary_dim: main_attn.rotary_dim,
    };
    let moe = MoeBlock {
        pre_norm: load(&format!("{l}.post_attention_layernorm"), Some(&[h]))?,
        w: load_moe(b, src, &format!("{l}.mlp"), mc, load_q, add_bytes)?,
    };
    let chain = b.alloc((h * 4).max(16), MemKind::Device)?;
    Ok(MtpHead {
        fc_e,
        fc_h,
        norm_e: load(&format!("{p}.pre_fc_norm_embedding"), Some(&[h]))?,
        norm_h: load(&format!("{p}.pre_fc_norm_hidden"), Some(&[h]))?,
        attn,
        moe,
        norm: load(&format!("{p}.norm"), Some(&[h]))?,
        chain,
        pending_rows,
        script: Mutex::new(None),
    })
}

/// The head's MoE block: router and shared expert as stored (bf16), routed experts from the fused bf16 tensors,
/// quantised to NVFP4 in the kernel layout of `quant::moe::Experts` (per expert: gate rows, then up rows).
fn load_moe(
    b: &dyn Backend,
    src: &dyn WeightSource,
    prefix: &str,
    cfg: &MoeConfig,
    load_q: &dyn Fn(&str, usize, usize) -> Result<QTensor>,
    add_bytes: &dyn Fn(u64),
) -> Result<MoeWeights> {
    let (e, h, i) = (cfg.experts, cfg.hidden, cfg.inter);
    let router = load_q(&format!("{prefix}.gate"), e, h)?;
    let shared = if cfg.shared_inter > 0 {
        let si = cfg.shared_inter;
        Some(SharedExpert {
            gate: load_q(&format!("{prefix}.shared_expert.gate_proj"), si, h)?,
            up: load_q(&format!("{prefix}.shared_expert.up_proj"), si, h)?,
            down: load_q(&format!("{prefix}.shared_expert.down_proj"), h, si)?,
            gate_logit: load_q(&format!("{prefix}.shared_expert_gate"), 1, h)?,
        })
    } else {
        None
    };
    let get = |name: &str, shape: &[usize]| -> Result<&[u8]> {
        let info = src.info(name).ok_or_else(|| Error::model(format!("tensor '{name}' missing")))?;
        if info.dtype != DType::Bf16 || info.shape != shape {
            return Err(Error::model(format!("'{name}' is {} {:?}, expected bf16 {shape:?}", info.dtype, info.shape)));
        }
        src.bytes(name)
    };
    let gu = get(&format!("{prefix}.experts.gate_up_proj"), &[e, 2 * i, h])?;
    let dn = get(&format!("{prefix}.experts.down_proj"), &[e, h, i])?;
    // per expert: (gate_up packed, gate_up scales, down packed, down scales, [gs gate, gs up, gs down])
    type Q = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, [f32; 3]);
    let quant_one = |x: usize| -> Q {
        let g = bf16_f32(&gu[x * 2 * i * h * 2..(x * 2 * i + i) * h * 2]);
        let u = bf16_f32(&gu[(x * 2 * i + i) * h * 2..(x + 1) * 2 * i * h * 2]);
        let d = bf16_f32(&dn[x * h * i * 2..(x + 1) * h * i * 2]);
        let (gw, gsc, ggs) = nvfp4::quantize_rows(&g, i, h);
        let (uw, usc, ugs) = nvfp4::quantize_rows(&u, i, h);
        let (dw, dsc, dgs) = nvfp4::quantize_rows(&d, h, i);
        ([gw, uw].concat(), [gsc, usc].concat(), dw, dsc, [ggs, ugs, dgs])
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let mut out: Vec<Option<Q>> = (0..e).map(|_| None).collect();
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(e.div_ceil(threads)).enumerate() {
            let base = ci * e.div_ceil(threads);
            let q = &quant_one;
            s.spawn(move || {
                for (j, slot) in chunk.iter_mut().enumerate() {
                    *slot = Some(q(base + j));
                }
            });
        }
    });
    let (mut guw, mut gus, mut dnw, mut dns, mut gs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for q in out.into_iter().map(|q| q.expect("every expert quantised")) {
        guw.extend_from_slice(&q.0);
        gus.extend_from_slice(&q.1);
        dnw.extend_from_slice(&q.2);
        dns.extend_from_slice(&q.3);
        for v in q.4 {
            gs.extend_from_slice(&v.to_le_bytes());
        }
    }
    let up = |bytes: &[u8], dt: DType, shape: &[usize]| -> Result<TensorRef> {
        add_bytes(bytes.len() as u64);
        Ok(TensorRef::new(crate::backend::upload_new(b, bytes)?, dt, shape))
    };
    let experts = Experts {
        gate_up_w: up(&guw, DType::U8, &[e, 2 * i, h / 2])?,
        gate_up_s: up(&gus, DType::Fp8E4m3, &[e, 2 * i, h / 16])?,
        down_w: up(&dnw, DType::U8, &[e, h, i / 2])?,
        down_s: up(&dns, DType::Fp8E4m3, &[e, h, i / 16])?,
        gscales: up(&gs, DType::F32, &[e, 3])?,
    };
    Ok(MoeWeights { cfg: cfg.clone(), router, experts, shared })
}

impl Model {
    /// Whether the MTP head is loaded (`LoadOptions::mtp`).
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// The drafter state a new sequence gets (`None` without the head).
    pub(super) fn new_mtp_state(&self, cap: usize) -> Result<Option<Box<MtpState>>> {
        let Some(head) = &self.mtp else { return Ok(None) };
        let spec = SlotSpec::AttnKv { n_kv_heads: self.dims.n_kv_heads, head_dim: self.dims.head_dim, dtype: self.kv };
        let row_bytes = self.dims.hidden * self.act.size_bytes();
        Ok(Some(Box::new(MtpState {
            kv: SequenceState::new(&*self.backend, &[spec], cap)?,
            pending: self.backend.alloc(head.pending_rows * row_bytes, MemKind::Device)?,
            pending_rows: 0,
            cap_rows: head.pending_rows,
            row_bytes,
            active: true,
            synced: true,
        })))
    }

    /// After a main forward pass over the `t` rows just committed to `st` (positions `st.len - t .. st.len`, normed hidden
    /// states in `hn`): queue their hidden rows for the drafter, and with `catch_up` run the head over every queued row
    /// whose next token is known (all but the newest), so the queue holds one row between steps. A verify pass queues
    /// without catching up: its rejected rows are dropped by the rollback, the accepted ones go through with the next draft.
    pub(super) fn mtp_after_forward(
        &self,
        st: &mut SequenceState,
        hn: &TensorRef,
        t: usize,
        catch_up: bool,
        profile: bool,
    ) -> Result<()> {
        let Some(m) = st.mtp.as_mut() else { return Ok(()) };
        if !m.active || !m.synced {
            // a request that never drafts does not pay for the head (its rows are not queued)
            m.synced = false;
            return Ok(());
        }
        let r = self.mtp_queue(st, hn, t, catch_up, profile);
        if r.is_err() {
            if let Some(m) = st.mtp.as_mut() {
                m.synced = false;
            }
        }
        r
    }

    fn mtp_queue(&self, st: &mut SequenceState, hn: &TensorRef, t: usize, catch_up: bool, profile: bool) -> Result<()> {
        let m = st.mtp.as_ref().unwrap();
        if m.pending_rows + t > m.cap_rows {
            // cannot happen with the sizes `Model::load` gives the queue (a prefill chunk + a verify pass); stay correct
            let n = m.pending_rows;
            self.mtp_run_pending(st, n, None, profile)?;
        }
        let m = st.mtp.as_mut().unwrap();
        if m.pending_rows + t > m.cap_rows {
            return Err(Error::invalid(format!("MTP queue of {} rows cannot take {t} more", m.cap_rows)));
        }
        if self.recorder.lock().unwrap().is_none() {
            self.backend.copy(&m.pending, m.pending_rows * m.row_bytes, &hn.buf, hn.offset, t * m.row_bytes, 0)?;
        }
        m.pending_rows += t;
        if catch_up && m.pending_rows > 1 {
            let n = m.pending_rows - 1;
            self.mtp_run_pending(st, n, None, profile)?;
        }
        Ok(())
    }

    /// Run the head over the first `n` queued rows (tokens `st.tokens[kv.len + 1 ..]`, and `next` as the token after the
    /// last committed one when the rows reach it), in pieces of at most a prefill chunk, then shift the rest of the queue
    /// to its front. Returns the argmax of the last row run (the first draft when `next` is given).
    fn mtp_run_pending(&self, st: &mut SequenceState, n: usize, next: Option<u32>, profile: bool) -> Result<u32> {
        let head = self.mtp.as_ref().ok_or_else(|| Error::invalid("no MTP head"))?;
        let mut m = st.mtp.take().ok_or_else(|| Error::invalid("sequence without MTP state"))?;
        let r = (|| -> Result<u32> {
            if n == 0 || n > m.pending_rows {
                return Err(Error::invalid(format!("MTP catch-up of {n} rows ({} queued)", m.pending_rows)));
            }
            // token of position p + 1 for queued row p
            let p0 = m.kv.len;
            let mut ids: Vec<u32> = st.tokens.get(p0 + 1..(p0 + n + 1).min(st.len)).unwrap_or(&[]).to_vec();
            if ids.len() < n {
                ids.extend(next);
            }
            if ids.len() != n {
                return Err(Error::invalid(format!("MTP catch-up of {n} rows has {} tokens", ids.len())));
            }
            let chunk = self.ws.max_tokens;
            let mut done = 0;
            let mut tok = 0;
            while done < n {
                let c = chunk.min(n - done);
                let hid = TensorRef::new(m.pending.clone(), self.act, &[m.cap_rows, self.dims.hidden]).rows(done, c);
                let out = self.mtp_layer(head, &m.kv, &ids[done..done + c], &hid, profile)?;
                m.kv.commit(&ids[done..done + c]);
                done += c;
                if done == n {
                    tok = if next.is_some() { self.head_argmax(&out, c - 1, profile)? } else { 0 };
                    if next.is_some() && self.recorder.lock().unwrap().is_none() {
                        self.backend.copy(
                            &head.chain,
                            0,
                            &out.buf,
                            out.offset + (c - 1) * m.row_bytes,
                            m.row_bytes,
                            0,
                        )?;
                    }
                }
            }
            let rest = m.pending_rows - n;
            if rest > 0 && self.recorder.lock().unwrap().is_none() {
                for i in 0..rest {
                    // row by row: source and destination never overlap within one copy
                    self.backend.copy(
                        &m.pending,
                        i * m.row_bytes,
                        &m.pending.clone(),
                        (n + i) * m.row_bytes,
                        m.row_bytes,
                        0,
                    )?;
                }
            }
            m.pending_rows = rest;
            Ok(tok)
        })();
        st.mtp = Some(m);
        r
    }

    /// Up to `k` greedy drafts after `next` (the token the main model emits at position `st.len`): the head catches up
    /// over the queued rows (the last of which pairs with `next`), its argmax is draft 1; drafts 2.. chain the head on its
    /// own output. Afterwards the head's cache holds the committed positions only (`kv.len == st.len`, empty queue).
    pub fn mtp_draft(&self, st: &mut SequenceState, next: u32, k: usize) -> Result<Vec<u32>> {
        let head =
            self.mtp.as_ref().ok_or_else(|| Error::invalid("--spec mtp without the MTP head (LoadOptions::mtp)"))?;
        let m = st.mtp.as_ref().ok_or_else(|| Error::invalid("sequence without MTP state"))?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let n = m.pending_rows;
        if !m.active || !m.synced || n == 0 || m.kv.len + n != st.len {
            return Err(Error::invalid(format!(
                "MTP state at {} + {n} queued rows (active {}, synced {}), sequence at {}",
                m.kv.len, m.active, m.synced, st.len
            )));
        }
        let profile = false;
        let first = self.mtp_run_pending(st, n, Some(next), profile);
        let first = match first {
            Ok(t) => t,
            Err(e) => {
                // the head cache and queue may be part-way through the catch-up
                if let Some(m) = st.mtp.as_mut() {
                    m.synced = false;
                }
                return Err(e);
            }
        };
        let mut drafts = vec![first];
        let mut m = st.mtp.take().unwrap();
        let base = m.kv.len;
        let r = (|| -> Result<()> {
            let k = k.min(m.kv.cap.saturating_sub(base) + 1);
            while drafts.len() < k {
                let hid = TensorRef::new(head.chain.clone(), self.act, &[1, self.dims.hidden]);
                let ids = [*drafts.last().unwrap()];
                let out = self.mtp_layer(head, &m.kv, &ids, &hid, profile)?;
                m.kv.commit(&ids);
                drafts.push(self.head_argmax(&out, 0, profile)?);
                if self.recorder.lock().unwrap().is_none() {
                    self.backend.copy(&head.chain, 0, &out.buf, out.offset, m.row_bytes, 0)?;
                }
            }
            Ok(())
        })();
        // the chained rows were speculative (and on an error, part-way): the head re-runs those positions from the
        // verified tokens, so its cache goes back to the committed length on both paths
        let cut = m.kv.truncate(base, &*self.backend);
        if cut.is_err() {
            m.synced = false;
        }
        st.mtp = Some(m);
        r?;
        cut?;
        if let Some(f) = head.script.lock().unwrap().as_ref() {
            let mut ctx = st.tokens.clone();
            ctx.push(next);
            let mut d = f(&ctx, drafts.len());
            d.truncate(drafts.len());
            if !d.is_empty() {
                drafts = d;
            }
        }
        Ok(drafts)
    }

    /// Plan dry-run (`Model::dry_run`, recorder installed): record the head's ops over `t` rows at `pos0`, so plan
    /// resolution and admission cover them (a server with `--require-admitted` would otherwise refuse the head's first
    /// pass at run time). Nothing without the head. The calls are marked `extra` (not part of the per-step op list).
    pub(super) fn mtp_dry_run(&self, t: usize, pos0: usize) -> Result<()> {
        let Some(head) = &self.mtp else { return Ok(()) };
        if self.recorder.lock().unwrap().is_none() {
            return Err(Error::invalid("mtp_dry_run outside a dry run"));
        }
        let spec = SlotSpec::AttnKv { n_kv_heads: self.dims.n_kv_heads, head_dim: self.dims.head_dim, dtype: self.kv };
        let kv = crate::plan::phantom_state(&[spec], self.plan.kv_cap_tokens, pos0);
        let ids = vec![0u32; t];
        let hid = self.view(&self.ws.blk, self.act, &[t, self.dims.hidden]);
        let extra = self.recorder.lock().unwrap().as_ref().is_some_and(|r| r.extra);
        if let Some(r) = self.recorder.lock().unwrap().as_mut() {
            r.extra = true;
        }
        // the head never runs inside a verify pass, so it selects ordinary (not row-exact) kernels
        let exact = self.exact_rows.swap(false, std::sync::atomic::Ordering::Relaxed);
        let out = self.mtp_layer(head, &kv, &ids, &hid, false);
        self.exact_rows.store(exact, std::sync::atomic::Ordering::Relaxed);
        if let Some(r) = self.recorder.lock().unwrap().as_mut() {
            r.extra = extra;
        }
        out.map(|_| ())
    }

    /// Diagnostic (`INTARSIA_MTP_VERIFY_CHECK`): run `ids` through k + 1 single-row decode steps and through one verify
    /// pass from the same state, and compare the rows. Leaves `st` as it was (a recurrent snapshot at the current length
    /// is taken for the restore). Returns per row: (max |logit diff|, single-step argmax, verify argmax, single-step top-2
    /// logit margin).
    pub fn verify_check(&self, st: &mut SequenceState, ids: &[u32]) -> Result<Vec<(f32, u32, u32, f32)>> {
        let len0 = st.len;
        let v = self.dims.vocab;
        let argmax = |r: &[f32]| {
            r.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
                as u32
        };
        st.snapshot(&*self.backend, crate::state::RECURRENT_SNAPSHOTS)?;
        let mut steps = Vec::with_capacity(ids.len());
        for &id in ids {
            match self.forward(st, &[id], HeadMode::LastLogits, None)? {
                StepOut::Logits(l) => steps.push(l),
                _ => return Err(Error::invalid("verify_check: no logits from a decode step")),
            }
        }
        st.truncate(len0, &*self.backend)?;
        let out = self.forward_verify(st, ids)?;
        let mut vl = vec![0u8; ids.len() * v * 4];
        self.backend.download(&self.ws.logits, 0, &mut vl, 0)?;
        st.truncate(len0, &*self.backend)?;
        let vl: Vec<f32> = vl.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(steps
            .iter()
            .enumerate()
            .map(|(r, s)| {
                let row = &vl[r * v..(r + 1) * v];
                let d = s.iter().zip(row).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                for &x in s.iter() {
                    if x > a {
                        b = a;
                        a = x;
                    } else if x > b {
                        b = x;
                    }
                }
                (d, argmax(s), out[r], a - b)
            })
            .collect())
    }

    /// Tests: draft with `f` instead of the head's argmax (the head still runs, so its cache and queue behave as in
    /// real use). Exercises every acceptance pattern of the verify and rollback path with known drafts.
    #[doc(hidden)]
    pub fn mtp_script(&self, f: Option<DraftScript>) -> Result<()> {
        let head = self.mtp.as_ref().ok_or_else(|| Error::invalid("no MTP head"))?;
        *head.script.lock().unwrap() = f;
        Ok(())
    }

    /// Tests: the head alone over `t = ids.len()` rows at positions `0..t` of an empty cache, with `hidden` (`[t, H]` f32,
    /// rounded to the activation dtype) as the main model's hidden rows. Returns the logits of every row.
    #[doc(hidden)]
    pub fn mtp_reference_pass(&self, ids: &[u32], hidden: &[f32]) -> Result<Vec<Vec<f32>>> {
        let head = self.mtp.as_ref().ok_or_else(|| Error::invalid("no MTP head"))?;
        let (t, h) = (ids.len(), self.dims.hidden);
        if hidden.len() != t * h || t > LOGIT_ROWS {
            return Err(Error::invalid("mtp_reference_pass: hidden must be [t, H], t <= 16"));
        }
        let host = crate::tensor::HostTensor::from_f32_as(self.act, &[t, h], hidden);
        let buf = crate::backend::upload_new(&*self.backend, &host.data)?;
        let st = self.new_mtp_state(t.max(1))?.ok_or_else(|| Error::invalid("no MTP state"))?;
        let out = self.mtp_layer(head, &st.kv, ids, &TensorRef::new(buf, self.act, &[t, h]), false)?;
        let flat = self.head_logits(&out, 0, t, false)?;
        Ok(flat.chunks(self.dims.vocab).map(|r| r.to_vec()).collect())
    }

    /// The head over `t` rows at positions `kv.len ..`: tokens `ids` (the token after each row's position) and the main
    /// model's hidden rows `hid` `[t, H]`. Returns its normed output `[t, H]` (in `ws.n`). Uses the main workspace.
    fn mtp_layer(
        &self,
        head: &MtpHead,
        kv: &SequenceState,
        ids: &[u32],
        hid: &TensorRef,
        profile: bool,
    ) -> Result<TensorRef> {
        let t = ids.len();
        let (d, act, ws) = (&self.dims, self.act, &self.ws);
        let h = d.hidden;
        if t == 0 || t > ws.max_tokens || kv.len + t > kv.cap {
            return Err(Error::invalid(format!(
                "MTP pass of {t} rows at {} (cap {}, chunk {})",
                kv.len, kv.cap, ws.max_tokens
            )));
        }
        if self.recorder.lock().unwrap().is_none() {
            for &i in ids {
                if i as usize >= d.vocab {
                    return Err(Error::BadRequest(format!("token id {i} >= vocab {}", d.vocab)));
                }
            }
            let ids_i32: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
            self.backend.upload(&ws.ids, 0, &ids_i32, 0)?;
        }
        let v = |b: &Buffer| self.view(b, act, &[t, h]);
        let (h0, h1, n, blk) = (v(&ws.h[0]), v(&ws.h[1]), v(&ws.n), v(&ws.blk));
        let eps = OpAttrs { eps: d.rms_eps, w_offset: self.recipe.model.norm_offset, ..Default::default() };
        self.exec(
            Op::Embedding,
            vec![self.view(&ws.ids, DType::I32, &[t]), self.embed.clone()],
            vec![h0.clone()],
            OpAttrs::default(),
            profile,
        )?;
        self.exec(Op::RmsNorm, vec![h0.clone(), head.norm_e.clone()], vec![n.clone()], eps.clone(), profile)?;
        self.exec(Op::RmsNorm, vec![hid.clone(), head.norm_h.clone()], vec![blk.clone()], eps.clone(), profile)?;
        self.proj(&n, &head.fc_e, &h1, profile)?;
        self.proj(&blk, &head.fc_h, &n, profile)?;
        self.exec(Op::Add, vec![h1.clone(), n.clone()], vec![h0.clone()], OpAttrs::default(), profile)?;
        // r = h0
        self.exec(Op::RmsNorm, vec![h0.clone(), head.attn.pre_norm.clone()], vec![n.clone()], eps.clone(), profile)?;
        self.attn_block(&head.attn, kv, &n, &blk, t, kv.len, None, false, profile)?;
        self.exec(
            Op::RmsNormAdd,
            vec![blk.clone(), h0.clone(), head.moe.pre_norm.clone()],
            vec![n.clone(), h1.clone()],
            eps.clone(),
            profile,
        )?;
        let mws = ws.moe.as_ref().ok_or_else(|| Error::invalid("MTP MoE block without the MoE workspace"))?;
        head.moe.w.forward(&ExecRunner { m: self, profile }, mws, &n, &blk)?;
        self.exec(Op::RmsNormAdd, vec![blk, h1, head.norm.clone()], vec![n.clone(), h0], eps, profile)?;
        Ok(n)
    }
}
