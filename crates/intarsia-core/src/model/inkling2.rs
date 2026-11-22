//! Inkling-2-Mini (`inkling2_loop`) in the generic model: one [`InkLayer`] per physical layer (weights), a model-level
//! table of 48 visits (`(layer, pass)`, state slots per visit), and the per-pass workspace (block stack, partial, bank,
//! messages). Formulas: docs/plans/2026-11-21-inkling2-port.md section a; op contracts: `crate::inkling2::ops`.
//!
//! Per visit (launches): `InkInProj` -> `InkKvPrologue` -> `RelSinkAttn` -> `Matmul` (wo) -> `ConvMix` (attn_sconv, the MLP
//! input mix and, on a looped visit, the MLP blend) -> `Matmul` (router, f32) -> `MoeRoute` -> `MoeExperts` (situ, NVFP4) or
//! `MoeExpertsMxfp8` -> `ConvMix` (mlp_sconv, block residual, the NEXT visit's attention mix, blend and norms, or the
//! output mix and final norm after the last visit). A looped visit adds one `AttnResMix` for the next visit's messages.
//! The next visit's mix cannot be fused where its sources are not `(blocks, sum)`: at the loop restart `(core_start, p >= 1)`
//! (the bank is copied out, the prelude restored, and a standalone `AttnResMix` (+ `LoopBlend`) runs), and on visit 0 (a
//! plain `RmsNorm` of the normed embedding).
//!
//! Workspace layout: the block stack `S`, partial `P`, bank `B`, attention inputs `X`, messages `M` and blend destinations are
//! planes of `[T, D]` at stride `T * D` for the pass's `T` (they live within one pass). The partial lives in `S[nb]` when the
//! visit it enters is the first of an attn-res block (so the push is free and the MLP mix reads `S[0..nb+1]`), else in
//! `P[0]` / `P[1]`; [`plan_visits`] fixes every location at load.

use super::*;
use crate::inkling2::config::InkDims;
use crate::inkling2::ops::{nq as nq_width, InkAttrs};
use crate::state::RingKind;
use std::collections::BTreeMap;

/// Rows a ring keeps beyond the history a call reads (conv: `K - 1`, SWA KV: the window): how far a truncate or a speculative
/// rollback may cut (spec section f: at least `max verify rows - 1 + rollback depth` = 6).
pub const INK_RING_MARGIN: usize = 16;

/// Where a partial lives (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Loc {
    Stack(usize),
    P(usize),
}

#[derive(Clone, Debug)]
pub(super) struct InkVisit {
    pub layer: usize,
    pub pass: usize,
    /// Blocks on the stack when the visit starts, and where its input partial is.
    pub nb_in: usize,
    pub partial_in: Loc,
    /// Where its output partial goes (the sum of the MLP-side `ConvMix`), and the blocks after it (`nb_in + first`).
    pub dest: Loc,
    pub nb_out: usize,
    /// `(core_start, p >= 1)`: copy the bank out of `(S[bank_stack], bank_partial)` and restore the prelude first.
    pub restart: bool,
    pub bank_stack: usize,
    pub bank_partial: Loc,
    /// State slots: KV (`Ring` for SWA, `AttnKv` for full), K/V conv windows (`Ring`, 2 planes), attn / mlp conv windows (`Ring`, 2 planes).
    pub kv_slot: usize,
    pub kvconv_slot: usize,
    pub hconv_slot: usize,
}

/// Routed experts of one layer.
pub(super) enum InkExperts {
    /// `MoeExperts` inputs 3..8: gate rows then up rows (de-interleaved at load), `gscales [E,3]` = amax / denominator.
    Nvfp4 { gate_up_w: TensorRef, gate_up_s: TensorRef, down_w: TensorRef, down_s: TensorRef, gscales: TensorRef },
    /// `MoeExpertsMxfp8` inputs 3..7 (w13 de-interleaved at load).
    Mxfp8 { w13: TensorRef, s13: TensorRef, w2: TensorRef, s2: TensorRef },
}

/// The loop controllers of a core layer, streams `(q, k, v, mlp)` stacked: `down [4,G,2D]`, `gain` / `forget [4,D,G]`, and the
/// message queries `[4,D]` f32 (x D^-0.5).
pub(super) struct InkCtrl {
    pub down: TensorRef,
    pub gain: TensorRef,
    pub forget: TensorRef,
    pub msg_q: TensorRef,
}

pub(super) struct InkLayer {
    pub local: bool,
    /// `layer % n_layers_per_attn_res_block == 0`: pushes its input partial as a block.
    pub first: bool,
    /// `[q | r | gate | sink]`, `[k | k_taps]`, `[v | v_taps]` rows stacked, `[nq + 2 nk, D]`.
    pub in_w: TensorRef,
    pub nq: usize,
    pub nk: usize,
    /// Lag-major static K / V taps `[K, Hk*Dh]` (SWA: from `[C,1,K]`; full: `static_w`).
    pub ktaps: TensorRef,
    pub vtaps: TensorRef,
    pub qscale: TensorRef,
    pub rscale: TensorRef,
    pub relproj: TensorRef,
    pub wo: TensorRef,
    /// Attention-input norms `[3, D]` (SWA: `attn_norm` three times; full: q / k / v input norms).
    pub norm3: TensorRef,
    pub mlp_norm: TensorRef,
    pub attn_sconv: TensorRef,
    pub mlp_sconv: TensorRef,
    pub router: TensorRef,
    pub router_bias: TensorRef,
    pub experts: InkExperts,
    /// `attn_queries.{i-1}` `[1|3, D]` f32 x D^-0.5 (None for layer 0), `mlp_queries.{i}` `[1, D]`.
    pub attn_q: Option<TensorRef>,
    pub mlp_q: TensorRef,
    pub ctrl: Option<InkCtrl>,
}

pub(super) struct InkWs {
    pub max_tokens: usize,
    pub s: Buffer,
    pub p: Buffer,
    pub bank: Buffer,
    pub x: Buffer,
    pub msg: Buffer,
    pub dst: Buffer,
    pub emb: Buffer,
    pub pq: Buffer,
    pub pk: Buffer,
    pub pv: Buffer,
    pub q: Buffer,
    pub rel: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub o: Buffer,
    pub aproj: Buffer,
    pub aout: Buffer,
    pub mlpin: Buffer,
    pub moe: Buffer,
    pub rl: Buffer,
    pub rw: Buffer,
    pub ri: Buffer,
}

pub(super) struct InkModel {
    pub layers: Vec<InkLayer>,
    pub visits: Vec<InkVisit>,
    /// Planes of the block stack (`max nb_out + 1`).
    pub n_stack: usize,
    pub embed_norm: TensorRef,
    /// `output_query` `[1, D]` f32 x D^-0.5 and the final norm as `[1, D]`.
    pub out_q: TensorRef,
    pub final_norm: TensorRef,
    pub ws: InkWs,
    pub dims: InkDims,
}

/// Teacher-forced inputs / dumped tensors by the oracle's names (`vNN.partial_in`, `vNN.attn_out`, `final_hidden`, ...):
/// `(shape, f32 values)`.
pub type InkTensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// The visit table with every partial / block location (module doc). Errors when the layer pattern breaks the rules the
/// locations rely on (a first-of-block visit whose partial is not on the stack).
pub(super) fn plan_visits(ink: &InkDims) -> Result<(Vec<InkVisit>, usize)> {
    let vis = ink.visits();
    let first = |l: usize| l % ink.block == 0;
    let restart = |l: usize, p: usize| ink.looped() && l == ink.core_start && p >= 1;
    let (mut nb, mut partial) = (0usize, Loc::Stack(0));
    let mut prelude = (0usize, Loc::Stack(0));
    let mut out = Vec::new();
    let mut n_stack = 1;
    for (vi, &(l, p)) in vis.iter().enumerate() {
        let (mut bank_stack, mut bank_partial) = (0, Loc::P(0));
        if ink.looped() && l == ink.core_start {
            if p == 0 {
                prelude = (nb, partial);
            } else {
                bank_stack = prelude.0 + 1;
                bank_partial = partial;
                if bank_stack + 1 != nb {
                    return Err(Error::model(format!("inkling2: the bank at visit {vi} is not one block and the partial ({nb} blocks, prelude {})", prelude.0)));
                }
                (nb, partial) = prelude;
            }
        }
        if first(l) && partial != Loc::Stack(nb) {
            return Err(Error::model(format!("inkling2: visit {vi} (layer {l}) starts a block but its partial is not on the stack")));
        }
        if !first(l) && matches!(partial, Loc::Stack(_)) {
            return Err(Error::model(format!("inkling2: visit {vi} (layer {l}) has its partial on the stack")));
        }
        let nb_out = nb + usize::from(first(l));
        let next_on_stack = vis.get(vi + 1).is_some_and(|&(nl, np)| first(nl) && !restart(nl, np));
        let dest = if next_on_stack {
            Loc::Stack(nb_out)
        } else {
            match partial {
                Loc::P(0) => Loc::P(1),
                _ => Loc::P(0),
            }
        };
        if let Loc::Stack(i) = dest {
            n_stack = n_stack.max(i + 1);
        }
        n_stack = n_stack.max(nb_out.max(1));
        out.push(InkVisit {
            layer: l,
            pass: p,
            nb_in: nb,
            partial_in: partial,
            dest,
            nb_out,
            restart: restart(l, p),
            bank_stack,
            bank_partial,
            kv_slot: 0,
            kvconv_slot: 0,
            hconv_slot: 0,
        });
        nb = nb_out;
        partial = dest;
    }
    Ok((out, n_stack))
}

/// The state slots of every visit, appended to `specs` (their indices go into the visits). `act` is the slots' dtype (the
/// slot-dtype rule: K, V and conv inputs are stored as the call computes them).
pub(super) fn push_slots(visits: &mut [InkVisit], ink: &InkDims, dims: &ModelDims, act: DType, specs: &mut Vec<SlotSpec>) {
    let ckv = dims.n_kv_heads * dims.head_dim;
    let m = INK_RING_MARGIN;
    for v in visits.iter_mut() {
        v.kv_slot = specs.len();
        specs.push(if ink.local[v.layer] {
            SlotSpec::Ring { rows: ink.window + m, width: ckv, planes: 2, dtype: act, margin: m, kind: RingKind::InkKv }
        } else {
            SlotSpec::AttnKv { n_kv_heads: dims.n_kv_heads, head_dim: dims.head_dim, dtype: act }
        });
        v.kvconv_slot = specs.len();
        specs.push(SlotSpec::Ring { rows: ink.k_conv - 1 + m, width: ckv, planes: 2, dtype: act, margin: m, kind: RingKind::InkConv });
        v.hconv_slot = specs.len();
        specs.push(SlotSpec::Ring { rows: ink.k_conv - 1 + m, width: dims.hidden, planes: 2, dtype: act, margin: m, kind: RingKind::InkConv });
    }
}

/// Host values of a float tensor.
fn host(src: &dyn WeightSource, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let info = src.info(name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
    Ok((info.shape.clone(), crate::weights::to_f32_vec(info.dtype, src.bytes(name)?)?))
}

fn encode(dt: DType, v: &[f32]) -> Result<Vec<u8>> {
    super::gemma::encode_floats(dt, v)
}

/// Rows `2j` then `2j + 1` of every expert of an `[E, 2I, W]` tensor of `esz`-byte elements: gate rows first, then up.
fn deinterleave(bytes: &[u8], e: usize, two_i: usize, row_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes.len()];
    let half = two_i / 2;
    for ex in 0..e {
        let base = ex * two_i * row_bytes;
        for r in 0..two_i {
            let src = if r < half { 2 * r } else { 2 * (r - half) + 1 };
            out[base + r * row_bytes..base + (r + 1) * row_bytes].copy_from_slice(&bytes[base + src * row_bytes..base + (src + 1) * row_bytes]);
        }
    }
    out
}

/// What [`load`] needs from the recipe: the layer module template (`...layers.{i}`) and the root of the top-level tensors.
pub(super) struct InkNames {
    pub layer: String,
    pub root: String,
}

/// Load every layer, the visit table, slots and workspace. `bytes` counts the device bytes.
#[allow(clippy::too_many_arguments)]
pub(super) fn load(
    b: &dyn Backend,
    src: &dyn WeightSource,
    dims: &ModelDims,
    names: &InkNames,
    act: DType,
    mt: usize,
    slot_specs: &mut Vec<SlotSpec>,
    bytes: &mut u64,
) -> Result<InkModel> {
    let ink = dims.ink.clone().ok_or_else(|| Error::model("inkling2 recipe on a config that is not inkling2_loop"))?;
    let d = dims.hidden;
    let (h, hk, dh, dr) = (dims.n_heads, dims.n_kv_heads, dims.head_dim, ink.d_rel);
    let ckv = hk * dh;
    let kc = ink.k_conv;
    let s = (d as f32).powf(-0.5);
    let mut put = |data: &[u8], dt: DType, shape: &[usize]| -> Result<TensorRef> {
        let buf = crate::backend::upload_new(b, data)?;
        *bytes += data.len() as u64;
        Ok(TensorRef::new(buf, dt, shape))
    };
    // zero-copy where the checkpoint layout is already the op layout
    let direct = |name: &str, shape: Option<&[usize]>| -> Result<TensorRef> { load_tensor(b, src, name, shape) };
    let root = |n: &str| format!("{}.{n}", names.root);
    let wdt = src.info(&root("embed.weight")).map(|i| i.dtype).unwrap_or(DType::Bf16);
    let queries = |put: &mut dyn FnMut(&[u8], DType, &[usize]) -> Result<TensorRef>, v: Vec<f32>, q: usize| -> Result<TensorRef> {
        let v: Vec<f32> = v.iter().map(|x| x * s).collect();
        put(&encode(DType::F32, &v)?, DType::F32, &[q, d])
    };
    let n_core = ink.core_end - ink.core_start;
    let (mcq, mckv) = if ink.looped() {
        let (sa, a) = host(src, &root("middle_cycle_cross_pass_queries"))?;
        let (sb, bq) = host(src, &root("middle_cycle_cross_pass_kv_queries"))?;
        if sa != [2 * n_core, d] || sb != [2 * n_core, d] {
            return Err(Error::model(format!("inkling2: cross-pass queries {sa:?} / {sb:?}, expected [{}, {d}]", 2 * n_core)));
        }
        (a, bq)
    } else {
        (Vec::new(), Vec::new())
    };
    let mut layers = Vec::new();
    for li in 0..dims.n_layers {
        let p = names.layer.replace("{i}", &li.to_string());
        let t = |n: &str| format!("{p}.{n}");
        let local = ink.local[li];
        // the tensor pattern must agree with local_layer_ids (the oracle asserts the same)
        let has_static = src.info(&t("attn.k_sconv.weight")).is_some();
        let has_dyn = src.info(&t("attn.k_sconv.static_w")).is_some();
        if local != has_static || local == has_dyn {
            return Err(Error::model(format!("inkling2: layer {li} is {} in local_layer_ids but its tensors say otherwise", if local { "SWA" } else { "full" })));
        }
        // stacked in-projection: [q | r | gate | sink] from x_q, [k | k taps] from x_k, [v | v taps] from x_v
        let mut parts: Vec<(String, usize)> = vec![
            (t("attn.wq_du.weight"), h * dh),
            (t("attn.wr_du.weight"), h * dr),
            (t("attn.gate_attn.weight"), h * dh),
            (t("attn.attn_sink_proj.weight"), h),
            (t("attn.wk_dv.weight"), ckv),
        ];
        if !local {
            parts.push((t("attn.k_sconv.weight_proj"), hk * kc));
        }
        parts.push((t("attn.wv_dv.weight"), ckv));
        if !local {
            parts.push((t("attn.v_sconv.weight_proj"), hk * kc));
        }
        let mut stacked = Vec::new();
        for (name, rows) in &parts {
            let info = src.info(name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
            if info.shape != [*rows, d] || info.dtype != wdt {
                return Err(Error::model(format!("tensor '{name}' is {} {:?}, expected {wdt} [{rows}, {d}]", info.dtype, info.shape)));
            }
            stacked.extend_from_slice(src.bytes(name)?);
        }
        let nq = nq_width(h, dh, dr);
        let nk = ckv + if local { 0 } else { hk * kc };
        let in_w = put(&stacked, wdt, &[nq + 2 * nk, d])?;
        // lag-major taps [K, C]: `[C,1,K]` has tap K-1 on the current token; `static_w [K,C]` already has lag 0 first
        let lag_major = |put: &mut dyn FnMut(&[u8], DType, &[usize]) -> Result<TensorRef>, name: &str, c: usize| -> Result<TensorRef> {
            let (sh, v) = host(src, name)?;
            if sh != [c, 1, kc] {
                return Err(Error::model(format!("tensor '{name}' is {sh:?}, expected [{c}, 1, {kc}]")));
            }
            let mut o = vec![0f32; kc * c];
            for ch in 0..c {
                for l in 0..kc {
                    o[l * c + ch] = v[ch * kc + (kc - 1 - l)];
                }
            }
            put(&encode(wdt, &o)?, wdt, &[kc, c])
        };
        let (ktaps, vtaps) = if local {
            (lag_major(&mut put, &t("attn.k_sconv.weight"), ckv)?, lag_major(&mut put, &t("attn.v_sconv.weight"), ckv)?)
        } else {
            (direct(&t("attn.k_sconv.static_w"), Some(&[kc, ckv]))?, direct(&t("attn.v_sconv.static_w"), Some(&[kc, ckv]))?)
        };
        let relproj = direct(&t("attn.rel_logits_proj.proj"), None)?;
        let ext = if local { ink.window } else { ink.rel_extent };
        if relproj.shape != [dr, ext] {
            return Err(Error::model(format!("layer {li}: rel_logits_proj.proj is {:?}, expected [{dr}, {ext}]", relproj.shape)));
        }
        let norm3 = {
            let mut v = Vec::new();
            for n in if local { ["attn_norm"; 3] } else { ["q_input_norm", "k_input_norm", "v_input_norm"] } {
                let (sh, x) = host(src, &t(&format!("{n}.weight")))?;
                if sh != [d] {
                    return Err(Error::model(format!("layer {li}: {n} is {sh:?}")));
                }
                v.extend(x);
            }
            put(&encode(wdt, &v)?, wdt, &[3, d])?
        };
        let attn_q = if li == 0 {
            None
        } else {
            let (sh, v) = host(src, &root(&format!("attn_res.attn_queries.{}", li - 1)))?;
            let q = if local { 1 } else { 3 };
            if sh.iter().product::<usize>() != q * d {
                return Err(Error::model(format!("attn_queries.{} is {sh:?}, expected {q} x {d}", li - 1)));
            }
            Some(queries(&mut put, v, q)?)
        };
        let (_, mq) = host(src, &root(&format!("attn_res.mlp_queries.{li}")))?;
        let mlp_q = queries(&mut put, mq, 1)?;
        let ctrl = if ink.looped() && (ink.core_start..ink.core_end).contains(&li) {
            let g = ink.gate_rank;
            let (mut dn, mut gn, mut fg) = (Vec::new(), Vec::new(), Vec::new());
            for sname in ["attn", "k", "v", "mlp"] {
                for (vec, n, shape) in [(&mut dn, "down_weight", [g, 2 * d]), (&mut gn, "gain_weight", [d, g]), (&mut fg, "forget_weight", [d, g])] {
                    let name = t(&format!("controllers.{sname}.{n}"));
                    let info = src.info(&name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
                    if info.shape != shape || info.dtype != wdt {
                        return Err(Error::model(format!("tensor '{name}' is {:?}, expected {shape:?}", info.shape)));
                    }
                    vec.extend_from_slice(src.bytes(&name)?);
                }
            }
            let j = li - ink.core_start;
            // streams (q, k, v, mlp) = (queries[j,0], kv[j,0], kv[j,1], queries[j,1]) (oracle a12)
            let row = |v: &[f32], r: usize| v[r * d..(r + 1) * d].to_vec();
            let mut q4 = row(&mcq, 2 * j);
            q4.extend(row(&mckv, 2 * j));
            q4.extend(row(&mckv, 2 * j + 1));
            q4.extend(row(&mcq, 2 * j + 1));
            Some(InkCtrl {
                down: put(&dn, wdt, &[4, g, 2 * d])?,
                gain: put(&gn, wdt, &[4, d, g])?,
                forget: put(&fg, wdt, &[4, d, g])?,
                msg_q: queries(&mut put, q4, 4)?,
            })
        } else {
            None
        };
        // experts: NVFP4 (packed u8 codes) or MXFP8 (e4m3 values), by the stored dtype
        let (e, inter) = (ink.experts, ink.inter);
        let w13n = t("mlp.experts.w13_weight");
        let w13i = src.info(&w13n).ok_or_else(|| Error::model(format!("tensor '{w13n}' not found in checkpoint")))?;
        let experts = match w13i.dtype {
            DType::U8 => {
                let sn = t("mlp.experts.w13_weight_scale");
                let gu = deinterleave(src.bytes(&w13n)?, e, 2 * inter, d / 2);
                let si = src.info(&sn).ok_or_else(|| Error::model(format!("tensor '{sn}' not found")))?;
                if w13i.shape != [e, 2 * inter, d / 2] || si.shape != [e, 2 * inter, d / 16] || si.dtype != DType::Fp8E4m3 {
                    return Err(Error::model(format!("layer {li}: NVFP4 w13 {:?} / scales {} {:?}", w13i.shape, si.dtype, si.shape)));
                }
                let gs = deinterleave(src.bytes(&sn)?, e, 2 * inter, d / 16);
                let scale = |w: &str| -> Result<Vec<f32>> {
                    let (sa, amax) = host(src, &t(&format!("mlp.experts.{w}_weight_amax")))?;
                    let (_, den) = host(src, &t(&format!("mlp.experts.{w}_weight_outer_scale_denominator")))?;
                    if sa != [e] || den.len() != 1 || !(den[0] > 0.0) {
                        return Err(Error::model(format!("layer {li}: {w} amax {sa:?} / denominator {den:?}")));
                    }
                    Ok(amax.iter().map(|a| a / den[0]).collect())
                };
                let (a13, a2) = (scale("w13")?, scale("w2")?);
                let g3: Vec<f32> = (0..e).flat_map(|x| [a13[x], a13[x], a2[x]]).collect();
                InkExperts::Nvfp4 {
                    gate_up_w: put(&gu, DType::U8, &[e, 2 * inter, d / 2])?,
                    gate_up_s: put(&gs, DType::Fp8E4m3, &[e, 2 * inter, d / 16])?,
                    down_w: direct(&t("mlp.experts.w2_weight"), Some(&[e, d, inter / 2]))?,
                    down_s: direct(&t("mlp.experts.w2_weight_scale"), Some(&[e, d, inter / 16]))?,
                    gscales: put(&encode(DType::F32, &g3)?, DType::F32, &[e, 3])?,
                }
            }
            DType::Fp8E4m3 => {
                let sn = t("mlp.experts.w13_weight_scale");
                if w13i.shape != [e, 2 * inter, d] {
                    return Err(Error::model(format!("layer {li}: MXFP8 w13 {:?}", w13i.shape)));
                }
                let w13 = deinterleave(src.bytes(&w13n)?, e, 2 * inter, d);
                let s13 = deinterleave(src.bytes(&sn)?, e, 2 * inter, d / 32);
                InkExperts::Mxfp8 {
                    w13: put(&w13, DType::Fp8E4m3, &[e, 2 * inter, d])?,
                    s13: put(&s13, DType::U8, &[e, 2 * inter, d / 32])?,
                    w2: direct(&t("mlp.experts.w2_weight"), Some(&[e, d, inter]))?,
                    s2: direct(&t("mlp.experts.w2_weight_scale"), Some(&[e, d, inter / 32]))?,
                }
            }
            other => return Err(Error::model(format!("layer {li}: experts stored as {other} (NVFP4 u8 or MXFP8 e4m3)"))),
        };
        let sconv = |put: &mut dyn FnMut(&[u8], DType, &[usize]) -> Result<TensorRef>, n: &str| lag_major(put, &t(&format!("{n}.weight")), d);
        let attn_sconv = sconv(&mut put, "attn_sconv")?;
        let mlp_sconv = sconv(&mut put, "mlp_sconv")?;
        layers.push(InkLayer {
            local,
            first: li % ink.block == 0,
            in_w,
            nq,
            nk,
            ktaps,
            vtaps,
            qscale: direct(&t("attn.q_l2_clip.scale"), Some(&[h, dh]))?,
            rscale: direct(&t("attn.rel_l2_clip.scale"), Some(&[h, dr]))?,
            relproj,
            wo: direct(&t("attn.wo_ud.weight"), Some(&[d, h * dh]))?,
            norm3,
            mlp_norm: direct(&t("mlp_norm.weight"), Some(&[d]))?.reshaped(&[1, d]),
            attn_sconv,
            mlp_sconv,
            router: direct(&t("mlp.gate.weight"), Some(&[e, d]))?,
            router_bias: direct(&t("mlp.gate.bias"), Some(&[e]))?,
            experts,
            attn_q,
            mlp_q,
            ctrl,
        });
    }
    let embed_norm = direct(&root("embed_norm.weight"), Some(&[d]))?;
    let (_, oq) = host(src, &root("attn_res.output_query"))?;
    let out_q = queries(&mut put, oq, 1)?;
    let final_norm = direct(&root("norm.weight"), Some(&[d]))?.reshaped(&[1, d]);
    let (mut visits, n_stack) = plan_visits(&ink)?;
    push_slots(&mut visits, &ink, dims, act, slot_specs);
    let ws = InkWs::alloc(b, mt, act, dims, &ink, n_stack)?;
    Ok(InkModel { layers, visits, n_stack, embed_norm, out_q, final_norm, ws, dims: ink })
}

impl InkWs {
    fn alloc(b: &dyn Backend, mt: usize, act: DType, dims: &ModelDims, ink: &InkDims, n_stack: usize) -> Result<InkWs> {
        let (d, esz) = (dims.hidden, act.size_bytes());
        let (h, hk, dh) = (dims.n_heads, dims.n_kv_heads, dims.head_dim);
        let ckv = hk * dh;
        let e_max = ink.window.max(ink.rel_extent);
        let a = |n: usize| -> Result<Buffer> {
            let buf = b.alloc(n.max(16), MemKind::Device)?;
            b.zero(&buf, 0)?;
            Ok(buf)
        };
        Ok(InkWs {
            max_tokens: mt,
            s: a(n_stack * mt * d * esz)?,
            p: a(2 * mt * d * esz)?,
            bank: a(2 * mt * d * esz)?,
            x: a(3 * mt * d * esz)?,
            msg: a(4 * mt * d * 4)?,
            dst: a(3 * mt * d * 4)?,
            emb: a(mt * d * esz)?,
            pq: a(mt * nq_width(h, dh, ink.d_rel) * esz)?,
            pk: a(mt * (ckv + hk * ink.k_conv) * esz)?,
            pv: a(mt * (ckv + hk * ink.k_conv) * esz)?,
            q: a(mt * h * dh * esz)?,
            rel: a(mt * h * e_max * esz)?,
            k: a(mt * ckv * esz)?,
            v: a(mt * ckv * esz)?,
            o: a(mt * h * dh * esz)?,
            aproj: a(mt * d * esz)?,
            aout: a(mt * d * esz)?,
            mlpin: a(mt * d * esz)?,
            moe: a(mt * d * esz)?,
            rl: a(mt * ink.experts * 4)?,
            rw: a(mt * ink.top_k * 4)?,
            ri: a(mt * ink.top_k * 4)?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        [&self.s, &self.p, &self.bank, &self.x, &self.msg, &self.dst, &self.emb, &self.pq, &self.pk, &self.pv, &self.q, &self.rel, &self.k, &self.v, &self.o, &self.aproj, &self.aout, &self.mlpin, &self.moe, &self.rl, &self.rw, &self.ri]
            .iter()
            .map(|b| b.bytes())
            .sum()
    }
}

/// `n` planes `[t, w]` of `buf` from plane `start`, at stride `t * w` (`[n, t, w]`).
fn planes(buf: &Buffer, dt: DType, start: usize, n: usize, t: usize, w: usize) -> TensorRef {
    let mut x = TensorRef::new(buf.clone(), dt, &[n, t, w]);
    x.offset = start * t * w * dt.size_bytes();
    x
}

fn plane(buf: &Buffer, dt: DType, i: usize, t: usize, w: usize) -> TensorRef {
    planes(buf, dt, i, 1, t, w).reshaped(&[t, w])
}

impl Model {
    fn ink_loc(&self, l: Loc, t: usize) -> TensorRef {
        let ink = self.ink.as_ref().expect("inkling2 model");
        let d = self.dims.hidden;
        match l {
            Loc::Stack(i) => plane(&ink.ws.s, self.act, i, t, d),
            Loc::P(i) => plane(&ink.ws.p, self.act, i, t, d),
        }
    }

    /// A copy inside the device (skipped while a plan is being recorded: phantom buffers have no memory).
    fn ink_copy(&self, dst: &TensorRef, src: &TensorRef) -> Result<()> {
        if self.recorder.lock().unwrap().is_some() {
            return Ok(());
        }
        self.backend.copy(&dst.buf, dst.offset, &src.buf, src.offset, src.nbytes(), self.stream)
    }

    fn ink_upload(&self, dst: &TensorRef, v: &[f32]) -> Result<()> {
        if v.len() != dst.numel() {
            return Err(Error::invalid(format!("inkling2 teacher forcing: {} values for {:?}", v.len(), dst.shape)));
        }
        self.backend.upload(&dst.buf, dst.offset, &encode(dst.dtype, v)?, self.stream)
    }

    fn ink_download(&self, x: &TensorRef) -> Result<Vec<f32>> {
        self.backend.synchronize(self.stream)?;
        let mut b = vec![0u8; x.nbytes()];
        self.backend.download(&x.buf, x.offset, &mut b, self.stream)?;
        if x.dtype == DType::I32 {
            return Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect());
        }
        let mut o = vec![0f32; x.numel()];
        crate::dtype::decode_to_f32(x.dtype, &b, &mut o);
        Ok(o)
    }

    /// One Inkling-2 pass over `t` rows at positions `pos0..` of `st`, the final normed hidden state into `out` (`[t, D]`).
    /// `tf`: teacher forcing (every visit's blocks / partial / bank and the head's from these tensors, every attention mix
    /// standalone); `dump`: collect the oracle-named intermediates (tests).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ink_pass(
        &self,
        st: &SequenceState,
        ids: &TensorRef,
        t: usize,
        pos0: usize,
        out: &TensorRef,
        profile: bool,
        tf: Option<&InkTensors>,
        mut dump: Option<&mut InkTensors>,
    ) -> Result<()> {
        let im = self.ink.as_ref().ok_or_else(|| Error::invalid("inkling2 pass on another model"))?;
        let (ink, ws) = (&im.dims, &im.ws);
        if t > ws.max_tokens {
            return Err(Error::invalid(format!("inkling2 pass of {t} rows > workspace {}", ws.max_tokens)));
        }
        let d = self.dims.hidden;
        let (h, hk, dh, dr) = (self.dims.n_heads, self.dims.n_kv_heads, self.dims.head_dim, ink.d_rel);
        let act = self.act;
        let eps = self.dims.rms_eps;
        let ex = |op: Op, i: Vec<TensorRef>, o: Vec<TensorRef>, a: OpAttrs| self.exec(op, i, o, a, profile);
        let a0 = OpAttrs { eps, ..OpAttrs::default() };
        let none = |dt: DType, shape: &[usize]| TensorRef::new(ws.emb.clone(), dt, shape);
        let stack = |start: usize, n: usize| planes(&ws.s, act, start, n, t, d);
        let xs = |n: usize| planes(&ws.x, act, 0, n, t, d);
        let msgs = |start: usize, n: usize| planes(&ws.msg, DType::F32, start, n, t, d);
        let bank = planes(&ws.bank, act, 0, 2, t, d);
        let dumping = dump.is_some();
        // host-computed dump tensors (router_sel = sigmoid(logits) + bias), merged into the dump at the end
        let mut extra: Vec<(String, (Vec<usize>, Vec<f32>))> = Vec::new();
        let mut put = |k: String, x: &TensorRef| -> Result<()> {
            if let Some(dm) = dump.as_deref_mut() {
                let v = self.ink_download(x)?;
                if k.ends_with(".sink") {
                    // the sink logits are the last H columns of pq
                    let w = x.shape[1];
                    let sink: Vec<f32> = (0..t).flat_map(|r| v[r * w + w - h..(r + 1) * w].to_vec()).collect();
                    dm.insert(k, (vec![t, h], sink));
                } else {
                    dm.insert(k, (x.shape.clone(), v));
                }
            }
            Ok(())
        };
        let get = |k: &str| -> Result<&(Vec<usize>, Vec<f32>)> {
            tf.and_then(|m| m.get(k)).ok_or_else(|| Error::invalid(format!("inkling2 teacher forcing: '{k}' missing")))
        };

        // a1: x0 = rms(embed[id], embed_norm), the partial of visit 0 (on the stack: layer 0 starts a block)
        let emb = plane(&ws.emb, act, 0, t, d);
        ex(Op::Embedding, vec![ids.clone(), self.embed.clone()], vec![emb.clone()], OpAttrs::default())?;
        ex(Op::RmsNorm, vec![emb, im.embed_norm.clone()], vec![stack(0, 1).reshaped(&[t, d])], a0.clone())?;
        put("embed".into(), &stack(0, 1).reshaped(&[t, d]))?;

        // the attention inputs of the next visit are already in X (the previous MLP-side ConvMix computed them)
        let mut x_ready = false;
        let n_vis = im.visits.len();
        for (vi, v) in im.visits.iter().enumerate() {
            let l = &im.layers[v.layer];
            let tag = format!("v{vi:02}");
            let looped = v.pass > 0;
            let partial = self.ink_loc(v.partial_in, t);
            if let Some(_m) = tf {
                if v.nb_in > 0 {
                    self.ink_upload(&stack(0, v.nb_in), &get(&format!("{tag}.blocks"))?.1)?;
                }
                self.ink_upload(&partial, &get(&format!("{tag}.partial_in"))?.1)?;
                if looped {
                    self.ink_upload(&bank, &get(&format!("{tag}.bank"))?.1)?;
                }
                x_ready = false;
            } else if v.restart {
                // a12: bank = [the block after the prelude's, the partial], then the prelude is back (it never left the stack)
                self.ink_copy(&plane(&ws.bank, act, 0, t, d), &stack(v.bank_stack, 1).reshaped(&[t, d]))?;
                self.ink_copy(&plane(&ws.bank, act, 1, t, d), &self.ink_loc(v.bank_partial, t))?;
            }
            let ctrl = l.ctrl.as_ref();
            if looped && ctrl.is_none() {
                return Err(Error::invalid("inkling2: a looped visit of a layer without controllers"));
            }
            // a12 messages: attn_res(bank[:1], bank[1], MQ) -> M (4 f32 rows), unless the previous visit made them
            if looped && !x_ready {
                let c = ctrl.unwrap();
                ex(Op::AttnResMix, vec![planes(&ws.bank, act, 0, 1, t, d), plane(&ws.bank, act, 1, t, d), none(act, &[0]), c.msg_q.clone(), none(act, &[0])], vec![msgs(0, 4)], a0.clone())?;
            }
            let nx = if l.local && !looped { 1 } else { 3 };
            if !x_ready {
                if v.nb_in == 0 {
                    // layer 0: no mix, the norms of the embedding itself
                    for j in 0..nx {
                        ex(Op::RmsNorm, vec![partial.clone(), l.norm3.rows(j, 1).reshaped(&[d])], vec![planes(&ws.x, act, j, 1, t, d).reshaped(&[t, d])], a0.clone())?;
                    }
                } else {
                    let aq = l.attn_q.as_ref().ok_or_else(|| Error::invalid("inkling2: attention queries missing"))?;
                    let nq = aq.shape[0];
                    if looped {
                        let c = ctrl.unwrap();
                        let dst = planes(&ws.dst, DType::F32, 0, nq, t, d);
                        ex(Op::AttnResMix, vec![stack(0, v.nb_in), partial.clone(), none(act, &[0]), aq.clone(), none(act, &[0])], vec![dst.clone()], a0.clone())?;
                        ex(Op::LoopBlend, vec![dst, msgs(0, 3), c.down.rows(0, 3), c.gain.rows(0, 3), c.forget.rows(0, 3), l.norm3.clone()], vec![xs(3)], a0.clone())?;
                    } else {
                        ex(Op::AttnResMix, vec![stack(0, v.nb_in), partial.clone(), none(act, &[0]), aq.clone(), l.norm3.rows(0, nq)], vec![xs(nq)], a0.clone())?;
                    }
                }
            }
            for (j, n) in ["attn_in_q", "attn_in_k", "attn_in_v"].iter().enumerate() {
                put(format!("{tag}.{n}"), &planes(&ws.x, act, j.min(nx - 1), 1, t, d).reshaped(&[t, d]))?;
            }

            // ---- attention (a4-a11) ----
            let nqw = l.nq;
            let pq = TensorRef::new(ws.pq.clone(), act, &[t, nqw]);
            let pk = TensorRef::new(ws.pk.clone(), act, &[t, l.nk]);
            let pv = TensorRef::new(ws.pv.clone(), act, &[t, l.nk]);
            ex(Op::InkInProj, vec![xs(nx), l.in_w.clone()], vec![pq.clone(), pk.clone(), pv.clone()], OpAttrs::default())?;
            let kvc = match &st.slots[v.kvconv_slot] {
                StateSlot::Ring(r) => r,
                _ => return Err(Error::invalid("inkling2: K/V conv slot is not a ring")),
            };
            let e = l.relproj.shape[1];
            let q = TensorRef::new(ws.q.clone(), act, &[t, h, dh]);
            let rel = TensorRef::new(ws.rel.clone(), act, &[t, h, e]);
            let k = TensorRef::new(ws.k.clone(), act, &[t, hk, dh]);
            let vv = TensorRef::new(ws.v.clone(), act, &[t, hk, dh]);
            let mut pa = OpAttrs { pos0, n_heads: h, n_kv_heads: hk, head_dim: dh, ..a0.clone() };
            pa.ink = InkAttrs {
                qk_clip: ink.qk_clip,
                rel_clip: ink.rel_clip,
                tau_floor: if l.local { 0 } else { ink.n_floor },
                tau_alpha: ink.alpha,
                dyn_conv: !l.local,
                d_rel: dr,
                ..InkAttrs::default()
            };
            ex(
                Op::InkKvPrologue,
                vec![pq.clone(), pk, pv, l.ktaps.clone(), l.vtaps.clone(), l.qscale.clone(), l.rscale.clone(), l.relproj.clone()],
                vec![q.clone(), rel.clone(), k.clone(), vv.clone(), kvc.plane(0), kvc.plane(1)],
                pa,
            )?;
            put(format!("{tag}.q"), &q)?;
            put(format!("{tag}.k"), &k)?;
            put(format!("{tag}.v"), &vv)?;
            put(format!("{tag}.rel"), &rel)?;
            put(format!("{tag}.sink"), &pq)?;
            let (kc, vc) = match &st.slots[v.kv_slot] {
                StateSlot::Ring(r) => (r.plane(0).reshaped(&[hk, r.rows, dh]), r.plane(1).reshaped(&[hk, r.rows, dh])),
                StateSlot::AttnKv(s) => (self.view(&s.k, s.dtype, &[hk, s.cap, dh]), self.view(&s.v, s.dtype, &[hk, s.cap, dh])),
                _ => return Err(Error::invalid("inkling2: KV slot kind")),
            };
            let o = TensorRef::new(ws.o.clone(), act, &[t, h * dh]);
            let mut aa = OpAttrs {
                pos0,
                scale: (dh as f32).powf(-0.5),
                n_heads: h,
                n_kv_heads: hk,
                head_dim: dh,
                window: if l.local { ink.window } else { 0 },
                ..a0.clone()
            };
            aa.ink = InkAttrs { kv_ring: l.local, d_rel: dr, ..InkAttrs::default() };
            ex(Op::RelSinkAttn, vec![q, k, vv, kc.clone(), vc.clone(), rel, pq], vec![o.clone(), kc, vc], aa)?;
            let aproj = plane(&ws.aproj, act, 0, t, d);
            ex(Op::Matmul, vec![o, l.wo.clone()], vec![aproj.clone()], OpAttrs::default())?;
            put(format!("{tag}.attn_proj"), &aproj)?;

            // ---- attention-side ConvMix: attn_out, the MLP input (a15 / a12) ----
            let hc = match &st.slots[v.hconv_slot] {
                StateSlot::Ring(r) => r,
                _ => return Err(Error::invalid("inkling2: conv slot is not a ring")),
            };
            let aout = plane(&ws.aout, act, 0, t, d);
            let mlpin = planes(&ws.mlpin, act, 0, 1, t, d);
            let res = if l.first { none(act, &[0]) } else { partial.clone() };
            let src = stack(0, v.nb_in + usize::from(l.first));
            let pc = OpAttrs { pos0, ..a0.clone() };
            let mut ins = vec![aproj, l.attn_sconv.clone(), res, none(act, &[0]), src, l.mlp_q.clone()];
            if looped {
                let c = ctrl.unwrap();
                ins.extend([l.mlp_norm.clone(), msgs(3, 1), c.down.rows(3, 1), c.gain.rows(3, 1), c.forget.rows(3, 1)]);
            } else {
                ins.extend([l.mlp_norm.clone(), none(act, &[0]), none(act, &[0]), none(act, &[0]), none(act, &[0])]);
            }
            ex(Op::ConvMix, ins, vec![aout.clone(), hc.plane(0), none(act, &[0]), mlpin.clone()], pc.clone())?;
            put(format!("{tag}.attn_out"), &aout)?;
            let mlpin = mlpin.reshaped(&[t, d]);
            put(format!("{tag}.mlp_in"), &mlpin)?;

            // ---- router (a13) and experts (a14) ----
            let rl = TensorRef::new(ws.rl.clone(), DType::F32, &[t, ink.experts]);
            ex(Op::Matmul, vec![mlpin.clone(), l.router.clone()], vec![rl.clone()], OpAttrs::default())?;
            let rw = TensorRef::new(ws.rw.clone(), DType::F32, &[t, ink.top_k]);
            let ri = TensorRef::new(ws.ri.clone(), DType::I32, &[t, ink.top_k]);
            let mut ra = OpAttrs { top_k: ink.top_k, scale: ink.route_scale, norm_topk: true, temperature: 1.0, ..OpAttrs::default() };
            ra.dsv.score_func = crate::dsv41::ops::ScoreFunc::Sigmoid;
            ex(Op::MoeRoute, vec![rl.clone(), l.router_bias.clone()], vec![rw.clone(), ri.clone()], ra)?;
            put(format!("{tag}.router_logits"), &rl)?;
            if dumping {
                let (lg, bias) = (self.ink_download(&rl)?, self.ink_download(&l.router_bias)?);
                let e = ink.experts;
                let sel: Vec<f32> = lg.iter().enumerate().map(|(j, z)| 1.0 / (1.0 + (-z).exp()) + bias[j % e]).collect();
                extra.push((format!("{tag}.router_sel"), (vec![t, e], sel)));
            }
            put(format!("{tag}.expert_ids"), &ri)?;
            put(format!("{tag}.expert_w"), &rw)?;
            let moe = plane(&ws.moe, act, 0, t, d);
            let mut xa = OpAttrs { top_k: ink.top_k, ..OpAttrs::default() };
            xa.ink = InkAttrs { situ: true, gate_cap: ink.gate_cap, up_cap: ink.up_cap, ..InkAttrs::default() };
            match &l.experts {
                InkExperts::Nvfp4 { gate_up_w, gate_up_s, down_w, down_s, gscales } => ex(
                    Op::MoeExperts,
                    vec![mlpin, ri, rw, gate_up_w.clone(), gate_up_s.clone(), down_w.clone(), down_s.clone(), gscales.clone()],
                    vec![moe.clone()],
                    xa,
                )?,
                InkExperts::Mxfp8 { w13, s13, w2, s2 } => {
                    ex(Op::MoeExpertsMxfp8, vec![mlpin, ri, rw, w13.clone(), s13.clone(), w2.clone(), s2.clone()], vec![moe.clone()], xa)?
                }
            }
            put(format!("{tag}.moe_out"), &moe)?;

            // ---- MLP-side ConvMix: the new partial (a15), then the next visit's attention inputs or the output mix ----
            let next = im.visits.get(vi + 1);
            let last = vi + 1 == n_vis;
            let fuse = tf.is_none() && next.is_some_and(|n| !n.restart);
            let dest = self.ink_loc(v.dest, t);
            let (res, add) = if l.first { (aout.clone(), none(act, &[0])) } else { (partial.clone(), aout.clone()) };
            let mut ins = vec![moe, l.mlp_sconv.clone(), res, add, stack(0, v.nb_out)];
            let mix_out;
            if last && tf.is_none() {
                ins.extend([im.out_q.clone(), im.final_norm.clone(), none(act, &[0]), none(act, &[0]), none(act, &[0]), none(act, &[0])]);
                mix_out = out.reshaped(&[1, t, d]);
            } else if fuse {
                let n = next.unwrap();
                let nl = &im.layers[n.layer];
                let nlooped = n.pass > 0;
                let aq = nl.attn_q.clone().ok_or_else(|| Error::invalid("inkling2: attention queries missing"))?;
                let nq = aq.shape[0];
                if nlooped {
                    let c = nl.ctrl.as_ref().ok_or_else(|| Error::invalid("inkling2: a looped visit of a layer without controllers"))?;
                    // the next visit's messages (this visit's are no longer read)
                    ex(Op::AttnResMix, vec![planes(&ws.bank, act, 0, 1, t, d), plane(&ws.bank, act, 1, t, d), none(act, &[0]), c.msg_q.clone(), none(act, &[0])], vec![msgs(0, 4)], a0.clone())?;
                    ins.extend([aq, nl.norm3.clone(), msgs(0, 3), c.down.rows(0, 3), c.gain.rows(0, 3), c.forget.rows(0, 3)]);
                    mix_out = xs(3);
                } else {
                    ins.extend([aq, nl.norm3.rows(0, nq), none(act, &[0]), none(act, &[0]), none(act, &[0]), none(act, &[0])]);
                    mix_out = xs(nq);
                }
            } else {
                ins.extend([TensorRef::new(ws.emb.clone(), DType::F32, &[0, d]), none(act, &[0]), none(act, &[0]), none(act, &[0]), none(act, &[0]), none(act, &[0])]);
                mix_out = none(act, &[0]);
            }
            ex(Op::ConvMix, ins, vec![plane(&ws.aproj, act, 0, t, d), hc.plane(1), dest.clone(), mix_out], pc)?;
            put(format!("{tag}.partial_out"), &dest)?;
            x_ready = fuse;
        }
        if tf.is_some() {
            // a16 on the reference's final blocks and partial
            let fb = get("final.blocks")?;
            let nb = fb.0[0];
            self.ink_upload(&stack(0, nb), &fb.1)?;
            let p = self.ink_loc(Loc::P(0), t);
            self.ink_upload(&p, &get("final.partial")?.1)?;
            ex(Op::AttnResMix, vec![stack(0, nb), p, none(act, &[0]), im.out_q.clone(), im.final_norm.clone()], vec![out.reshaped(&[1, t, d])], a0)?;
        }
        put("final_hidden".into(), out)?;
        if let Some(dm) = dump {
            dm.extend(extra);
        }
        Ok(())
    }
}
