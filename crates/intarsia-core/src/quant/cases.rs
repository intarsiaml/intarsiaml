//! Admission cases for the quantised-weight and MoE ops and the fused decode ops (`MatmulPair`, `MatmulFp8Pair`,
//! `MatmulNvfp4Glu`, `GatedRmsnorm`, `RmsNormAddMatmulPair`, `GatedRmsnormMatmulFp8`, `RmsNormAddMatmulFp8Pair`,
//! `SigmoidMulMatmulFp8`, `SharedDownCombine`, `MoeBlock`): the `intarsia kernels test` sweep and the exact-shape
//! builders used by plan-time admission (`admission::case_from_call`). Data is generated through the format
//! encoders so every case is valid (no e4m3 NaN encodings, finite scales, distinct experts per token).
//!
//! Non-quick sweeps include the real Qwen3.6-35B-A3B shapes: FP8 projections (8192x2048, 4096x2048, 2048x4096,
//! 512x2048), NVFP4 shared experts (512x2048, 2048x512), the full-vocab NVFP4 head (248320 rows, K=2048 and a
//! small-K case that catches grid limits on the N axis) and routed experts at H=2048, I=512, top-8.

use crate::admission::{Case, ShapeCaps, SweepConfig};
use crate::dtype::DType;
use crate::exec::{HostCall, OutSpec};
use crate::ops::{Op, OpAttrs, OpCall};
use crate::quant::{fp8, mxfp4, nvfp4, Mxfp4Act, FP8_E4M3, MXFP4, NVFP4};
use crate::rng::SplitMix64;
use crate::tensor::HostTensor;

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

pub fn matmul_fp8_case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, act_quant: bool, xdt: DType, odt: DType) -> HostCall {
    let q = fp8::random_rows(rng, n, k, act_quant);
    let x = rand_t(rng, xdt, &[m, k], 1.0);
    let mut parts = q.parts.into_iter();
    HostCall {
        op: Op::MatmulFp8,
        inputs: vec![x, parts.next().unwrap(), parts.next().unwrap()],
        outputs: vec![OutSpec::new(odt, &[m, n])],
        attrs: OpAttrs::default(),
    }
}

pub fn matmul_nvfp4_case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, xdt: DType, odt: DType) -> HostCall {
    let q = nvfp4::random_rows(rng, n, k);
    let x = rand_t(rng, xdt, &[m, k], 1.0);
    let mut parts = q.parts.into_iter();
    HostCall {
        op: Op::MatmulNvfp4,
        inputs: vec![x, parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap()],
        outputs: vec![OutSpec::new(odt, &[m, n])],
        attrs: OpAttrs::default(),
    }
}

/// `MatmulMxfp4`: random MXFP4 weights; `act` picks the activation mode (the op's `act` input).
pub fn matmul_mxfp4_case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, act: Mxfp4Act, xdt: DType, odt: DType) -> HostCall {
    let q = mxfp4::random_rows(rng, n, k, act);
    let x = rand_t(rng, xdt, &[m, k], 1.0);
    let mut parts = q.parts.into_iter();
    HostCall {
        op: Op::MatmulMxfp4,
        inputs: vec![x, parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap()],
        outputs: vec![OutSpec::new(odt, &[m, n])],
        attrs: OpAttrs::default(),
    }
}

/// Router logits for `t` tokens over `e` experts. `ties`: draw from a handful of values so exact ties (and the
/// lowest-index rule) are exercised; ties are also rounded through the logits dtype like a real router output.
pub fn topk_case(rng: &mut SplitMix64, t: usize, e: usize, k: usize, ties: bool, dt: DType, norm: bool) -> HostCall {
    let vals: Vec<f32> = if ties {
        (0..t * e).map(|_| (rng.next_u64() % 6) as f32 * 0.5 - 1.0).collect()
    } else {
        rng.normal_vec(t * e, 2.0)
    };
    HostCall {
        op: Op::MoeTopk,
        inputs: vec![HostTensor::from_f32_as(dt, &[t, e], &vals)],
        outputs: vec![OutSpec::new(DType::F32, &[t, k]), OutSpec::new(DType::I32, &[t, k])],
        attrs: OpAttrs { top_k: k, norm_topk: norm, ..OpAttrs::default() },
    }
}

/// Random stacked NVFP4 experts in the `MoeExperts` layout.
pub struct HostExperts {
    pub gate_up_w: HostTensor,
    pub gate_up_s: HostTensor,
    pub down_w: HostTensor,
    pub down_s: HostTensor,
    pub gscales: HostTensor,
}

pub fn random_experts(rng: &mut SplitMix64, e: usize, h: usize, i: usize) -> HostExperts {
    let (mut guw, mut gus, mut dw, mut ds, mut gs) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..e {
        for _ in 0..2 {
            let q = nvfp4::random_rows(rng, i, h);
            guw.extend_from_slice(&q.parts[0].data);
            gus.extend_from_slice(&q.parts[1].data);
            gs.push(q.parts[2].to_f32()[0]);
        }
        let q = nvfp4::random_rows(rng, h, i);
        dw.extend_from_slice(&q.parts[0].data);
        ds.extend_from_slice(&q.parts[1].data);
        gs.push(q.parts[2].to_f32()[0]);
        // the gate/up/down order of the loader: [gate, up, down]
    }
    HostExperts {
        gate_up_w: HostTensor { dtype: DType::U8, shape: vec![e, 2 * i, h / 2], data: guw },
        gate_up_s: HostTensor { dtype: DType::Fp8E4m3, shape: vec![e, 2 * i, h / 16], data: gus },
        down_w: HostTensor { dtype: DType::U8, shape: vec![e, h, i / 2], data: dw },
        down_s: HostTensor { dtype: DType::Fp8E4m3, shape: vec![e, h, i / 16], data: ds },
        gscales: HostTensor::from_f32(&[e, 3], &gs),
    }
}

/// `k` distinct expert ids per token and positive weights summing to one.
pub fn random_routing(rng: &mut SplitMix64, t: usize, e: usize, k: usize) -> (HostTensor, HostTensor) {
    let mut ids = Vec::with_capacity(t * k);
    let mut ws = Vec::with_capacity(t * k);
    for _ in 0..t {
        let mut pool: Vec<i32> = (0..e as i32).collect();
        for j in 0..k {
            let r = j + (rng.next_u64() % (e - j) as u64) as usize;
            pool.swap(j, r);
            ids.push(pool[j]);
        }
        let raw: Vec<f32> = (0..k).map(|_| 0.2 + rng.next_f32()).collect();
        let s: f32 = raw.iter().sum();
        ws.extend(raw.iter().map(|v| v / s));
    }
    (HostTensor::from_i32(&[t, k], &ids), HostTensor::from_f32(&[t, k], &ws))
}

pub fn experts_case(rng: &mut SplitMix64, t: usize, e: usize, k: usize, h: usize, i: usize, dt: DType) -> HostCall {
    let ex = random_experts(rng, e, h, i);
    let (ids, ws) = random_routing(rng, t, e, k);
    HostCall {
        op: Op::MoeExperts,
        inputs: vec![
            rand_t(rng, dt, &[t, h], 1.0),
            ids,
            ws,
            ex.gate_up_w,
            ex.gate_up_s,
            ex.down_w,
            ex.down_s,
            ex.gscales,
        ],
        outputs: vec![OutSpec::new(dt, &[t, h])],
        attrs: OpAttrs { top_k: k, ..OpAttrs::default() },
    }
}

pub fn combine_case(rng: &mut SplitMix64, t: usize, h: usize, dt: DType) -> HostCall {
    HostCall {
        op: Op::MoeCombine,
        inputs: vec![rand_t(rng, dt, &[t, h], 1.0), rand_t(rng, dt, &[t, h], 1.0), rand_t(rng, dt, &[t, 1], 2.0)],
        outputs: vec![OutSpec::new(dt, &[t, h])],
        attrs: OpAttrs::default(),
    }
}

/// `MatmulPair`: two dense `[n0,k]`, `[n1,k]` projections of one `x`.
pub fn matmul_pair_case(rng: &mut SplitMix64, m: usize, n0: usize, n1: usize, k: usize, dt: DType) -> HostCall {
    HostCall {
        op: Op::MatmulPair,
        inputs: vec![rand_t(rng, dt, &[m, k], 1.0), rand_t(rng, dt, &[n0, k], 0.5), rand_t(rng, dt, &[n1, k], 0.5)],
        outputs: vec![OutSpec::new(dt, &[m, n0]), OutSpec::new(dt, &[m, n1])],
        attrs: OpAttrs::default(),
    }
}

/// `RmsNormAddMatmulPair`: `x` and `res` summed and normed with a trained-like weight (`w_offset` 1.0 is the zero-centred
/// norm of Qwen3.5, whose stored weights sit around zero), then the two dense projections of the normed row.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_add_matmul_pair_case(
    rng: &mut SplitMix64,
    m: usize,
    n0: usize,
    n1: usize,
    k: usize,
    dt: DType,
    w_offset: f32,
) -> HostCall {
    let centre = 1.0 - w_offset;
    let nw: Vec<f32> = (0..k).map(|_| centre + 0.25 * rng.normal()).collect();
    HostCall {
        op: Op::RmsNormAddMatmulPair,
        inputs: vec![
            rand_t(rng, dt, &[m, k], 1.0),
            rand_t(rng, dt, &[m, k], 1.0),
            HostTensor::from_f32_as(dt, &[k], &nw),
            rand_t(rng, dt, &[n0, k], 0.5),
            rand_t(rng, dt, &[n1, k], 0.5),
        ],
        outputs: vec![OutSpec::new(dt, &[m, k]), OutSpec::new(dt, &[m, k]), OutSpec::new(dt, &[m, n0]), OutSpec::new(dt, &[m, n1])],
        attrs: OpAttrs { eps: 1e-6, w_offset, ..OpAttrs::default() },
    }
}

/// `MatmulFp8Pair`: two fp8 projections of one `x`; `aq0` / `aq1` select W8A8 (static input scale) per segment.
#[allow(clippy::too_many_arguments)]
pub fn matmul_fp8_pair_case(
    rng: &mut SplitMix64,
    m: usize,
    n0: usize,
    n1: usize,
    k: usize,
    aq0: bool,
    aq1: bool,
    xdt: DType,
    odt: DType,
) -> HostCall {
    let (a, b) = (fp8::random_rows(rng, n0, k, aq0), fp8::random_rows(rng, n1, k, aq1));
    let x = rand_t(rng, xdt, &[m, k], 1.0);
    let (mut pa, mut pb) = (a.parts.into_iter(), b.parts.into_iter());
    HostCall {
        op: Op::MatmulFp8Pair,
        inputs: vec![x, pa.next().unwrap(), pa.next().unwrap(), pb.next().unwrap(), pb.next().unwrap()],
        outputs: vec![OutSpec::new(odt, &[m, n0]), OutSpec::new(odt, &[m, n1])],
        attrs: OpAttrs::default(),
    }
}

/// `MatmulNvfp4Glu`: gate and up `[n,k]` NVFP4 projections of one `x`, combined as `silu(gate) * up`.
pub fn matmul_nvfp4_glu_case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, xdt: DType, odt: DType) -> HostCall {
    let (g, u) = (nvfp4::random_rows(rng, n, k), nvfp4::random_rows(rng, n, k));
    let x = rand_t(rng, xdt, &[m, k], 1.0);
    let (mut pg, mut pu) = (g.parts.into_iter(), u.parts.into_iter());
    HostCall {
        op: Op::MatmulNvfp4Glu,
        inputs: vec![
            x,
            pg.next().unwrap(),
            pg.next().unwrap(),
            pg.next().unwrap(),
            pu.next().unwrap(),
            pu.next().unwrap(),
            pu.next().unwrap(),
        ],
        outputs: vec![OutSpec::new(odt, &[m, n])],
        attrs: OpAttrs::default(),
    }
}

/// `GatedRmsnorm` over `r` rows of width `d` (weights around one, like a trained norm).
pub fn gated_rmsnorm_case(rng: &mut SplitMix64, r: usize, d: usize, dt: DType) -> HostCall {
    let w: Vec<f32> = (0..d).map(|_| 1.0 + 0.25 * rng.normal()).collect();
    HostCall {
        op: Op::GatedRmsnorm,
        inputs: vec![rand_t(rng, dt, &[r, d], 1.0), rand_t(rng, dt, &[r, d], 2.0), HostTensor::from_f32_as(dt, &[d], &w)],
        outputs: vec![OutSpec::new(dt, &[r, d])],
        attrs: OpAttrs { eps: 1e-6, ..OpAttrs::default() },
    }
}

/// `GatedRmsnormMatmulFp8`: the GDN output norm over `r` rows of 128 (core `x`, gate `z`, weights around one) and the fp8
/// projection `[n, r * 128]` of the result read as one row; `aq` selects W8A8 (static input scale).
pub fn gated_rmsnorm_matmul_fp8_case(rng: &mut SplitMix64, r: usize, n: usize, aq: bool, dt: DType) -> HostCall {
    let d = 128;
    let nw: Vec<f32> = (0..d).map(|_| 1.0 + 0.25 * rng.normal()).collect();
    let wq = fp8::random_rows(rng, n, r * d, aq);
    let mut parts = wq.parts.into_iter();
    HostCall {
        op: Op::GatedRmsnormMatmulFp8,
        inputs: vec![
            rand_t(rng, dt, &[r, d], 1.0),
            rand_t(rng, dt, &[r, d], 2.0),
            HostTensor::from_f32_as(dt, &[d], &nw),
            parts.next().unwrap(),
            parts.next().unwrap(),
        ],
        outputs: vec![OutSpec::new(dt, &[1, n])],
        attrs: OpAttrs { eps: 1e-6, ..OpAttrs::default() },
    }
}

/// `RmsNormAddMatmulFp8Pair`: `x` and `res` summed and normed (trained-like weight, `w_offset` as in
/// [`rmsnorm_add_matmul_pair_case`]), then the two fp8 projections of the normed row; `aq0` / `aq1` select W8A8 per segment.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_add_matmul_fp8_pair_case(
    rng: &mut SplitMix64,
    m: usize,
    n0: usize,
    n1: usize,
    k: usize,
    aq0: bool,
    aq1: bool,
    dt: DType,
    w_offset: f32,
) -> HostCall {
    let centre = 1.0 - w_offset;
    let nw: Vec<f32> = (0..k).map(|_| centre + 0.25 * rng.normal()).collect();
    let x = rand_t(rng, dt, &[m, k], 1.0);
    let res = rand_t(rng, dt, &[m, k], 1.0);
    let (a, b) = (fp8::random_rows(rng, n0, k, aq0), fp8::random_rows(rng, n1, k, aq1));
    let (mut pa, mut pb) = (a.parts.into_iter(), b.parts.into_iter());
    HostCall {
        op: Op::RmsNormAddMatmulFp8Pair,
        inputs: vec![
            x,
            res,
            HostTensor::from_f32_as(dt, &[k], &nw),
            pa.next().unwrap(),
            pa.next().unwrap(),
            pb.next().unwrap(),
            pb.next().unwrap(),
        ],
        outputs: vec![OutSpec::new(dt, &[m, k]), OutSpec::new(dt, &[m, k]), OutSpec::new(dt, &[m, n0]), OutSpec::new(dt, &[m, n1])],
        attrs: OpAttrs { eps: 1e-6, w_offset, ..OpAttrs::default() },
    }
}

/// `SigmoidMulMatmulFp8`: the attention output `a` times `sigmoid(gate)`, then the fp8 projection `[n, k]` of the product;
/// `aq` selects W8A8 (static input scale).
pub fn sigmoid_mul_matmul_fp8_case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, aq: bool, xdt: DType, odt: DType) -> HostCall {
    let a = rand_t(rng, xdt, &[m, k], 1.0);
    let gate = rand_t(rng, xdt, &[m, k], 2.0);
    let q = fp8::random_rows(rng, n, k, aq);
    let mut parts = q.parts.into_iter();
    HostCall {
        op: Op::SigmoidMulMatmulFp8,
        inputs: vec![a, gate, parts.next().unwrap(), parts.next().unwrap()],
        outputs: vec![OutSpec::new(odt, &[m, n])],
        attrs: OpAttrs::default(),
    }
}

/// `SharedDownCombine`: the NVFP4 shared-expert down projection `[n, k]` of `x`, combined with the routed output through the
/// shared-gate logit (`routed + sigmoid(logit) * down`).
pub fn shared_down_combine_case(rng: &mut SplitMix64, m: usize, n: usize, k: usize, xdt: DType, odt: DType) -> HostCall {
    let x = rand_t(rng, xdt, &[m, k], 1.0);
    let q = nvfp4::random_rows(rng, n, k);
    let mut parts = q.parts.into_iter();
    HostCall {
        op: Op::SharedDownCombine,
        inputs: vec![
            x,
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
            rand_t(rng, xdt, &[m, n], 1.0),
            rand_t(rng, xdt, &[m, 1], 2.0),
        ],
        outputs: vec![OutSpec::new(odt, &[m, n])],
        attrs: OpAttrs::default(),
    }
}

/// `MoeBlock` with `norm_topk` on: see [`moe_block_case_with`].
#[allow(clippy::too_many_arguments)]
pub fn moe_block_case(rng: &mut SplitMix64, t: usize, e: usize, k: usize, h: usize, i: usize, is: usize, dtype: DType) -> HostCall {
    moe_block_case_with(rng, t, e, k, h, i, is, dtype, true)
}

/// `MoeBlock`: the whole MoE block after the router logits. `n` `[t,h]`, router `logits` `[t,e]` (every row a shuffle of
/// distinct, dtype-exact values, so the top-`k` set and its order have no ties at any dtype), shared-gate logit `sl`
/// `[t,1]`, the five routed-expert tensors (`e` experts, intermediate `i`), the shared gate / up (`[is,h]`) and down
/// (`[h,is]`) NVFP4 projections, all in the dtype `dtype`; `out` `[t,h]`.
#[allow(clippy::too_many_arguments)]
pub fn moe_block_case_with(
    rng: &mut SplitMix64,
    t: usize,
    e: usize,
    k: usize,
    h: usize,
    i: usize,
    is: usize,
    dtype: DType,
    norm_topk: bool,
) -> HostCall {
    let n = rand_t(rng, dtype, &[t, h], 1.0);
    // distinct logits: multiples of 1/32 (exactly representable in bf16 / f16 for |v| < 8), a fresh shuffle per row
    let step = if e <= 256 { 1.0 / 32.0 } else { (e as f32 / 256.0) / 32.0 };
    let mut logits = Vec::with_capacity(t * e);
    for _ in 0..t {
        let mut pool: Vec<usize> = (0..e).collect();
        for j in 0..e {
            let r = j + (rng.next_u64() % (e - j) as u64) as usize;
            pool.swap(j, r);
        }
        logits.extend(pool.iter().map(|&p| (p as f32 - (e / 2) as f32) * step));
    }
    let logits = HostTensor::from_f32_as(dtype, &[t, e], &logits);
    let sl = rand_t(rng, dtype, &[t, 1], 2.0);
    let ex = random_experts(rng, e, h, i);
    let (g, u, d) = (nvfp4::random_rows(rng, is, h), nvfp4::random_rows(rng, is, h), nvfp4::random_rows(rng, h, is));
    let mut inputs = vec![n, logits, sl, ex.gate_up_w, ex.gate_up_s, ex.down_w, ex.down_s, ex.gscales];
    for q in [g, u, d] {
        inputs.extend(q.parts);
    }
    HostCall {
        op: Op::MoeBlock,
        inputs,
        outputs: vec![OutSpec::new(dtype, &[t, h])],
        attrs: OpAttrs { top_k: k, norm_topk, ..OpAttrs::default() },
    }
}

const ACT: [DType; 2] = [DType::Bf16, DType::F32];

/// The sweep for one of the new ops, or `None` for any other op.
pub fn sweep(op: Op, cfg: &SweepConfig) -> Option<Vec<Case>> {
    let mut rng = SplitMix64::new(cfg.seed ^ (op as u64).wrapping_mul(0x9E37) ^ 0x51ED);
    let quick = cfg.quick;
    let mut cases = Vec::new();
    let mut push = |label: String, call: HostCall| cases.push(Case { label, call });
    match op {
        Op::MatmulFp8 => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 8, 9, 40] };
            let mut nk = vec![(64usize, 64usize), (40, 96)];
            if !quick {
                // Qwen3.6-35B-A3B: qkv/q_proj, z, out_proj/o_proj, k/v_proj
                nk.extend([(8192, 2048), (4096, 2048), (2048, 4096), (512, 2048)]);
            }
            for &m in ms {
                for &(n, k) in &nk {
                    for (xdt, odt, aq) in [(DType::Bf16, DType::Bf16, true), (DType::Bf16, DType::Bf16, false), (DType::F32, DType::F32, true)] {
                        push(
                            format!("M={m} N={n} K={k} x={xdt} out={odt} act_quant={aq}"),
                            matmul_fp8_case(&mut rng, m, n, k, aq, xdt, odt),
                        );
                    }
                }
            }
            if !quick {
                // a prefill-shaped batch (M and N tails, 75 tiles of 128 x 128): the CUDA matmul_fp8_gemm_reg kernel
                // accepts M >= 256 only. Appended after the cases above, so their random data is unchanged.
                for (xdt, odt, aq) in [(DType::Bf16, DType::Bf16, true), (DType::Bf16, DType::Bf16, false), (DType::Bf16, DType::F32, true)] {
                    push(
                        format!("M=300 N=3080 K=256 x={xdt} out={odt} act_quant={aq}"),
                        matmul_fp8_case(&mut rng, 300, 3080, 256, aq, xdt, odt),
                    );
                }
            }
        }
        Op::MatmulNvfp4 => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 8, 9, 40] };
            let mut nk = vec![(64usize, 64usize), (40, 96)];
            if !quick {
                nk.extend([(512, 2048), (2048, 512), (8192, 2048)]);
            }
            for &m in ms {
                for &(n, k) in &nk {
                    for (xdt, odt) in [(DType::Bf16, DType::Bf16), (DType::F32, DType::F32)] {
                        push(format!("M={m} N={n} K={k} x={xdt} out={odt}"), matmul_nvfp4_case(&mut rng, m, n, k, xdt, odt));
                    }
                }
            }
            // lm_head: f32 logits; full-vocab N (grid limits on the N axis) with a small K, and the real K
            let vocab = 248_320usize;
            let head: Vec<(usize, usize)> = if quick { vec![(2048, 256)] } else { vec![(vocab, 128), (vocab, 2048), (8192, 2048)] };
            // m = 40: a prefill-sized head shape, graded by admission only. It selects `*_gemm` (M >= 33), but the
            // model's head never runs that path: `head_logits` computes at most LOGIT_ROWS (16) rows per call, so real
            // heads use `*_gemv` / `*_mma`. The case keeps the large-M f32-output path of `*_gemm` under test.
            for &m in if quick { &[1usize, 40][..] } else { &[1usize, 4, 40][..] } {
                for &(n, k) in &head {
                    push(format!("M={m} N={n} K={k} x=bf16 out=f32 (head)"), matmul_nvfp4_case(&mut rng, m, n, k, DType::Bf16, DType::F32));
                }
            }
        }
        Op::MatmulMxfp4 => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 8, 9, 40] };
            let mut nk = vec![(64usize, 64usize), (40, 96)];
            if !quick {
                // Qwen3.8-27B (hidden 5120): GDN in_proj_a/b, k/v_proj, o_proj / out_proj
                nk.extend([(48, 5120), (1024, 5120), (5120, 6144)]);
            }
            for &m in ms {
                for &(n, k) in &nk {
                    for (xdt, odt, act) in [
                        (DType::Bf16, DType::Bf16, Mxfp4Act::Dynamic),
                        (DType::Bf16, DType::Bf16, Mxfp4Act::None),
                        (DType::F32, DType::F32, Mxfp4Act::Dynamic),
                    ] {
                        push(
                            format!("M={m} N={n} K={k} x={xdt} out={odt} act={}", act.name()),
                            matmul_mxfp4_case(&mut rng, m, n, k, act, xdt, odt),
                        );
                    }
                }
            }
            if !quick {
                // the 27B gate/up projection, one row, both activation modes
                for act in [Mxfp4Act::Dynamic, Mxfp4Act::None] {
                    push(
                        format!("M=1 N=17408 K=5120 x=bf16 out=bf16 act={}", act.name()),
                        matmul_mxfp4_case(&mut rng, 1, 17408, 5120, act, DType::Bf16, DType::Bf16),
                    );
                }
                // a prefill-shaped batch with M and N tails
                push("M=300 N=1000 K=512 x=bf16 out=bf16 act=dynamic".into(), matmul_mxfp4_case(&mut rng, 300, 1000, 512, Mxfp4Act::Dynamic, DType::Bf16, DType::Bf16));
            }
        }
        Op::MoeTopk => {
            let ts: &[usize] = if quick { &[1, 5] } else { &[1, 5, 33, 256] };
            let ek: &[(usize, usize)] = if quick { &[(8, 2), (64, 8)] } else { &[(8, 2), (64, 8), (256, 8), (256, 1)] };
            for &t in ts {
                for &(e, k) in ek {
                    for dt in ACT {
                        for (ties, norm) in [(false, true), (true, true), (false, false)] {
                            push(
                                format!("T={t} E={e} K={k} {dt} ties={ties} norm={norm}"),
                                topk_case(&mut rng, t, e, k, ties, dt, norm),
                            );
                        }
                    }
                }
            }
        }
        Op::MoeExperts => {
            // (T, E, K, H, I)
            let mut shapes: Vec<(usize, usize, usize, usize, usize)> = vec![(1, 8, 2, 64, 32), (5, 16, 4, 128, 48), (36, 16, 4, 128, 128)];
            if !quick {
                shapes.extend([(1, 64, 8, 2048, 512), (4, 32, 8, 2048, 512), (40, 32, 8, 2048, 512), (130, 16, 8, 256, 64), (130, 16, 8, 256, 128)]);
                // grouped tensor-core kernel (T >= 32, H % 128, I % 128): short K loops, odd step counts, multi-tile experts
                shapes.extend([(60, 16, 8, 2048, 512), (45, 16, 4, 640, 384)]);
                // the real expert count (Qwen3.6-35B-A3B: 256 experts, top-8) at a prefill-sized batch
                shapes.push((256, 256, 8, 2048, 512));
                // a batch at the CUDA FP4 experts kernel's minimum (NVFP4_MIN_TOKENS = 1024), small H/I to keep the cpu reference cheap
                shapes.push((1024, 16, 8, 256, 128));
            }
            for (t, e, k, h, i) in shapes {
                for dt in ACT {
                    push(format!("T={t} E={e} K={k} H={h} I={i} {dt}"), experts_case(&mut rng, t, e, k, h, i, dt));
                }
            }
        }
        Op::MoeCombine => {
            for &t in if quick { &[1usize, 5][..] } else { &[1usize, 5, 33][..] } {
                for &h in if quick { &[64usize][..] } else { &[64usize, 2048][..] } {
                    for dt in ACT {
                        push(format!("T={t} H={h} {dt}"), combine_case(&mut rng, t, h, dt));
                    }
                }
            }
        }
        Op::MatmulPair => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 5, 40] };
            // (n0, n1, k): the two GDN gate projections and the MoE router + shared-expert gate of Qwen3.6-35B-A3B
            let mut nnk = vec![(64usize, 40usize, 64usize), (33, 1, 96), (9, 17, 40)];
            if !quick {
                nnk.extend([(32, 32, 2048), (256, 1, 2048), (300, 5, 520)]);
            }
            for &m in ms {
                for &(n0, n1, k) in &nnk {
                    for dt in ACT {
                        push(format!("M={m} N0={n0} N1={n1} K={k} {dt}"), matmul_pair_case(&mut rng, m, n0, n1, k, dt));
                    }
                }
            }
        }
        Op::RmsNormAddMatmulPair => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 5, 40] };
            // (n0, n1, k): the GDN gate projections and the MoE router + shared-expert gate of Qwen3.6-35B-A3B at its
            // hidden size, plus K narrower than the 256-thread block (K = 64, 96) and past it (4104: 513 vectors per row)
            let mut nnk = vec![(64usize, 40usize, 64usize), (33, 1, 96), (9, 17, 40)];
            if !quick {
                nnk.extend([(32, 32, 2048), (256, 1, 2048), (300, 5, 520), (40, 24, 4104)]);
            }
            for &m in ms {
                for &(n0, n1, k) in &nnk {
                    for dt in ACT {
                        for w_offset in [0.0f32, 1.0] {
                            push(
                                format!("M={m} N0={n0} N1={n1} K={k} {dt} w_offset={w_offset}"),
                                rmsnorm_add_matmul_pair_case(&mut rng, m, n0, n1, k, dt, w_offset),
                            );
                        }
                    }
                }
            }
        }
        Op::GatedRmsnormMatmulFp8 => {
            // (rows, n): the GDN out_proj of Qwen3.6-35B-A3B (32 value heads, N = hidden), and small shapes
            let mut rn = vec![(4usize, 40usize), (1, 64), (7, 33)];
            if !quick {
                rn.extend([(32, 2048), (32, 1), (16, 520)]);
            }
            for (r, n) in rn {
                for aq in [true, false] {
                    for dt in ACT {
                        push(format!("R={r} N={n} {dt} act_quant={aq}"), gated_rmsnorm_matmul_fp8_case(&mut rng, r, n, aq, dt));
                    }
                }
            }
        }
        Op::MatmulFp8Pair => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 5, 40] };
            let mut nnk = vec![(64usize, 48usize, 64usize), (40, 96, 96), (17, 33, 48)];
            if !quick {
                // Qwen3.6-35B-A3B: GDN qkv + z, attention k + v
                nnk.extend([(8192, 4096, 2048), (512, 512, 2048)]);
            }
            for &m in ms {
                for &(n0, n1, k) in &nnk {
                    for (xdt, odt, aq0, aq1) in [
                        (DType::Bf16, DType::Bf16, true, true),
                        (DType::Bf16, DType::Bf16, false, false),
                        (DType::Bf16, DType::Bf16, true, false),
                        (DType::F32, DType::F32, true, true),
                    ] {
                        push(
                            format!("M={m} N0={n0} N1={n1} K={k} x={xdt} out={odt} act_quant={aq0}/{aq1}"),
                            matmul_fp8_pair_case(&mut rng, m, n0, n1, k, aq0, aq1, xdt, odt),
                        );
                    }
                }
            }
        }
        Op::MatmulNvfp4Glu => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 5, 40] };
            let mut nk = vec![(64usize, 64usize), (40, 96), (9, 16)];
            if !quick {
                nk.extend([(512, 2048), (2048, 512), (300, 4112)]);
            }
            for &m in ms {
                for &(n, k) in &nk {
                    for (xdt, odt) in [(DType::Bf16, DType::Bf16), (DType::F32, DType::F32)] {
                        push(format!("M={m} N={n} K={k} x={xdt} out={odt}"), matmul_nvfp4_glu_case(&mut rng, m, n, k, xdt, odt));
                    }
                }
            }
        }
        Op::GatedRmsnorm => {
            let rs: &[usize] = if quick { &[1, 5] } else { &[1, 5, 33, 960] };
            let ds: &[usize] = if quick { &[128] } else { &[64, 128, 130, 256] };
            for &r in rs {
                for &d in ds {
                    for dt in ACT {
                        push(format!("R={r} D={d} {dt}"), gated_rmsnorm_case(&mut rng, r, d, dt));
                    }
                }
            }
        }
        Op::RmsNormAddMatmulFp8Pair => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 5, 40] };
            // (n0, n1, k): small and odd shapes, K narrower than the 256-thread block; Qwen3.6-35B-A3B's attention k + v at
            // its hidden size and a K past one chunk per thread (4112: 514 chunks)
            let mut nnk = vec![(64usize, 48usize, 64usize), (40, 96, 96), (17, 33, 48)];
            if !quick {
                nnk.extend([(512, 512, 2048), (24, 40, 4112)]);
            }
            for &m in ms {
                for &(n0, n1, k) in &nnk {
                    for (aq0, aq1, w_offset) in [(true, true, 1.0f32), (false, false, 0.0), (true, false, 1.0)] {
                        for dt in ACT {
                            push(
                                format!("M={m} N0={n0} N1={n1} K={k} {dt} w_offset={w_offset} act_quant={aq0}/{aq1}"),
                                rmsnorm_add_matmul_fp8_pair_case(&mut rng, m, n0, n1, k, aq0, aq1, dt, w_offset),
                            );
                        }
                    }
                }
            }
        }
        Op::SigmoidMulMatmulFp8 => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 5, 40] };
            // (n, k): small and odd shapes, the 35B's o_proj (2048 x 4096) and a tail case
            let mut nk = vec![(64usize, 64usize), (40, 96), (17, 48)];
            if !quick {
                nk.extend([(2048, 4096), (520, 2064)]);
            }
            for &m in ms {
                for &(n, k) in &nk {
                    for (xdt, odt, aq) in [(DType::Bf16, DType::Bf16, true), (DType::Bf16, DType::Bf16, false), (DType::F32, DType::F32, true)] {
                        push(
                            format!("M={m} N={n} K={k} x={xdt} out={odt} act_quant={aq}"),
                            sigmoid_mul_matmul_fp8_case(&mut rng, m, n, k, aq, xdt, odt),
                        );
                    }
                }
            }
        }
        Op::SharedDownCombine => {
            let ms: &[usize] = if quick { &[1, 5] } else { &[1, 3, 5, 8, 9, 40] };
            // (n, k): small and odd shapes, the 35B's shared-expert down projection (2048 x 512) and a long-K tail case
            let mut nk = vec![(64usize, 64usize), (40, 96), (9, 16)];
            if !quick {
                nk.extend([(2048, 512), (300, 4112)]);
            }
            for &m in ms {
                for &(n, k) in &nk {
                    for (xdt, odt) in [(DType::Bf16, DType::Bf16), (DType::F32, DType::F32)] {
                        push(format!("M={m} N={n} K={k} x={xdt} out={odt}"), shared_down_combine_case(&mut rng, m, n, k, xdt, odt));
                    }
                }
            }
        }
        Op::MoeBlock => {
            // (T, E, K, H, I, IS): small and odd shapes (H = 208 is a multiple of 16 but not of 32, IS < I), and the 35B's
            // decode shape (256 experts, top-8, H = 2048, I = IS = 512) at one and two tokens
            let mut shapes: Vec<(usize, usize, usize, usize, usize, usize)> =
                vec![(1, 16, 2, 256, 16, 32), (2, 32, 4, 256, 48, 64), (1, 16, 8, 208, 64, 16)];
            if !quick {
                shapes.extend([(2, 16, 2, 256, 16, 32), (5, 32, 4, 256, 48, 64), (3, 16, 8, 208, 64, 16)]);
            }
            for (t, e, k, h, i, is) in shapes {
                for dt in ACT {
                    for norm in [true, false] {
                        push(
                            format!("T={t} E={e} K={k} H={h} I={i} IS={is} {dt} norm={norm}"),
                            moe_block_case_with(&mut rng, t, e, k, h, i, is, dt, norm),
                        );
                    }
                }
            }
            if !quick {
                for (t, norm) in [(1, true), (1, false), (2, true)] {
                    push(
                        format!("T={t} E=256 K=8 H=2048 I=512 IS=512 {} norm={norm}", DType::Bf16),
                        moe_block_case_with(&mut rng, t, 256, 8, 2048, 512, 512, DType::Bf16, norm),
                    );
                }
            }
        }
        _ => return None,
    }
    Some(cases)
}

/// Clamped-case label of a planned call (also the admission cache key part), or `None` for other ops.
/// Rows and `N` are clamped per `caps`, experts to 32; `K`, `H`, `I`, top-k and dtypes stay exact.
pub fn label(call: &OpCall, caps: &ShapeCaps) -> Option<String> {
    let (i, o) = (&call.inputs, &call.outputs);
    let s = match call.op {
        Op::MatmulFp8 => format!(
            "M={} N={} K={} x={} out={} act_quant={}",
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[1].shape[1],
            i[0].dtype,
            o[0].dtype,
            // scale values are not visible at plan time; the case grades the W8A8 mode (the sweep covers both)
            "static"
        ),
        Op::MatmulNvfp4 => format!(
            "M={} N={} K={} x={} out={}",
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            o[0].dtype
        ),
        Op::MatmulMxfp4 => format!(
            "M={} N={} K={} x={} out={} act={}",
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            o[0].dtype,
            // the act flag is a value, not visible at plan time; the case grades the declared W4A4 mode
            "dynamic"
        ),
        Op::MoeTopk => format!(
            "T={} E={} K={} {} norm={}",
            i[0].shape[0].min(caps.rows),
            i[0].shape[1],
            call.attrs.top_k,
            i[0].dtype,
            call.attrs.norm_topk
        ),
        Op::MoeExperts => format!(
            "T={} E={} K={} H={} I={} {}",
            i[0].shape[0].min(caps.rows),
            i[3].shape[0].min(32).max(i[1].shape[1]),
            i[1].shape[1],
            i[0].shape[1],
            i[3].shape[1] / 2,
            i[0].dtype
        ),
        Op::MoeCombine => format!("T={} H={} {}", i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].dtype),
        Op::MatmulPair => format!(
            "M={} N0={} N1={} K={} {}",
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[2].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype
        ),
        Op::RmsNormAddMatmulPair => format!(
            "M={} N0={} N1={} K={} {} w_offset={}",
            i[0].shape[0].min(caps.rows),
            i[3].shape[0].min(caps.n),
            i[4].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            call.attrs.w_offset
        ),
        Op::GatedRmsnormMatmulFp8 => {
            format!("R={} N={} {} act_quant={}", i[0].shape[0], i[3].shape[0].min(caps.n), i[0].dtype, "static")
        }
        Op::MatmulFp8Pair => format!(
            "M={} N0={} N1={} K={} x={} out={} act_quant={}",
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[3].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            o[0].dtype,
            "static/static"
        ),
        Op::MatmulNvfp4Glu => format!(
            "M={} N={} K={} x={} out={}",
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            o[0].dtype
        ),
        Op::GatedRmsnorm => format!("R={} D={} {}", i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].dtype),
        Op::RmsNormAddMatmulFp8Pair => format!(
            "M={} N0={} N1={} K={} {} w_offset={} act_quant={}",
            i[0].shape[0].min(caps.rows),
            i[3].shape[0].min(caps.n),
            i[5].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            call.attrs.w_offset,
            "static/static"
        ),
        Op::SigmoidMulMatmulFp8 => format!(
            "M={} N={} K={} x={} out={} act_quant={}",
            i[0].shape[0].min(caps.rows),
            i[2].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            o[0].dtype,
            "static"
        ),
        Op::SharedDownCombine => format!(
            "M={} N={} K={} x={} out={}",
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            o[0].dtype
        ),
        Op::MoeBlock => format!(
            "T={} E={} K={} H={} I={} IS={} {} norm={}",
            i[0].shape[0].min(caps.rows),
            i[3].shape[0].min(32).max(call.attrs.top_k),
            call.attrs.top_k,
            i[0].shape[1],
            i[3].shape[1] / 2,
            i[8].shape[0],
            i[0].dtype,
            call.attrs.norm_topk
        ),
        _ => return None,
    };
    Some(format!("{} {}", call.op, s))
}

/// Host case that exercises the same kernel path as `call` (see [`label`]). The FP8 case alternates the
/// activation mode with the row count's parity so a plan that uses either mode gets it tested.
pub fn case_from_call(call: &OpCall, caps: &ShapeCaps, rng: &mut SplitMix64) -> Option<(HostCall, String)> {
    let label = label(call, caps)?;
    let (i, o) = (&call.inputs, &call.outputs);
    let hc = match call.op {
        Op::MatmulFp8 => {
            let (m, n, k) = (i[0].shape[0].min(caps.rows), i[1].shape[0].min(caps.n), i[1].shape[1]);
            // plan time sees shapes, not values: grade the W8A8 mode, the one the checkpoint declares
            matmul_fp8_case(rng, m, n, k, true, i[0].dtype, o[0].dtype)
        }
        Op::MatmulNvfp4 => {
            matmul_nvfp4_case(rng, i[0].shape[0].min(caps.rows), i[1].shape[0].min(caps.n), i[0].shape[1], i[0].dtype, o[0].dtype)
        }
        Op::MatmulMxfp4 => matmul_mxfp4_case(
            rng,
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[0].shape[1],
            Mxfp4Act::Dynamic,
            i[0].dtype,
            o[0].dtype,
        ),
        Op::MoeTopk => topk_case(rng, i[0].shape[0].min(caps.rows), i[0].shape[1], call.attrs.top_k, true, i[0].dtype, call.attrs.norm_topk),
        Op::MoeExperts => {
            let k = i[1].shape[1];
            let e = i[3].shape[0].min(32).max(k);
            experts_case(rng, i[0].shape[0].min(caps.rows), e, k, i[0].shape[1], i[3].shape[1] / 2, i[0].dtype)
        }
        Op::MoeCombine => combine_case(rng, i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].dtype),
        Op::MatmulPair => matmul_pair_case(
            rng,
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[2].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
        ),
        Op::RmsNormAddMatmulPair => rmsnorm_add_matmul_pair_case(
            rng,
            i[0].shape[0].min(caps.rows),
            i[3].shape[0].min(caps.n),
            i[4].shape[0].min(caps.n),
            i[0].shape[1],
            i[0].dtype,
            call.attrs.w_offset,
        ),
        Op::GatedRmsnormMatmulFp8 => gated_rmsnorm_matmul_fp8_case(rng, i[0].shape[0], i[3].shape[0].min(caps.n), true, i[0].dtype),
        Op::MatmulFp8Pair => matmul_fp8_pair_case(
            rng,
            i[0].shape[0].min(caps.rows),
            i[1].shape[0].min(caps.n),
            i[3].shape[0].min(caps.n),
            i[0].shape[1],
            true,
            true,
            i[0].dtype,
            o[0].dtype,
        ),
        Op::MatmulNvfp4Glu => {
            matmul_nvfp4_glu_case(rng, i[0].shape[0].min(caps.rows), i[1].shape[0].min(caps.n), i[0].shape[1], i[0].dtype, o[0].dtype)
        }
        Op::GatedRmsnorm => gated_rmsnorm_case(rng, i[0].shape[0].min(caps.rows), i[0].shape[1], i[0].dtype),
        // plan time sees shapes, not values: grade the W8A8 mode, the one the checkpoint declares
        Op::RmsNormAddMatmulFp8Pair => rmsnorm_add_matmul_fp8_pair_case(
            rng,
            i[0].shape[0].min(caps.rows),
            i[3].shape[0].min(caps.n),
            i[5].shape[0].min(caps.n),
            i[0].shape[1],
            true,
            true,
            i[0].dtype,
            call.attrs.w_offset,
        ),
        Op::SigmoidMulMatmulFp8 => sigmoid_mul_matmul_fp8_case(
            rng,
            i[0].shape[0].min(caps.rows),
            i[2].shape[0].min(caps.n),
            i[0].shape[1],
            true,
            i[0].dtype,
            o[0].dtype,
        ),
        Op::SharedDownCombine => {
            shared_down_combine_case(rng, i[0].shape[0].min(caps.rows), i[1].shape[0].min(caps.n), i[0].shape[1], i[0].dtype, o[0].dtype)
        }
        Op::MoeBlock => {
            let k = call.attrs.top_k;
            moe_block_case_with(
                rng,
                i[0].shape[0].min(caps.rows),
                i[3].shape[0].min(32).max(k),
                k,
                i[0].shape[1],
                i[3].shape[1] / 2,
                i[8].shape[0],
                i[0].dtype,
                call.attrs.norm_topk,
            )
        }
        _ => return None,
    };
    Some((hc, label))
}

/// Format ids this module generates cases for (documentation / tests).
pub const FORMATS: [&str; 3] = [FP8_E4M3, NVFP4, MXFP4];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_case_validates() {
        for op in [
            Op::MatmulFp8,
            Op::MatmulNvfp4,
            Op::MoeTopk,
            Op::MoeExperts,
            Op::MoeCombine,
            Op::MatmulPair,
            Op::MatmulFp8Pair,
            Op::MatmulNvfp4Glu,
            Op::GatedRmsnorm,
            Op::RmsNormAddMatmulPair,
            Op::GatedRmsnormMatmulFp8,
            Op::RmsNormAddMatmulFp8Pair,
            Op::SigmoidMulMatmulFp8,
            Op::SharedDownCombine,
            Op::MoeBlock,
            Op::MatmulMxfp4,
        ] {
            let cases = sweep(op, &SweepConfig { quick: true, ..SweepConfig::default() }).unwrap();
            assert!(!cases.is_empty());
            for c in cases {
                let mk = |t: &HostTensor| crate::backend::TensorRef::new(crate::backend::Buffer::new(P(t.nbytes())), t.dtype, &t.shape);
                let call = OpCall::new(
                    op,
                    c.call.inputs.iter().map(mk).collect(),
                    c.call.outputs.iter().map(|o| mk(&HostTensor::zeros(o.dtype, &o.shape))).collect(),
                    c.call.attrs.clone(),
                );
                call.check().unwrap_or_else(|e| panic!("{}: {e}", c.label));
            }
        }
    }

    struct P(usize);
    impl crate::backend::BufferImpl for P {
        fn bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}
