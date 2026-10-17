//! Admission cases for the quantised-weight and MoE ops and the fused decode ops (`MatmulPair`, `MatmulFp8Pair`,
//! `MatmulNvfp4Glu`, `GatedRmsnorm`, `RmsNormAddMatmulPair`, `GatedRmsnormMatmulFp8`): the `intarsia kernels test` sweep and the exact-shape
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
use crate::quant::{fp8, nvfp4, FP8_E4M3, NVFP4};
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
        _ => return None,
    };
    Some((hc, label))
}

/// Format ids this module generates cases for (documentation / tests).
pub const FORMATS: [&str; 2] = [FP8_E4M3, NVFP4];

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
