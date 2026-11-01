//! Admission sweep cases of the DeepSeek-V4.1 ops (`admission::sweep` hands them here): small shapes that exercise every
//! rule of the contract (positions before the sequence start, ring wrap-around, groups spanning chunks, -1 indices,
//! candidate masks). A stub op still gets cases, so admission reports its "not implemented" error instead of passing
//! vacuously.

use super::ops::{compressed_rows, n_blocks, QdqFmt, QdqScale};
use crate::admission::{Case, SweepConfig};
use crate::dtype::DType;
use crate::exec::{HostCall, OutSpec};
use crate::ops::{Op, OpAttrs};
use crate::rng::SplitMix64;
use crate::tensor::HostTensor;

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

fn bytes_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], f: impl Fn(u64) -> u8) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::new(dt, shape.to_vec(), (0..n * dt.size_bytes()).map(|_| f(rng.next_u64())).collect()).expect("sized")
}

fn out(dt: DType, shape: &[usize]) -> OutSpec {
    OutSpec::new(dt, shape)
}

fn inout(t: HostTensor) -> OutSpec {
    OutSpec { dtype: t.dtype, shape: t.shape.clone(), init: Some(t) }
}

pub fn sweep(op: Op, cfg: &SweepConfig) -> Vec<Case> {
    let mut rng = SplitMix64::new(cfg.seed ^ (op as u64).wrapping_mul(0x9E37));
    let rng = &mut rng;
    let rows: &[usize] = if cfg.quick { &[1, 5] } else { &[1, 5, 17] };
    let dts: &[DType] = if cfg.quick { &[DType::Bf16] } else { &[DType::Bf16, DType::F32] };
    let mut cases = Vec::new();
    let mut push = |label: String, op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs| {
        cases.push(Case { label, call: HostCall { op, inputs, outputs, attrs } })
    };
    let (s, d) = (4usize, 64usize);
    for &t in rows {
        for &dt in dts {
            let a = OpAttrs::default();
            match op {
                Op::HcExpand => push(format!("T={t} S={s} D={d} {dt}"), op, vec![rand_t(rng, dt, &[t, d], 1.0)], vec![out(dt, &[t, s, d])], a),
                Op::HcMixes => {
                    let m = (2 + s) * s;
                    let inputs = vec![
                        rand_t(rng, dt, &[t, s, d], 1.0),
                        rand_t(rng, DType::F32, &[m, s * d], 0.05),
                        rand_t(rng, DType::F32, &[3], 0.5),
                        rand_t(rng, DType::F32, &[m], 0.5),
                    ];
                    let at = OpAttrs { eps: 1e-20, ..a };
                    push(format!("T={t} S={s} D={d} {dt}"), op, inputs, vec![out(DType::F32, &[t, s]), out(DType::F32, &[t, s]), out(DType::F32, &[t, s, s])], at);
                }
                Op::HcPre => push(
                    format!("T={t} S={s} D={d} {dt}"),
                    op,
                    vec![rand_t(rng, dt, &[t, s, d], 1.0), rand_t(rng, DType::F32, &[t, s], 0.7)],
                    vec![out(dt, &[t, d])],
                    a,
                ),
                Op::HcPost => push(
                    format!("T={t} S={s} D={d} {dt}"),
                    op,
                    vec![
                        rand_t(rng, dt, &[t, d], 1.0),
                        rand_t(rng, dt, &[t, s, d], 1.0),
                        rand_t(rng, DType::F32, &[t, s], 1.0),
                        rand_t(rng, DType::F32, &[t, s, s], 0.5),
                    ],
                    vec![out(dt, &[t, s, d])],
                    a,
                ),
                Op::MoeRoute => {
                    let (e, k) = (64, 6);
                    let at = OpAttrs { top_k: k, scale: 1.5, norm_topk: true, ..a };
                    push(
                        format!("T={t} E={e} K={k}"),
                        op,
                        vec![rand_t(rng, DType::F32, &[t, e], 2.0), rand_t(rng, DType::F32, &[e], 0.1)],
                        vec![out(DType::F32, &[t, k]), out(DType::I32, &[t, k])],
                        at,
                    );
                }
                Op::SiluGluClamp => {
                    let mut at = a;
                    at.dsv.limit = 10.0;
                    push(format!("T={t} I=96 limit=10 {dt}"), op, vec![rand_t(rng, dt, &[t, 96], 8.0), rand_t(rng, dt, &[t, 96], 8.0)], vec![out(dt, &[t, 96])], at);
                }
                Op::ActQdq => {
                    for (fmt, block, sc) in [(QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0), (QdqFmt::Fp4E2m1, 16, QdqScale::E4m3), (QdqFmt::Fp4E2m1, 32, QdqScale::Ue8m0)] {
                        let mut at = a.clone();
                        (at.dsv.qdq_fmt, at.dsv.qdq_block, at.dsv.qdq_scale) = (fmt, block, sc);
                        push(format!("T={t} N=128 {fmt:?}/{block}/{sc:?} {dt}"), op, vec![rand_t(rng, dt, &[t, 128], 3.0)], vec![out(dt, &[t, 128])], at);
                    }
                }
                Op::RopeTail => {
                    for (stride, inverse, pos0) in [(1, false, 0), (1, true, 7), (2, false, 64)] {
                        let mut at = OpAttrs { pos0, ..a.clone() };
                        (at.dsv.pos_stride, at.dsv.inverse) = (stride, inverse);
                        let f: Vec<f32> = (0..32).map(|j| 10000f32.powf(-(2 * j) as f32 / 64.0)).collect();
                        push(
                            format!("T={t} N=4 D=128 rd=64 stride={stride} inverse={inverse} pos0={pos0} {dt}"),
                            op,
                            vec![rand_t(rng, dt, &[t, 4, 128], 1.0), HostTensor::from_f32(&[32], &f)],
                            vec![out(dt, &[t, 4, 128])],
                            at,
                        );
                    }
                }
                Op::WindowKv => {
                    let (w, wc) = (8, 12);
                    for pos0 in [0, 3, 21] {
                        let at = OpAttrs { window: w, pos0, ..a.clone() };
                        push(
                            format!("T={t} D={d} W={w} Wc={wc} pos0={pos0} {dt}"),
                            op,
                            vec![rand_t(rng, dt, &[t, d], 1.0)],
                            vec![out(dt, &[w + t, d]), inout(rand_t(rng, dt, &[wc, d], 1.0))],
                            at,
                        );
                    }
                }
                Op::AttnSparse => {
                    let (h, w, nc, kc) = (4, 8, 12, 5);
                    for pos0 in [0, 10] {
                        let idx: Vec<i32> = (0..t * kc).map(|_| (rng.next_u64() % (nc as u64 + 3)) as i32 - 3).map(|v| v.max(-1)).collect();
                        let at = OpAttrs { window: w, pos0, scale: 0.125, ..a.clone() };
                        push(
                            format!("T={t} H={h} D={d} W={w} Nc={nc} Kc={kc} pos0={pos0} {dt}"),
                            op,
                            vec![
                                rand_t(rng, dt, &[t, h, d], 1.0),
                                rand_t(rng, dt, &[w + t, d], 1.0),
                                rand_t(rng, dt, &[nc, d], 1.0),
                                HostTensor::from_i32(&[t, kc], &idx),
                                rand_t(rng, DType::F32, &[h], 1.0),
                            ],
                            vec![out(dt, &[t, h, d])],
                            at,
                        );
                    }
                }
                Op::CompressPool => {
                    for (ratio, pos0) in [(2, 0), (2, 3), (4, 7)] {
                        let rc = 2 * ratio;
                        let (_, g) = compressed_rows(pos0, t, ratio);
                        let mut at = OpAttrs { pos0, ..a.clone() };
                        at.dsv.ratio = ratio;
                        push(
                            format!("T={t} D={d} ratio={ratio} Rc={rc} pos0={pos0} {dt}"),
                            op,
                            vec![rand_t(rng, DType::F32, &[t, d], 1.0), rand_t(rng, DType::F32, &[t, d], 1.0)],
                            vec![out(dt, &[g, d]), inout(rand_t(rng, DType::F32, &[rc, d], 1.0)), inout(rand_t(rng, DType::F32, &[rc, d], 1.0))],
                            at,
                        );
                    }
                }
                Op::IndexerBlocks | Op::IndexerTopk => {
                    let (hi, di, block) = (4, 32, 4);
                    for (ratio, pos0) in [(1, 0), (2, 30), (1, 40)] {
                        let (nk, _) = compressed_rows(pos0, t, ratio);
                        if nk == 0 {
                            continue;
                        }
                        let mut at = OpAttrs { pos0, scale: 0.0883, ..a.clone() };
                        (at.dsv.ratio, at.dsv.block) = (ratio, block);
                        let mut inputs = vec![rand_t(rng, dt, &[t, hi, di], 1.0), rand_t(rng, dt, &[nk, di], 1.0), rand_t(rng, dt, &[t, hi], 1.0)];
                        let nb = n_blocks(nk, block);
                        let (label, outputs) = if op == Op::IndexerBlocks {
                            at.top_k = 3;
                            (format!("T={t} Hi={hi} Di={di} Nk={nk} ratio={ratio} B={block} keep=3 pos0={pos0} {dt}"), vec![out(DType::U8, &[t, nb])])
                        } else {
                            at.top_k = 8;
                            let cand = if pos0 == 40 { bytes_t(rng, DType::U8, &[t, nb], |v| (v % 3 != 0) as u8) } else { HostTensor::zeros(DType::U8, &[0]) };
                            inputs.push(cand);
                            (format!("T={t} Hi={hi} Di={di} Nk={nk} ratio={ratio} topk=8 pos0={pos0} cand={} {dt}", pos0 == 40), vec![out(DType::I32, &[t, 8.min(nk)])])
                        };
                        push(label, op, inputs, outputs, at);
                    }
                }
                Op::GroupedMatmul => push(
                    format!("T={t} G=4 K=32 N=16 {dt}"),
                    op,
                    vec![rand_t(rng, dt, &[t, 4 * 32], 1.0), rand_t(rng, dt, &[4, 16, 32], 0.2)],
                    vec![out(dt, &[t, 4 * 16])],
                    a,
                ),
                Op::Transpose01 => push(format!("A=3 B={t} C=16 {dt}"), op, vec![rand_t(rng, dt, &[3, t, 16], 1.0)], vec![out(dt, &[t, 3, 16])], a),
                Op::MatmulFp8Block => {
                    let mut at = a;
                    at.dsv.block = 32;
                    push(
                        format!("M={t} N=32 K=64 {dt}"),
                        op,
                        vec![rand_t(rng, dt, &[t, 64], 1.0), bytes_t(rng, DType::Fp8E4m3, &[32, 64], |v| (v % 0x7E) as u8), bytes_t(rng, DType::U8, &[1, 2], |v| 120 + (v % 8) as u8)],
                        vec![out(dt, &[t, 32])],
                        at,
                    );
                }
                Op::MoeExpertsExl3 => {
                    let (e, k, h, i, bits) = (4usize, 2usize, 128usize, 128usize, 3usize);
                    let ids: Vec<i32> = (0..t * k).map(|j| ((j * 3 + 1) % e) as i32).collect();
                    let mut inputs = vec![rand_t(rng, dt, &[t, h], 1.0), HostTensor::from_i32(&[t, k], &ids), rand_t(rng, DType::F32, &[t, k], 0.5)];
                    for (kk, nn) in [(h, i), (h, i), (i, h)] {
                        inputs.push(bytes_t(rng, DType::I16, &[e, kk / 16, nn / 16, 16 * bits], |v| v as u8));
                        inputs.push(rand_t(rng, DType::F16, &[e, kk], 0.1));
                        inputs.push(rand_t(rng, DType::F16, &[e, nn], 1.0));
                    }
                    inputs.push(HostTensor::from_i32(&[1], &[2]));
                    let mut at = OpAttrs { top_k: k, ..a };
                    at.dsv.limit = 10.0;
                    push(format!("T={t} E={e} K={k} H={h} I={i} bits={bits} {dt}"), op, inputs, vec![out(dt, &[t, h])], at);
                }
                Op::EngramGather => {
                    // records of Dh 64: 64 e4m3 bytes (finite, either sign), then 2 E8M0 scales near 2^0
                    let mut rec = bytes_t(rng, DType::U8, &[t, 3, 66], |v| (v % 0x7E) as u8 | ((v >> 8) as u8 & 0x80));
                    for (j, b) in rec.data.iter_mut().enumerate() {
                        if j % 66 >= 64 {
                            *b = 120 + *b % 10;
                        }
                    }
                    push(format!("T={t} C=3 Dh=64 {dt}"), op, vec![rec], vec![out(dt, &[t, 3 * 64])], a)
                }
                Op::EngramMerge => push(
                    format!("T={t} S={s} D={d} {dt}"),
                    op,
                    vec![rand_t(rng, dt, &[t, s, d], 1.0), rand_t(rng, dt, &[t, (s + 1) * d], 1.0), rand_t(rng, dt, &[s, d], 1.0), rand_t(rng, dt, &[s, d], 1.0)],
                    vec![out(dt, &[t, s, d])],
                    OpAttrs { eps: 1e-20, ..a },
                ),
                _ => {}
            }
        }
    }
    cases
}
