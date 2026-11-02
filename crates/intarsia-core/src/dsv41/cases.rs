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
                Op::HcSublayer => {
                    let m = (2 + s) * s;
                    for fold in [false, true] {
                        let mut inputs = vec![
                            rand_t(rng, dt, &[t, s, d], 1.0),
                            rand_t(rng, DType::F32, &[m, s * d], 0.05),
                            rand_t(rng, DType::F32, &[3], 0.5),
                            rand_t(rng, DType::F32, &[m], 0.5),
                            rand_t(rng, DType::F32, &[t, s], 0.7),
                            rand_t(rng, dt, &[d], 1.0),
                        ];
                        let h_out = if fold {
                            inputs.extend([rand_t(rng, dt, &[t, d], 1.0), rand_t(rng, DType::F32, &[t, s], 1.0), rand_t(rng, DType::F32, &[t, s, s], 0.5)]);
                            out(dt, &[t, s, d])
                        } else {
                            inputs.extend([HostTensor::zeros(dt, &[0]), HostTensor::zeros(DType::F32, &[0]), HostTensor::zeros(DType::F32, &[0])]);
                            out(dt, &[0])
                        };
                        let mut at = OpAttrs { eps: 1e-6, ..a.clone() };
                        (at.dsv.iters, at.dsv.hc_eps) = (20, 1e-6);
                        push(
                            format!("T={t} S={s} D={d} fold={fold} {dt}"),
                            op,
                            inputs,
                            vec![out(dt, &[t, d]), out(DType::F32, &[t, s]), out(DType::F32, &[t, s]), out(DType::F32, &[t, s, s]), h_out],
                            at,
                        );
                    }
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
                    // DeepSeek-V4.1's per-rank head shape (32 heads of 512, the 128-token window): the tensor-core path of
                    // the CUDA kernel (heads a multiple of 16), split-KV for T <= 16 and the direct prefill write above
                    {
                        let (h, d, w, nc, kc, pos0) = (32, 512, 128, 80, 64, 140);
                        let idx: Vec<i32> = (0..t * kc).map(|_| (rng.next_u64() % (nc as u64 + 8)) as i32 - 8).map(|v| v.max(-1)).collect();
                        let at = OpAttrs { window: w, pos0, scale: (d as f32).powf(-0.5), ..a.clone() };
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
                    // DeepSeek-V4.1's indexer shape (32 heads of 128, candidate blocks of 8): the CUDA tensor-core score
                    for (ratio, pos0) in [(1, 300), (2, 600)] {
                        let (hi, di, block) = (32, 128, 8);
                        let (nk, _) = compressed_rows(pos0, t, ratio);
                        let mut at = OpAttrs { pos0, scale: (di as f32).powf(-0.5) * (hi as f32).powf(-0.5), ..a.clone() };
                        (at.dsv.ratio, at.dsv.block) = (ratio, block);
                        let mut inputs = vec![rand_t(rng, dt, &[t, hi, di], 1.0), rand_t(rng, dt, &[nk, di], 1.0), rand_t(rng, dt, &[t, hi], 1.0)];
                        let nb = n_blocks(nk, block);
                        let (label, outputs) = if op == Op::IndexerBlocks {
                            at.top_k = 12;
                            (format!("T={t} Hi={hi} Di={di} Nk={nk} ratio={ratio} B={block} keep=12 pos0={pos0} {dt}"), vec![out(DType::U8, &[t, nb])])
                        } else {
                            at.top_k = 64;
                            inputs.push(bytes_t(rng, DType::U8, &[t, nb], |v| (v % 4 != 0) as u8));
                            (format!("T={t} Hi={hi} Di={di} Nk={nk} ratio={ratio} topk=64 pos0={pos0} cand=true {dt}"), vec![out(DType::I32, &[t, 64.min(nk)])])
                        };
                        push(label, op, inputs, outputs, at);
                    }
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
    for &t in &[1usize, 4] {
        for &dt in dts {
            dev_cases(op, t, dt, rng, &mut push);
        }
    }
    cases
}

/// Device-position cases (`DsvAttrs::host_dev_pos`: the sweep uploads it as `OpCall::pos`) of the positioned ops at
/// capacity shapes: no compressed rows yet, fewer rows than top_k, more, groups complete and not, outputs wider than the
/// rows written (the untouched / -1 / 0 regions are compared too: outputs start random).
fn dev_cases(op: Op, t: usize, dt: DType, rng: &mut SplitMix64, push: &mut impl FnMut(String, Op, Vec<HostTensor>, Vec<OutSpec>, OpAttrs)) {
    let dev = |p: usize, ratio: usize| {
        let mut a = OpAttrs { pos0: 0, ..OpAttrs::default() };
        (a.dsv.host_dev_pos, a.dsv.ratio) = (Some(p as i32), ratio);
        a
    };
    let d = 64usize;
    match op {
        Op::RopeTail => {
            for (stride, inverse, p, ratio) in [(1, false, 37, 1), (4, false, 37, 4), (2, true, 9, 2)] {
                let mut at = dev(p, ratio);
                (at.dsv.pos_stride, at.dsv.inverse) = (stride, inverse);
                let f: Vec<f32> = (0..32).map(|j| 10000f32.powf(-(2 * j) as f32 / 64.0)).collect();
                push(
                    format!("dev P={p} ratio={ratio} T={t} N=4 D=128 rd=64 stride={stride} inverse={inverse} {dt}"),
                    op,
                    vec![rand_t(rng, dt, &[t, 4, 128], 1.0), HostTensor::from_f32(&[32], &f)],
                    vec![out(dt, &[t, 4, 128])],
                    at,
                );
            }
        }
        Op::WindowKv => {
            let (w, wc) = (8, 12);
            for p in [0, 3, 21] {
                let at = OpAttrs { window: w, ..dev(p, 1) };
                push(
                    format!("dev P={p} T={t} D={d} W={w} Wc={wc} {dt}"),
                    op,
                    vec![rand_t(rng, dt, &[t, d], 1.0)],
                    vec![out(dt, &[w + t, d]), inout(rand_t(rng, dt, &[wc, d], 1.0))],
                    at,
                );
            }
        }
        Op::ActQdq => {
            // y [10,128]: rows P/ratio .. (the last case runs past y's end: only the rows that fit are written)
            for (fmt, block, sc, p, ratio) in [(QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0, 3, 1), (QdqFmt::Fp4E2m1, 16, QdqScale::E4m3, 13, 2), (QdqFmt::Fp4E2m1, 32, QdqScale::Ue8m0, 8, 1)] {
                let mut at = dev(p, ratio);
                (at.dsv.qdq_fmt, at.dsv.qdq_block, at.dsv.qdq_scale) = (fmt, block, sc);
                push(
                    format!("dev P={p} ratio={ratio} G={t} Ry=10 N=128 {fmt:?}/{block}/{sc:?} {dt}"),
                    op,
                    vec![rand_t(rng, dt, &[t, 128], 3.0)],
                    vec![inout(rand_t(rng, dt, &[10, 128], 1.0))],
                    at,
                );
            }
        }
        Op::CompressPool => {
            // (ratio, P, Gmax): ratio 2 at T = 1: P = 4 completes no group, P = 5 one; T = 4: P = 1 completes 2 (Gmax 1 clips)
            for (ratio, p, gmax) in [(2, 4, 2), (2, 5, 2), (2, 1, 1), (4, 7, 2)] {
                let rc = 2 * ratio;
                push(
                    format!("dev P={p} ratio={ratio} T={t} D={d} Rc={rc} Gmax={gmax} {dt}"),
                    op,
                    vec![rand_t(rng, DType::F32, &[t, d], 1.0), rand_t(rng, DType::F32, &[t, d], 1.0)],
                    vec![inout(rand_t(rng, dt, &[gmax, d], 1.0)), inout(rand_t(rng, DType::F32, &[rc, d], 1.0)), inout(rand_t(rng, DType::F32, &[rc, d], 1.0))],
                    dev(p, ratio),
                );
            }
        }
        Op::IndexerBlocks | Op::IndexerTopk => {
            // (Hi, Di, block, Ncap, ratio, P, top_k, candidates): N = min((P + T) / ratio, Ncap) from 0 up, Ncap > N
            let shapes: [(usize, usize, usize, usize, usize, usize, usize, bool); 7] = [
                (4, 32, 4, 40, 2, 0, 8, false),
                (4, 32, 4, 40, 1, 3, 8, false),
                (4, 32, 4, 40, 1, 3, 8, true),
                (4, 32, 4, 40, 1, 27, 8, true),
                (4, 32, 4, 40, 2, 41, 8, false),
                (32, 128, 8, 400, 1, 300, 64, true),
                (32, 128, 8, 400, 2, 600, 64, false),
            ];
            for (hi, di, block, ncap, ratio, p, top_k, with_cand) in shapes {
                let n = ((p + t) / ratio).min(ncap);
                let mut at = OpAttrs { scale: (di as f32).powf(-0.5) * (hi as f32).powf(-0.5), ..dev(p, ratio) };
                at.dsv.block = block;
                let mut inputs = vec![rand_t(rng, dt, &[t, hi, di], 1.0), rand_t(rng, dt, &[ncap, di], 1.0), rand_t(rng, dt, &[t, hi], 1.0)];
                let nb = n_blocks(ncap, block);
                let (label, outputs) = if op == Op::IndexerBlocks {
                    if with_cand {
                        continue;
                    }
                    at.top_k = if hi == 32 { 12 } else { 3 };
                    (format!("dev P={p} ratio={ratio} T={t} Hi={hi} Di={di} Ncap={ncap} N={n} B={block} keep={} {dt}", at.top_k), vec![inout(bytes_t(rng, DType::U8, &[t, nb], |v| v as u8))])
                } else {
                    at.top_k = top_k;
                    inputs.push(if with_cand { bytes_t(rng, DType::U8, &[t, nb], |v| (v % 4 != 0) as u8) } else { HostTensor::zeros(DType::U8, &[0]) });
                    let kw = top_k.min(ncap);
                    (
                        format!("dev P={p} ratio={ratio} T={t} Hi={hi} Di={di} Ncap={ncap} N={n} topk={top_k} cand={with_cand} {dt}"),
                        vec![inout(bytes_t(rng, DType::I32, &[t, kw], |v| v as u8))],
                    )
                };
                push(label, op, inputs, outputs, at);
            }
        }
        Op::AttnSparse => {
            // (H, D, W, Ncap, Kw, ratio, P): Ncap 0 (window only), Kc = min(Kw, N) below and at Kw, the model's shape
            for (h, dd, w, ncap, kw, ratio, p) in [(4, 64, 8, 0, 5, 1, 9), (4, 64, 8, 12, 5, 1, 2), (4, 64, 8, 12, 5, 1, 10), (4, 64, 8, 12, 5, 2, 4), (32, 512, 128, 80, 64, 2, 140), (32, 512, 128, 80, 64, 2, 60)] {
                let n = ((p + t) / ratio).min(ncap);
                let kc = if ncap > 0 { kw.min(n) } else { kw };
                // the first Kc columns hold -1 or a visible row; the columns past Kc are never read (garbage)
                let idx: Vec<i32> = (0..t * kw)
                    .map(|e| if e % kw >= kc { 9999 } else if n == 0 { -1 } else { (rng.next_u64() % (n as u64 + 2)) as i32 - 2 })
                    .map(|v| v.max(-1))
                    .collect();
                let at = OpAttrs { window: w, scale: (dd as f32).powf(-0.5), ..dev(p, ratio) };
                push(
                    format!("dev P={p} ratio={ratio} T={t} H={h} D={dd} W={w} Ncap={ncap} N={n} Kw={kw} Kc={kc} {dt}"),
                    op,
                    vec![
                        rand_t(rng, dt, &[t, h, dd], 1.0),
                        rand_t(rng, dt, &[w + t, dd], 1.0),
                        rand_t(rng, dt, &[ncap, dd], 1.0),
                        HostTensor::from_i32(&[t, kw], &idx),
                        rand_t(rng, DType::F32, &[h], 1.0),
                    ],
                    vec![out(dt, &[t, h, dd])],
                    at,
                );
            }
        }
        _ => {}
    }
}
