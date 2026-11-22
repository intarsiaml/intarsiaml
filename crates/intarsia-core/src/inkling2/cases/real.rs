//! Real-shape admission cases of the Inkling-2 mix / conv / in-projection ops (lane ink-m-mix), appended to `super::sweep`
//! outside `--quick`.

use super::{fp8_stack, inout, none, out, pos_t, rand_t, K, W};
use crate::admission::Case;
use crate::dtype::DType;
use crate::exec::{HostCall, OutSpec};
use crate::ops::{Op, OpAttrs};
use crate::rng::SplitMix64;
use crate::tensor::HostTensor;

/// The real model's mix / conv / in-projection shapes beyond the one-row cases above (lane ink-m-mix): D 2304, the conv ring of
/// `K-1 + 16` rows read across a wrap (pos0 past the ring), up to 7 stored sources, 1 or 3 queries, 3 blend streams of rank 64,
/// verify rows 3 / 7 / 16 and one 2048-row prefill chunk (longer than the ring: the rows it writes are rows it reads).
pub(super) fn real_cases(op: Op, rng: &mut SplitMix64) -> Vec<Case> {
    const RD: usize = 2304;
    const RG: usize = 64;
    let rr = K - 1 + 16;
    let bf = DType::Bf16;
    let a0 = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
    let mut cases = Vec::new();
    let mut push = |label: String, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs| {
        cases.push(Case { label, call: HostCall { op, inputs, outputs, attrs } })
    };
    let rows = [3usize, 7, 16, 2048];
    match op {
        Op::InkInProj => {
            let ns = [6552usize, 1056, 1056];
            for &t in &rows {
                for s in [1usize, 3] {
                    if t == 2048 && s == 1 {
                        continue;
                    }
                    let inputs = vec![rand_t(rng, bf, &[s, t, RD], 1.0), rand_t(rng, W, &[ns.iter().sum(), RD], (RD as f32).powf(-0.5))];
                    push(format!("real T={t} S={s} D={RD} N={ns:?} bf16"), inputs, ns.iter().map(|&n| out(bf, &[t, n])).collect(), a0.clone());
                }
            }
            // e4m3 weights with per-pack scales (lane ink-fp8: INTARSIA_INK_ATTN_QUANT=fp8), decode / verify / prefill rows
            for &t in &[1usize, 4, 16, 2048] {
                for s in [1usize, 3] {
                    if t == 2048 && s == 1 {
                        continue;
                    }
                    let (w, sc) = fp8_stack(rng, ns, RD, (RD as f32).powf(-0.5));
                    let inputs = vec![rand_t(rng, bf, &[s, t, RD], 1.0), w, sc];
                    push(format!("real T={t} S={s} D={RD} N={ns:?} bf16 e4m3"), inputs, ns.iter().map(|&n| out(bf, &[t, n])).collect(), a0.clone());
                }
            }
        }
        Op::AttnResMix => {
            for &t in &rows {
                for &(n, q, addend, norm) in &[(4usize, 1usize, true, true), (7, 3, false, true), (5, 4, false, false), (2, 1, true, false)] {
                    if t == 2048 && n != 4 {
                        continue;
                    }
                    let inputs = vec![
                        rand_t(rng, bf, &[n, t, RD], 1.5),
                        rand_t(rng, bf, &[t, RD], 1.0),
                        if addend { rand_t(rng, bf, &[t, RD], 1.0) } else { none() },
                        rand_t(rng, DType::F32, &[q, RD], 0.2 * (RD as f32).powf(-0.5) * 8.0),
                        if norm { pos_t(rng, W, &[q, RD], 0.5, 1.5) } else { none() },
                    ];
                    let o = if norm { out(bf, &[q, t, RD]) } else { out(DType::F32, &[q, t, RD]) };
                    push(format!("real R={t} N={n} Q={q} D={RD} addend={addend} norm={norm} bf16"), inputs, vec![o], a0.clone());
                }
            }
        }
        Op::LoopBlend => {
            for &t in &rows {
                for &(s, sd) in &[(3usize, 1usize), (3, 3), (1, 1)] {
                    if t == 2048 && s == 1 {
                        continue;
                    }
                    let inputs = vec![
                        rand_t(rng, DType::F32, &[sd, t, RD], 1.0),
                        rand_t(rng, DType::F32, &[s, t, RD], 1.0),
                        rand_t(rng, W, &[s, RG, 2 * RD], (2.0 * RD as f32).powf(-0.5)),
                        rand_t(rng, W, &[s, RD, RG], (RG as f32).powf(-0.5) * 2.0),
                        rand_t(rng, W, &[s, RD, RG], (RG as f32).powf(-0.5) * 2.0),
                        pos_t(rng, W, &[s, RD], 0.5, 1.5),
                    ];
                    push(format!("real R={t} S={s} Sd={sd} D={RD} G={RG} bf16"), inputs, vec![out(bf, &[s, t, RD])], a0.clone());
                }
            }
        }
        Op::SconvRes | Op::ConvMix => {
            for &t in &[1usize, 3, 7, 16, 2048] {
                for pos0 in [0usize, 29] {
                    // (res, addend, sum), then the mix variants (N, Q, norm, blend streams)
                    let epis: &[(bool, bool, bool)] = if op == Op::SconvRes { &[(true, true, true)] } else { &[(true, true, true), (true, false, false)] };
                    for &(res, add, sum) in epis {
                        let base = vec![
                            rand_t(rng, bf, &[t, RD], 1.0),
                            rand_t(rng, W, &[K, RD], 0.5),
                            if res { rand_t(rng, bf, &[t, RD], 1.0) } else { none() },
                            if add { rand_t(rng, bf, &[t, RD], 1.0) } else { none() },
                        ];
                        let obase = vec![out(bf, &[t, RD]), inout(rand_t(rng, bf, &[rr, RD], 1.0)), if sum { out(bf, &[t, RD]) } else { out(bf, &[0]) }];
                        let a = OpAttrs { pos0, ..a0.clone() };
                        if op == Op::SconvRes {
                            push(format!("real T={t} C={RD} K={K} Rr={rr} pos0={pos0} res={res} add={add} bf16"), base, obase, a);
                            continue;
                        }
                        let mixes: &[(usize, usize, bool, usize)] = if t == 2048 { &[(4, 1, true, 0), (3, 1, false, 3)] } else { &[(7, 3, true, 0), (4, 1, true, 0), (3, 1, false, 3), (4, 3, false, 3), (2, 1, false, 1)] };
                        for &(n, q, norm, s) in mixes {
                            let mut ins = base.clone();
                            ins.push(rand_t(rng, bf, &[n, t, RD], 1.5));
                            ins.push(rand_t(rng, DType::F32, &[q, RD], 0.2 * (RD as f32).powf(-0.5) * 8.0));
                            let nrows = if s > 0 { s } else { q };
                            ins.push(pos_t(rng, W, &[nrows, RD], 0.5, 1.5));
                            if s > 0 {
                                ins.push(rand_t(rng, DType::F32, &[s, t, RD], 1.0));
                                ins.push(rand_t(rng, W, &[s, RG, 2 * RD], (2.0 * RD as f32).powf(-0.5)));
                                ins.push(rand_t(rng, W, &[s, RD, RG], (RG as f32).powf(-0.5) * 2.0));
                                ins.push(rand_t(rng, W, &[s, RD, RG], (RG as f32).powf(-0.5) * 2.0));
                            } else {
                                ins.extend([none(), none(), none(), none()]);
                            }
                            let mut outs = obase.clone();
                            outs.push(out(bf, &[if s > 0 { s } else { q }, t, RD]));
                            push(format!("real T={t} C={RD} Rr={rr} pos0={pos0} res={res} add={add} sum={sum} N={n} Q={q} norm={norm} S={s} bf16"), ins, outs, a.clone());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    cases
}
