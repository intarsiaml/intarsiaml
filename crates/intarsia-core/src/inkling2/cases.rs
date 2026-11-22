//! Admission sweep cases of the Inkling-2 ops (`admission::sweep` hands them here). Rows: decode (1), verify (3, 7, 16) and a
//! prefill chunk longer than the rings (40); positions before, at and past a ring wrap, past the window, past the rel extent
//! and past the tau floor; static and dynamic convs; every optional operand present and absent. Small shapes (D 64, 6 q
//! heads over 2 kv heads, head_dim 16, d_rel 8 so that a head_dim / d_rel stride mix-up shows) plus the real model's widths
//! once (outside `--quick`).

use super::ops::nq;
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

fn pos_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], lo: f32, hi: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| lo + (hi - lo) * rng.next_f32()).collect();
    HostTensor::from_f32_as(dt, shape, &v)
}

fn none() -> HostTensor {
    HostTensor::zeros(DType::Bf16, &[0])
}

fn out(dt: DType, shape: &[usize]) -> OutSpec {
    OutSpec::new(dt, shape)
}

fn inout(t: HostTensor) -> OutSpec {
    OutSpec { dtype: t.dtype, shape: t.shape.clone(), init: Some(t) }
}

const W: DType = DType::Bf16;

/// Small model shape: hidden, q heads, kv heads, head dim, d_rel, conv K, controller rank.
const D: usize = 64;
const H: usize = 6;
const HK: usize = 2;
const DH: usize = 16;
const DR: usize = 8;
const K: usize = 4;
const G: usize = 8;

pub fn sweep(op: Op, cfg: &SweepConfig) -> Vec<Case> {
    let mut rng = SplitMix64::new(cfg.seed ^ (op as u64).wrapping_mul(0x9E37));
    let rng = &mut rng;
    let quick = cfg.quick;
    let rows: &[usize] = if quick { &[1, 7, 40] } else { &[1, 3, 7, 16, 40] };
    let dts: &[DType] = if quick { &[DType::Bf16] } else { &[DType::Bf16, DType::F32] };
    let mut cases = Vec::new();
    let mut push = |label: String, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs| {
        cases.push(Case { label, call: HostCall { op, inputs, outputs, attrs } })
    };
    let a0 = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
    match op {
        Op::AttnResMix => {
            let mut shapes: Vec<(usize, usize, usize)> = vec![(0, 1, D), (1, 3, D), (4, 4, D), (3, 1, D)];
            if !quick {
                shapes.push((4, 48, 2304));
            }
            for &t in rows {
                for &dt in dts {
                    for &(n, q, d) in &shapes {
                        if d > D && t > 1 {
                            continue;
                        }
                        for (addend, norm) in [(false, true), (true, true), (false, false), (true, false)] {
                            let inputs = vec![
                                rand_t(rng, dt, &[n, t, d], 1.5),
                                rand_t(rng, dt, &[t, d], 1.0),
                                if addend { rand_t(rng, dt, &[t, d], 1.0) } else { none() },
                                rand_t(rng, DType::F32, &[q, d], 0.2),
                                if norm { pos_t(rng, W, &[q, d], 0.5, 1.5) } else { none() },
                            ];
                            let o = if norm { out(dt, &[q, t, d]) } else { out(DType::F32, &[q, t, d]) };
                            push(format!("R={t} N={n} Q={q} D={d} addend={addend} norm={norm} {dt}"), inputs, vec![o], a0.clone());
                        }
                    }
                }
            }
        }
        Op::LoopBlend => {
            let mut shapes: Vec<(usize, usize, usize, usize)> = vec![(1, 1, D, G), (3, 1, D, G), (3, 3, D, G)];
            if !quick {
                shapes.push((3, 3, 2304, 64));
            }
            for &t in rows {
                for &dt in dts {
                    for &(s, sd, d, g) in &shapes {
                        if d > D && t > 1 {
                            continue;
                        }
                        let inputs = vec![
                            rand_t(rng, DType::F32, &[sd, t, d], 1.0),
                            rand_t(rng, DType::F32, &[s, t, d], 1.0),
                            rand_t(rng, W, &[s, g, 2 * d], (2.0 * d as f32).powf(-0.5)),
                            rand_t(rng, W, &[s, d, g], (g as f32).powf(-0.5) * 2.0),
                            rand_t(rng, W, &[s, d, g], (g as f32).powf(-0.5) * 2.0),
                            pos_t(rng, W, &[s, d], 0.5, 1.5),
                        ];
                        push(format!("R={t} S={s} Sd={sd} D={d} G={g} {dt}"), inputs, vec![out(dt, &[s, t, d])], a0.clone());
                    }
                }
            }
        }
        Op::SconvRes | Op::ConvMix => {
            let rr = K - 1 + 4;
            let mut widths = vec![D];
            if !quick {
                widths.push(2304);
            }
            for &t in rows {
                for &dt in dts {
                    for &c in &widths {
                        if c > D && t > 1 {
                            continue;
                        }
                        for pos0 in [0usize, 2, 29] {
                            // (res, addend, sum)
                            let epis: &[(bool, bool, bool)] =
                                if op == Op::SconvRes { &[(false, false, false), (true, false, true), (true, true, true)] } else { &[(false, false, false), (true, false, false), (true, false, true), (true, true, true)] };
                            for &(res, add, sum) in epis {
                                let mut inputs = vec![
                                    rand_t(rng, dt, &[t, c], 1.0),
                                    rand_t(rng, W, &[K, c], 0.5),
                                    if res { rand_t(rng, dt, &[t, c], 1.0) } else { none() },
                                    if add { rand_t(rng, dt, &[t, c], 1.0) } else { none() },
                                ];
                                let mut outputs = vec![out(dt, &[t, c]), inout(rand_t(rng, dt, &[rr, c], 1.0)), if sum { out(dt, &[t, c]) } else { out(DType::Bf16, &[0]) }];
                                let a = OpAttrs { pos0, ..a0.clone() };
                                if op == Op::SconvRes {
                                    push(format!("T={t} C={c} K={K} Rr={rr} pos0={pos0} res={res} add={add} {dt}"), inputs, outputs, a);
                                    continue;
                                }
                                // the mix variants: (N, Q, norm, blend streams)
                                let mixes: &[(usize, usize, bool, usize)] = if c > D { &[(4, 3, true, 0), (3, 1, false, 3)] } else { &[(0, 0, false, 0), (2, 1, true, 0), (3, 3, true, 0), (1, 1, false, 0), (2, 1, false, 1), (3, 1, false, 3), (3, 3, false, 3)] };
                                for &(n, q, norm, s) in mixes {
                                    let mut ins = inputs.clone();
                                    ins.push(rand_t(rng, dt, &[n, t, c], 1.5));
                                    ins.push(rand_t(rng, DType::F32, &[q, c], 0.2));
                                    let nrows = if s > 0 { s } else { q };
                                    ins.push(if norm || s > 0 { pos_t(rng, W, &[nrows, c], 0.5, 1.5) } else { none() });
                                    if s > 0 {
                                        ins.push(rand_t(rng, DType::F32, &[s, t, c], 1.0));
                                        ins.push(rand_t(rng, W, &[s, G, 2 * c], (2.0 * c as f32).powf(-0.5)));
                                        ins.push(rand_t(rng, W, &[s, c, G], 0.7));
                                        ins.push(rand_t(rng, W, &[s, c, G], 0.7));
                                    } else {
                                        ins.extend([none(), none(), none(), none()]);
                                    }
                                    let mut outs = outputs.clone();
                                    outs.push(if q == 0 {
                                        out(DType::Bf16, &[0])
                                    } else if s > 0 {
                                        out(dt, &[s, t, c])
                                    } else if norm {
                                        out(dt, &[q, t, c])
                                    } else {
                                        out(DType::F32, &[q, t, c])
                                    });
                                    push(
                                        format!("T={t} C={c} pos0={pos0} res={res} add={add} sum={sum} N={n} Q={q} norm={norm} S={s} {dt}"),
                                        ins,
                                        outs,
                                        a.clone(),
                                    );
                                }
                                inputs.clear();
                                outputs.clear();
                            }
                        }
                    }
                }
            }
        }
        Op::InkKvPrologue => {
            // (H, Hk, Dh, dr, E)
            let mut shapes: Vec<(usize, usize, usize, usize, usize)> = vec![(H, HK, DH, DR, 8), (H, HK, DH, DR, 12)];
            if !quick {
                shapes.push((24, 8, 128, 16, 1024));
            }
            for &t in rows {
                for &dt in dts {
                    for &(h, hk, dh, dr, e) in &shapes {
                        if dh > DH && t > 1 {
                            continue;
                        }
                        let ckv = hk * dh;
                        let rr = K - 1 + 4;
                        for pos0 in [0usize, 3, 29] {
                            for (dynamic, tau) in [(false, false), (true, true), (true, false)] {
                                let pw = if dynamic { ckv + hk * K } else { ckv };
                                let inputs = vec![
                                    rand_t(rng, dt, &[t, nq(h, dh, dr)], 1.0),
                                    rand_t(rng, dt, &[t, pw], 1.0),
                                    rand_t(rng, dt, &[t, pw], 1.0),
                                    rand_t(rng, W, &[K, ckv], 0.5),
                                    rand_t(rng, W, &[K, ckv], 0.5),
                                    pos_t(rng, W, &[h, dh], 0.2, 3.0),
                                    pos_t(rng, W, &[h, dr], 0.2, 1.5),
                                    rand_t(rng, W, &[dr, e], 0.25),
                                ];
                                let outputs = vec![
                                    out(dt, &[t, h, dh]),
                                    out(dt, &[t, h, e]),
                                    out(dt, &[t, hk, dh]),
                                    out(dt, &[t, hk, dh]),
                                    inout(rand_t(rng, dt, &[rr, ckv], 1.0)),
                                    inout(rand_t(rng, dt, &[rr, ckv], 1.0)),
                                ];
                                let mut a = OpAttrs { pos0, n_heads: h, n_kv_heads: hk, head_dim: dh, ..a0.clone() };
                                // thresholds that clip a share of the rows (head_dim 16: |q| is around 4 x the scale)
                                a.ink.qk_clip = if dh > DH { 16.0 * 2f32.sqrt() } else { 3.0 };
                                a.ink.rel_clip = 2.0;
                                a.ink.dyn_conv = dynamic;
                                a.ink.d_rel = dr;
                                if tau {
                                    (a.ink.tau_floor, a.ink.tau_alpha) = (4, 0.1);
                                }
                                push(format!("T={t} H={h} Hk={hk} Dh={dh} dr={dr} E={e} pos0={pos0} dyn={dynamic} tau={tau} {dt}"), inputs, outputs, a);
                            }
                        }
                    }
                }
            }
            if !quick {
                // the real model at decode / verify rows and one prefill chunk: K 4, conv ring 3 + 16 rows, tau floor 1024
                // (positions before, at and far past it), positions that wrap the conv ring
                let (h, hk, dh, dr, e, kk) = (24usize, 8usize, 128usize, 16usize, 1024usize, 4usize);
                let (ckv, rr) = (hk * dh, kk - 1 + 16);
                for (t, pos0s) in [(1usize, vec![0usize, 1023, 8192]), (4, vec![1021, 8190]), (16, vec![5, 1016, 8192]), (2048, vec![2048])] {
                    for pos0 in pos0s {
                        for dynamic in [false, true] {
                            let pw = if dynamic { ckv + hk * kk } else { ckv };
                            let inputs = vec![
                                rand_t(rng, W, &[t, nq(h, dh, dr)], 1.0),
                                rand_t(rng, W, &[t, pw], 1.0),
                                rand_t(rng, W, &[t, pw], 1.0),
                                rand_t(rng, W, &[kk, ckv], 0.5),
                                rand_t(rng, W, &[kk, ckv], 0.5),
                                pos_t(rng, W, &[h, dh], 0.2, 3.0),
                                pos_t(rng, W, &[h, dr], 0.2, 1.5),
                                rand_t(rng, W, &[dr, e], 0.25),
                            ];
                            let outputs = vec![
                                out(W, &[t, h, dh]),
                                out(W, &[t, h, e]),
                                out(W, &[t, hk, dh]),
                                out(W, &[t, hk, dh]),
                                inout(rand_t(rng, W, &[rr, ckv], 1.0)),
                                inout(rand_t(rng, W, &[rr, ckv], 1.0)),
                            ];
                            let mut a = OpAttrs { pos0, n_heads: h, n_kv_heads: hk, head_dim: dh, ..a0.clone() };
                            a.ink.qk_clip = 16.0 * 2f32.sqrt();
                            a.ink.rel_clip = 2.0;
                            a.ink.dyn_conv = dynamic;
                            a.ink.d_rel = dr;
                            if dynamic {
                                (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
                            }
                            push(format!("real T={t} pos0={pos0} dyn={dynamic} tau={dynamic} Rr={rr} bf16"), inputs, outputs, a);
                        }
                    }
                }
            }
        }
        Op::RelSinkAttn => {
            // (H, Hk, Dh, dr, window, E, ring, cache rows, positions)
            let mut shapes: Vec<(usize, usize, usize, usize, usize, usize, bool, usize, Vec<usize>)> = vec![
                (H, HK, DH, DR, 8, 8, true, 8 + 4, vec![0, 5, 29]),
                (H, HK, DH, DR, 0, 12, false, 96, vec![0, 5, 29]),
                (H, HK, DH, DR, 8, 12, false, 96, vec![3, 29]),
            ];
            if !quick {
                shapes.push((24, 8, 128, 16, 1024, 1024, true, 1040, vec![1500]));
                shapes.push((24, 8, 128, 16, 0, 1024, false, 1600, vec![1500]));
            }
            for &t in rows {
                for &dt in dts {
                    for (h, hk, dh, dr, win, e, ring, cr, poss) in &shapes {
                        let (h, hk, dh, dr, win, e, ring, cr) = (*h, *hk, *dh, *dr, *win, *e, *ring, *cr);
                        if dh > DH && t > 1 {
                            continue;
                        }
                        for &pos0 in poss {
                            if !ring && pos0 + t > cr {
                                continue;
                            }
                            let inputs = vec![
                                rand_t(rng, dt, &[t, h, dh], 1.5),
                                rand_t(rng, dt, &[t, hk, dh], 1.5),
                                rand_t(rng, dt, &[t, hk, dh], 1.0),
                                rand_t(rng, dt, &[hk, cr, dh], 1.5),
                                rand_t(rng, dt, &[hk, cr, dh], 1.0),
                                rand_t(rng, dt, &[t, h, e], 1.0),
                                rand_t(rng, dt, &[t, nq(h, dh, dr)], 2.0),
                            ];
                            // the caches are in place: the case's input copies are their initial contents
                            let outputs = vec![out(dt, &[t, h * dh]), inout(inputs[3].clone()), inout(inputs[4].clone())];
                            let mut a = OpAttrs { pos0, n_heads: h, n_kv_heads: hk, head_dim: dh, window: win, scale: (dh as f32).powf(-0.5), ..a0.clone() };
                            a.ink.kv_ring = ring;
                            a.ink.d_rel = dr;
                            push(format!("T={t} H={h} Hk={hk} Dh={dh} W={win} E={e} ring={ring} Cr={cr} pos0={pos0} {dt}"), inputs, outputs, a);
                        }
                    }
                }
            }
            if !quick {
                // the real model: sliding layers (window 1024, ring of 1040 rows: positions before the window fills, straddling
                // 1024, wrapping the ring) and full layers (linear cache, positions straddling the 1024 rel extent, 8192), rows
                // 1 / 4 / 16 and one 2048-token prefill chunk of each
                // head_dim 128 at small head counts and long chunks (the prefill kernels): window 64 over a ring of 80 rows,
                // chunks that wrap it, a linear cache, rel extent 48
                for (win, ring, cr, t, pos0) in [(64usize, true, 80usize, 40usize, 0usize), (64, true, 80, 100, 70), (64, true, 80, 33, 150), (0, false, 400, 100, 37), (0, false, 400, 200, 0)] {
                    let (h, hk, dh, dr, e) = (6usize, 2usize, 128usize, 16usize, 48usize);
                    let inputs = vec![
                        rand_t(rng, W, &[t, h, dh], 1.5),
                        rand_t(rng, W, &[t, hk, dh], 1.5),
                        rand_t(rng, W, &[t, hk, dh], 1.0),
                        rand_t(rng, W, &[hk, cr, dh], 1.5),
                        rand_t(rng, W, &[hk, cr, dh], 1.0),
                        rand_t(rng, W, &[t, h, e], 1.0),
                        rand_t(rng, W, &[t, nq(h, dh, dr)], 2.0),
                    ];
                    let outputs = vec![out(W, &[t, h * dh]), inout(inputs[3].clone()), inout(inputs[4].clone())];
                    let mut a = OpAttrs { pos0, n_heads: h, n_kv_heads: hk, head_dim: dh, window: win, scale: (dh as f32).powf(-0.5), ..a0.clone() };
                    a.ink.kv_ring = ring;
                    a.ink.d_rel = dr;
                    push(format!("mid T={t} H={h} Hk={hk} Dh={dh} W={win} E={e} ring={ring} Cr={cr} pos0={pos0} bf16"), inputs, outputs, a);
                }
                let (h, hk, dh, dr, e) = (24usize, 8usize, 128usize, 16usize, 1024usize);
                let lin = 8192 + 2048;
                let shapes: [(usize, bool, usize, Vec<(usize, Vec<usize>)>); 2] = [
                    (1024, true, 1040, vec![(1, vec![0, 1023, 2100]), (4, vec![1021, 1038, 5000]), (16, vec![1016, 1030, 3000]), (2048, vec![2048])]),
                    (0, false, lin, vec![(1, vec![1023, 8192]), (4, vec![1022]), (16, vec![1016, 8180]), (2048, vec![1024])]),
                ];
                for (win, ring, cr, rows) in shapes {
                    for (t, poss) in rows {
                        for pos0 in poss {
                            let inputs = vec![
                                rand_t(rng, W, &[t, h, dh], 1.5),
                                rand_t(rng, W, &[t, hk, dh], 1.5),
                                rand_t(rng, W, &[t, hk, dh], 1.0),
                                rand_t(rng, W, &[hk, cr, dh], 1.5),
                                rand_t(rng, W, &[hk, cr, dh], 1.0),
                                rand_t(rng, W, &[t, h, e], 1.0),
                                rand_t(rng, W, &[t, nq(h, dh, dr)], 2.0),
                            ];
                            let outputs = vec![out(W, &[t, h * dh]), inout(inputs[3].clone()), inout(inputs[4].clone())];
                            let mut a = OpAttrs { pos0, n_heads: h, n_kv_heads: hk, head_dim: dh, window: win, scale: (dh as f32).powf(-0.5), ..a0.clone() };
                            a.ink.kv_ring = ring;
                            a.ink.d_rel = dr;
                            push(format!("real T={t} W={win} ring={ring} Cr={cr} pos0={pos0} bf16"), inputs, outputs, a);
                        }
                    }
                }
            }
        }
        Op::InkAttnDecode => {
            // (H, Hk, Dh, dr, E, K, window, ring, cache rows, dyn+tau, positions, qk clip): small shapes and the real model
            type Shape = (usize, usize, usize, usize, usize, usize, usize, bool, usize, bool, Vec<usize>, f32);
            let mut shapes: Vec<Shape> = vec![
                (H, HK, DH, DR, 8, K, 8, true, 8 + 4, false, vec![0, 5, 29], 3.0),
                (H, HK, DH, DR, 12, K, 0, false, 96, true, vec![0, 5, 29], 3.0),
                (H, HK, DH, DR, 12, K, 8, false, 96, false, vec![3, 29], 3.0),
            ];
            if !quick {
                shapes.push((24, 8, 128, 16, 1024, 4, 1024, true, 1040, false, vec![0, 1023, 1030, 2100], 16.0 * 2f32.sqrt()));
                shapes.push((24, 8, 128, 16, 1024, 4, 0, false, 8192 + 64, true, vec![1022, 8192], 16.0 * 2f32.sqrt()));
            }
            let drows: &[usize] = if quick { &[1, 7] } else { &[1, 3, 4, 7, 16] };
            for &t in drows {
                for (h, hk, dh, dr, e, kk, win, ring, cr, dynamic, poss, qkc) in &shapes {
                    let (h, hk, dh, dr, e, kk, win, ring, cr, dynamic, qkc) = (*h, *hk, *dh, *dr, *e, *kk, *win, *ring, *cr, *dynamic, *qkc);
                    let ckv = hk * dh;
                    let rr = if dh > DH { kk - 1 + 16 } else { kk - 1 + 4 };
                    for &pos0 in poss {
                        if !ring && pos0 + t > cr {
                            continue;
                        }
                        let pw = if dynamic { ckv + hk * kk } else { ckv };
                        let kc = rand_t(rng, W, &[hk, cr, dh], 1.5);
                        let vc = rand_t(rng, W, &[hk, cr, dh], 1.0);
                        let inputs = vec![
                            rand_t(rng, W, &[t, nq(h, dh, dr)], 1.0),
                            rand_t(rng, W, &[t, pw], 1.0),
                            rand_t(rng, W, &[t, pw], 1.0),
                            rand_t(rng, W, &[kk, ckv], 0.5),
                            rand_t(rng, W, &[kk, ckv], 0.5),
                            pos_t(rng, W, &[h, dh], 0.2, 3.0),
                            pos_t(rng, W, &[h, dr], 0.2, 1.5),
                            rand_t(rng, W, &[dr, e], 0.25),
                            kc.clone(),
                            vc.clone(),
                        ];
                        let outputs = vec![
                            out(W, &[t, h * dh]),
                            inout(kc),
                            inout(vc),
                            inout(rand_t(rng, W, &[rr, ckv], 1.0)),
                            inout(rand_t(rng, W, &[rr, ckv], 1.0)),
                        ];
                        let mut a = OpAttrs { pos0, n_heads: h, n_kv_heads: hk, head_dim: dh, window: win, scale: (dh as f32).powf(-0.5), ..a0.clone() };
                        a.ink.qk_clip = qkc;
                        a.ink.rel_clip = 2.0;
                        a.ink.dyn_conv = dynamic;
                        a.ink.d_rel = dr;
                        a.ink.kv_ring = ring;
                        if dynamic {
                            (a.ink.tau_floor, a.ink.tau_alpha) = (if dh > DH { 1024 } else { 4 }, 0.1);
                        }
                        push(format!("T={t} H={h} Hk={hk} Dh={dh} E={e} W={win} ring={ring} Cr={cr} Rr={rr} pos0={pos0} dyn={dynamic} bf16"), inputs, outputs, a);
                    }
                }
            }
        }
        Op::MoeExpertsMxfp8 => {
            let mut shapes: Vec<(usize, usize, usize, usize, usize)> = vec![(1, 16, 4, 64, 32), (7, 16, 4, 64, 64), (16, 16, 4, 128, 32), (40, 16, 4, 64, 32)];
            if !quick {
                shapes.push((1, 8, 8, 2304, 1152));
            }
            for (t, e, k, hd, i) in shapes {
                for &dt in dts {
                    for epi in ["silu", "gelu", "situ"] {
                        // e4m3 bytes with exponents 4..=10 (|w| in 2^-3 .. 2^3 times 1.x), random sign and mantissa; never NaN
                        let fp8 = |rng: &mut SplitMix64, shape: &[usize]| {
                            let n: usize = shape.iter().product();
                            let data = (0..n)
                                .map(|_| {
                                    let r = rng.next_u64();
                                    ((r & 0x80) as u8) | ((4 + (r >> 8) % 7) as u8) << 3 | ((r >> 16) & 7) as u8
                                })
                                .collect();
                            HostTensor::new(DType::Fp8E4m3, shape.to_vec(), data).expect("sized")
                        };
                        let sc = |rng: &mut SplitMix64, shape: &[usize]| {
                            let n: usize = shape.iter().product();
                            HostTensor::new(DType::U8, shape.to_vec(), (0..n).map(|_| 120 + (rng.next_u64() % 4) as u8).collect()).expect("sized")
                        };
                        let (ids, ws) = crate::quant::cases::random_routing(rng, t, e, k);
                        let ws = if epi == "situ" { HostTensor::from_f32(&[t, k], &ws.to_f32().iter().map(|v| v * 8.0).collect::<Vec<_>>()) } else { ws };
                        let inputs = vec![
                            rand_t(rng, dt, &[t, hd], 1.0),
                            ids,
                            ws,
                            fp8(rng, &[e, 2 * i, hd]),
                            sc(rng, &[e, 2 * i, hd / 32]),
                            fp8(rng, &[e, hd, i]),
                            sc(rng, &[e, hd, i / 32]),
                        ];
                        let mut a = OpAttrs { top_k: k, gelu: epi == "gelu", ..a0.clone() };
                        if epi == "situ" {
                            (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
                        }
                        push(format!("T={t} E={e} K={k} H={hd} I={i} {epi} {dt}"), inputs, vec![out(dt, &[t, hd])], a);
                    }
                }
            }
        }
        Op::InkInProj => {
            let mut shapes: Vec<(usize, [usize; 3])> = vec![(D, [nq(H, DH, DR), HK * DH, HK * DH]), (D, [nq(H, DH, DR), HK * DH + HK * K, HK * DH + HK * K])];
            if !quick {
                shapes.push((2304, [6552, 1056, 1056]));
            }
            for &t in rows {
                for &dt in dts {
                    for &(d, ns) in &shapes {
                        if d > D && t > 1 {
                            continue;
                        }
                        for s in [1usize, 3] {
                            let inputs = vec![rand_t(rng, dt, &[s, t, d], 1.0), rand_t(rng, W, &[ns.iter().sum(), d], (d as f32).powf(-0.5))];
                            push(format!("T={t} S={s} D={d} N={ns:?} {dt}"), inputs, ns.iter().map(|&n| out(dt, &[t, n])).collect(), a0.clone());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    cases
}
