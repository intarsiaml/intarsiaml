//! Lane mmoedec: `INTARSIA_METAL_MOE_BITDEC=1` gives 1..=8-token `moe_experts_bf16` (one token) / `moe_experts_bf16_rows` (2..=8
//! tokens) calls the arithmetic-decode kernels `moe_dec_gu_a` / `moe_dec_dn_a` (kernels/metal/moe.metal, lane mmoe16's). Claim:
//! bitwise the bytes of today's path, i.e. of the T one-token `moe_experts_bf16` calls with the knob off.
//!
//! * `bitdec_moe_calls_are_bitwise_the_one_token_call`: T 1..=8, seven routings (random, identical, overlap, invalid ids, an empty
//!   token, duplicates, a 72-expert pool), bf16 and f32 out, aligned and 8-byte-offset weights, activations of normal, tiny, large
//!   and edge (bf16 subnormal, near 2^-126, near 2^100, signed zero) magnitude; knob-on rows call and knob-on one-token calls
//!   against the knob-off one-token calls and the knob-off rows call.
//! * `bitdec_negative_control`: the compare is live (one changed activation of the last token gives different bytes).
//! * `timing_moe_bitdec` (`--ignored`): us per call at 35B shapes, knob off against on, T = 1, 2, 3, 8.
//!
//! Uses the GPU (about 0.5 GB): on endor take a fleetq slot first.

#![cfg(target_os = "macos")]

mod common;
use common::*;
use intarsia_backend_metal::{with_moe_bitdec, MetalBackend};
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::quant::cases::{random_experts, random_routing};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, KernelImpl, Op, OpAttrs, OpCall, ShapeInfo};

const E: usize = 256;
const K: usize = 8;
const H: usize = 2048;
const I: usize = 512;

fn put_bytes(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
    let b = be.alloc(off + data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() + 16], 0).unwrap();
    be.upload(&b, off, data, 0).unwrap();
    b
}

fn put_host(be: &MetalBackend, h: &HostTensor, off: usize) -> TensorRef {
    let b = put_bytes(be, off, &h.data);
    TensorRef { buf: b, offset: off, dtype: h.dtype, shape: h.shape.clone() }
}

fn guarded(be: &MetalBackend, bytes: usize) -> Buffer {
    let b = be.alloc(bytes + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; bytes + 16], 0).unwrap();
    b
}

fn dl(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; n];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Each token picks K distinct experts out of a pool of `pool` (the real 16-row verify pass routes its 128 pairs to about 61
/// distinct experts, docs/results/2026-11-18-mrows16k.md; a pool of 72 gives about that).
fn pooled(rng: &mut SplitMix64, t: usize, pool: usize) -> Vec<i32> {
    let mut all: Vec<i32> = (0..E as i32).collect();
    for i in 0..pool {
        let j = i + (rng.next_u64() as usize) % (E - i);
        all.swap(i, j);
    }
    let mut ids = Vec::with_capacity(t * K);
    for _ in 0..t {
        let mut p = all[..pool].to_vec();
        for k in 0..K {
            let j = k + (rng.next_u64() as usize) % (pool - k);
            p.swap(k, j);
            ids.push(p[k]);
        }
    }
    ids
}

/// (name, ids, weights) for `t` tokens: rows_exact.rs's routings plus the pooled one.
fn routings(rng: &mut SplitMix64, t: usize) -> Vec<(&'static str, Vec<i32>, Vec<f32>)> {
    let (ids, ws) = random_routing(rng, t, E, K);
    let (rids, rws) = (ids.to_i32().unwrap(), ws.to_f32());
    let mut out = vec![("random", rids.clone(), rws.clone())];
    out.push(("identical", (0..t).flat_map(|_| rids[..K].to_vec()).collect(), rws.clone()));
    let mut ov = Vec::new();
    for i in 0..t {
        for j in 0..K {
            ov.push(if j < 4 { rids[j] } else { rids[(i * K + j) % rids.len()] });
        }
    }
    for tok in ov.chunks_mut(K) {
        for j in 0..K {
            while tok[..j].contains(&tok[j]) {
                tok[j] = (tok[j] + 1) % E as i32;
            }
        }
    }
    out.push(("overlap", ov, rws.clone()));
    let mut bad = rids.clone();
    for (i, v) in bad.iter_mut().enumerate() {
        match i % 11 {
            3 => *v = -1,
            7 => *v = E as i32,
            9 => *v = i32::MIN,
            _ => {}
        }
    }
    out.push(("invalid", bad, rws.clone()));
    let mut none = rids.clone();
    for v in none[..K].iter_mut() {
        *v = -1;
    }
    out.push(("empty-token", none, rws.clone()));
    out.push(("duplicates", (0..t * K).map(|i| if i / K % 2 == 0 { 5 } else { rids[i] }).collect(), rws.clone()));
    out.push(("pool72", pooled(rng, t, 72), rws));
    out
}

struct Rig {
    be: MetalBackend,
    solo: KernelImpl,
    rows: KernelImpl,
}

impl Rig {
    fn new() -> Self {
        let be = backend();
        let ks = be.kernels();
        let solo = ks.iter().find(|k| k.name == "moe_experts_bf16").unwrap().clone();
        let rows = ks.iter().find(|k| k.name == "moe_experts_bf16_rows").unwrap().clone();
        Rig { be, solo, rows }
    }

    /// A MoeExperts call of `n` tokens from row `r0`, writing `out` at row `r0`, with its scratch.
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        w: &[TensorRef; 5],
        x: &TensorRef,
        ids: &TensorRef,
        ws: &TensorRef,
        out: &Buffer,
        odt: DType,
        r0: usize,
        n: usize,
        kern: &KernelImpl,
    ) -> OpCall {
        let esz = odt.size_bytes();
        let mut c = OpCall::new(
            Op::MoeExperts,
            vec![x.rows(r0, n), ids.rows(r0, n), ws.rows(r0, n), w[0].clone(), w[1].clone(), w[2].clone(), w[3].clone(), w[4].clone()],
            vec![TensorRef { buf: out.clone(), offset: r0 * H * esz, dtype: odt, shape: vec![n, H] }],
            OpAttrs { top_k: K, ..OpAttrs::default() },
        );
        c.check().unwrap();
        let need = kern.scratch_bytes(&ShapeInfo::from_call(&c));
        c.scratch = Some(TensorRef::new(self.be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
        c
    }
}

fn weights(be: &MetalBackend, rng: &mut SplitMix64, w_off: usize) -> [TensorRef; 5] {
    let ex = random_experts(rng, E, H, I);
    [
        put_host(be, &ex.gate_up_w, w_off),
        put_host(be, &ex.gate_up_s, 0),
        put_host(be, &ex.down_w, w_off),
        put_host(be, &ex.down_s, 0),
        put_host(be, &ex.gscales, 0),
    ]
}

#[test]
fn bitdec_moe_calls_are_bitwise_the_one_token_call() {
    let rig = Rig::new();
    let be = &rig.be;
    let mut rng = SplitMix64::new(0x16_0e_77);
    let (mut cases, mut rows_cmp) = (0usize, 0usize);
    for w_off in [0usize, 8] {
        let w = weights(be, &mut rng, w_off);
        for tr in 1..=8usize {
            // activations of mixed magnitude: normal, and every third token scaled up / down (wide exponent range)
            let mut xv = rng.normal_vec(tr * H, 1.0);
            for (i, v) in xv.iter_mut().enumerate() {
                match (i / H) % 3 {
                    1 => *v *= 1e-6,
                    2 => *v *= 3e4,
                    _ => {}
                }
            }
            let mut xh = HostTensor::from_f32_as(DType::Bf16, &[tr, H], &xv);
            // token 3: raw bf16 bit patterns at the edges of the exactness argument: subnormals (0x0001..0x007F and the
            // negative ones), values just above 2^-126, values near +-2^100, zeros (built from bits, not from f32)
            for j in 0..H {
                let bits: u16 = match j % 6 {
                    0 => 1 + (j as u16 / 6) % 0x7F,
                    1 => 0x8001 + (j as u16 / 6) % 0x7F,
                    2 => 0x0080 + (j as u16 / 6) % 0x80,
                    3 => 0x7180 + (j as u16 / 6) % 0x80,
                    4 => 0xF180 + (j as u16 / 6) % 0x80,
                    _ => if j % 12 == 5 { 0 } else { 0x8000 },
                };
                let o = (tr.min(4) - 1) * H * 2 + j * 2;
                xh.data[o..o + 2].copy_from_slice(&bits.to_le_bytes());
            }
            let x = put_host(be, &xh, 0);
            for (name, ids, ws) in routings(&mut rng, tr) {
                let idt = put_host(be, &HostTensor::from_i32(&[tr, K], &ids), 0);
                let wst = put_host(be, &HostTensor::from_f32(&[tr, K], &ws), 0);
                for odt in [DType::Bf16, DType::F32] {
                    let tot = tr * H * odt.size_bytes() + 16;
                    let (ys, yoff, yon, ysn) =
                        (guarded(be, tot - 16), guarded(be, tot - 16), guarded(be, tot - 16), guarded(be, tot - 16));
                    for r in 0..tr {
                        let c = rig.call(&w, &x, &idt, &wst, &ys, odt, r, 1, &rig.solo);
                        with_moe_bitdec(false, || be.launch("moe_experts_bf16", &c, 0).unwrap());
                        let c = rig.call(&w, &x, &idt, &wst, &ysn, odt, r, 1, &rig.solo);
                        with_moe_bitdec(true, || be.launch("moe_experts_bf16", &c, 0).unwrap());
                    }
                    let (a, d) = (dl(be, &ys, tot), dl(be, &ysn, tot));
                    let what = format!("T={tr} routing={name} out={odt} w_off={w_off}");
                    assert_eq!(differing(&a, &d), 0, "one-token knob on vs off, {what}");
                    if tr >= 2 {
                        let c_off = rig.call(&w, &x, &idt, &wst, &yoff, odt, 0, tr, &rig.rows);
                        let c_on = rig.call(&w, &x, &idt, &wst, &yon, odt, 0, tr, &rig.rows);
                        with_moe_bitdec(false, || be.launch("moe_experts_bf16_rows", &c_off, 0).unwrap());
                        with_moe_bitdec(true, || be.launch("moe_experts_bf16_rows", &c_on, 0).unwrap());
                        let (b, c) = (dl(be, &yoff, tot), dl(be, &yon, tot));
                        assert_eq!(differing(&a, &b), 0, "knob off rows vs one-token calls, {what}");
                        assert_eq!(differing(&a, &c), 0, "knob on rows vs one-token calls, {what}");
                    }
                    // the outputs are not trivially empty (an all-invalid routing would leave zeros everywhere)
                    assert!((name == "empty-token" && tr == 1) || d[..tot - 16].iter().any(|&v| v != 0 && v != 0xEE), "{what}: output looks unwritten");
                    cases += 1;
                    rows_cmp += tr;
                }
            }
        }
    }
    eprintln!("MMOEDEC bitwise: {cases} cases, {rows_cmp} token rows equal (knob on == knob off == one-token calls)");
}

/// The compare can see a difference: the knob-on call against a reference whose last token has one activation changed.
#[test]
fn bitdec_negative_control() {
    let rig = Rig::new();
    let be = &rig.be;
    let mut rng = SplitMix64::new(0x0e_9a_77);
    let w = weights(be, &mut rng, 0);
    for tr in [1usize, 2, 3, 8] {
        let xv = rng.normal_vec(tr * H, 1.0);
        let mut xp = xv.clone();
        xp[(tr - 1) * H + 37] += 0.5;
        let x = put_host(be, &HostTensor::from_f32_as(DType::Bf16, &[tr, H], &xv), 0);
        let xq = put_host(be, &HostTensor::from_f32_as(DType::Bf16, &[tr, H], &xp), 0);
        let ids = pooled(&mut rng, tr, 72);
        let ws: Vec<f32> = (0..tr * K).map(|i| 0.05 + (i % 7) as f32 * 0.02).collect();
        let idt = put_host(be, &HostTensor::from_i32(&[tr, K], &ids), 0);
        let wst = put_host(be, &HostTensor::from_f32(&[tr, K], &ws), 0);
        let tot = tr * H * 2 + 16;
        let (yr, yon) = (guarded(be, tot - 16), guarded(be, tot - 16));
        let (kern, name) = if tr == 1 { (&rig.solo, "moe_experts_bf16") } else { (&rig.rows, "moe_experts_bf16_rows") };
        let c_ref = rig.call(&w, &xq, &idt, &wst, &yr, DType::Bf16, 0, tr, kern);
        let c_on = rig.call(&w, &x, &idt, &wst, &yon, DType::Bf16, 0, tr, kern);
        with_moe_bitdec(false, || be.launch(name, &c_ref, 0).unwrap());
        with_moe_bitdec(true, || be.launch(name, &c_on, 0).unwrap());
        let (a, b) = (dl(be, &yr, tot), dl(be, &yon, tot));
        let d = differing(&a, &b);
        let first = a.iter().zip(&b).position(|(p, q)| p != q).unwrap_or(usize::MAX);
        eprintln!("MMOEDEC negctl T={tr}: {d} of {tot} bytes differ, first at byte {first} (last token starts at {})", (tr - 1) * H * 2);
        assert!(d > 0, "negative control T={tr}: the changed activation was not seen");
        assert!(first >= (tr - 1) * H * 2, "negative control T={tr}: a token other than the last changed");
    }
}

fn time_us(be: &MetalBackend, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        be.synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        be.synchronize(0).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
    }
    best
}

/// `cargo test --release -p intarsia-backend-metal --test moe_bitdec -- --ignored --nocapture timing_moe_bitdec`: us per
/// `moe_experts_bf16` (T = 1) / `moe_experts_bf16_rows` call at the 35B's shapes (E 256, top 8, H 2048, I 512), knob off and on, pooled routing (72-expert
/// pool, about the real pass's distinct-expert count) and random routing, 24 routings cycled over 4 weight sets (1.7 GB, so the
/// experts a call reads are mostly not cached), 6 alternating rounds, medians. x 40 = one pass's routed-expert time. Quiet fleetq slot.
#[test]
#[ignore]
fn timing_moe_bitdec() {
    let rig = Rig::new();
    let be = &rig.be;
    let mut rng = SplitMix64::new(0x71_16);
    let sets: Vec<[TensorRef; 5]> = (0..4).map(|_| weights(be, &mut rng, 0)).collect();
    const VARIANTS: usize = 24;
    for tr in [1usize, 2, 3, 8] {
        let x = put_host(be, &HostTensor::from_f32_as(DType::Bf16, &[tr, H], &rng.normal_vec(tr * H, 1.0)), 0);
        let y = guarded(be, tr * H * 2);
        let (kern, name) = if tr == 1 { (&rig.solo, "moe_experts_bf16") } else { (&rig.rows, "moe_experts_bf16_rows") };
        for kind in ["pool72", "random"] {
            let mut calls = Vec::new();
            let mut distinct = 0usize;
            for v in 0..VARIANTS {
                let ids = if kind == "pool72" { pooled(&mut rng, tr, 72) } else { random_routing(&mut rng, tr, E, K).0.to_i32().unwrap() };
                let mut d = ids.clone();
                d.sort();
                d.dedup();
                distinct += d.len();
                let ws: Vec<f32> = (0..tr * K).map(|i| 0.05 + (i % 7) as f32 * 0.02).collect();
                let idt = put_host(be, &HostTensor::from_i32(&[tr, K], &ids), 0);
                let wst = put_host(be, &HostTensor::from_f32(&[tr, K], &ws), 0);
                calls.push(rig.call(&sets[v % sets.len()], &x, &idt, &wst, &y, DType::Bf16, 0, tr, kern));
            }
            // arms off / on, 6 rounds, the order alternating per round (ABBA); median of the 6 per arm
            let mut res: [Vec<f64>; 2] = Default::default();
            for round in 0..6 {
                for a in 0..2 {
                    let arm: usize = (a + round) % 2;
                    let mut i = 0usize;
                    let us = with_moe_bitdec(arm == 1, || {
                        time_us(be, 48, || {
                            i += 1;
                            be.launch(name, &calls[i % VARIANTS], 0).unwrap();
                        })
                    });
                    res[arm].push(us);
                }
            }
            let med = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
            };
            let (off, on) = (med(&mut res[0]), med(&mut res[1]));
            eprintln!(
                "TIMING moe_bitdec T={tr} routing={kind} distinct={:.1}: off {off:.1} us, on {on:.1} us ({:.3}x), x40 layers off {:.2} ms on {:.2} ms",
                distinct as f64 / VARIANTS as f64,
                off / on,
                off * 40.0 / 1000.0,
                on * 40.0 / 1000.0
            );
        }
    }
}
