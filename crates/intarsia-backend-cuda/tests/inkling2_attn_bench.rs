//! Timing of the Inkling-2-Mini attention kernels on CUDA at the real shapes (24 / 8 heads, head_dim 128, d_rel 16, extent
//! 1024, K 4; sliding layers: window 1024 on a ring of 1040 rows; full layers: a linear cache). Only with
//! `INTARSIA_INK_BENCH=1` (else it returns at once); prints microseconds per visit, eager and replayed from a captured CUDA
//! graph, for the unfused pair (prologue + attention), each half, and the fused `InkAttnDecode`, at 1 / 4 / 16 rows for
//! contexts 512 / 4k / 32k, plus one 2048-row prefill chunk (`INTARSIA_INK_BENCH_PREFILL=1`). 16 visits rotate over their own
//! operands so the K / V never sits in L2 between visits.
#![cfg(ik_cuda)]

mod common;

use intarsia_core::inkling2::ops::nq;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, Buffer, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, ShapeInfo, TensorRef};
use std::time::Instant;

const H: usize = 24;
const HK: usize = 8;
const DH: usize = 128;
const DR: usize = 16;
const E: usize = 1024;
const K: usize = 4;
const RR: usize = K - 1 + 16;
const NV: usize = 16;

fn tr(b: &Buffer, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: 0, dtype: DType::Bf16, shape: shape.to_vec() }
}

fn rand_buf(be: &dyn Backend, n: usize, seed: u64, scale: f32) -> Buffer {
    let mut rng = SplitMix64::new(seed);
    let data = HostTensor::from_f32_as(DType::Bf16, &[n], &rng.normal_vec(n, scale)).data;
    let b = be.alloc(data.len().max(16), MemKind::Device).unwrap();
    be.upload(&b, 0, &data, 0).unwrap();
    b
}

fn scratch(be: &dyn Backend, name: &str, call: &OpCall) -> Option<TensorRef> {
    let need = be.kernels().iter().find(|k| k.name == name).unwrap().scratch_bytes(&ShapeInfo::from_call(call));
    (need > 0).then(|| TensorRef { buf: be.alloc(need, MemKind::Device).unwrap(), offset: 0, dtype: DType::U8, shape: vec![need] })
}

/// One visit's calls: (prologue, attention, fused).
struct Visit {
    pro: OpCall,
    att: OpCall,
    fused: Option<OpCall>,
}

fn visit(be: &dyn Backend, window: usize, ring: bool, cr: usize, t: usize, pos0: usize, seed: u64) -> Visit {
    let ckv = HK * DH;
    let dynamic = !ring;
    let pw = if dynamic { ckv + HK * K } else { ckv };
    let nqw = nq(H, DH, DR);
    let mut a = OpAttrs { pos0, n_heads: H, n_kv_heads: HK, head_dim: DH, window, scale: (DH as f32).powf(-0.5), eps: 1e-6, ..OpAttrs::default() };
    a.ink.qk_clip = 6.0;
    a.ink.rel_clip = 1.5;
    a.ink.d_rel = DR;
    a.ink.kv_ring = ring;
    a.ink.dyn_conv = dynamic;
    if dynamic {
        (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
    }
    let pq = rand_buf(be, t * nqw, seed, 1.0);
    let pk = rand_buf(be, t * pw, seed + 1, 1.0);
    let pv = rand_buf(be, t * pw, seed + 2, 1.0);
    let w = [rand_buf(be, K * ckv, seed + 3, 0.5), rand_buf(be, K * ckv, seed + 4, 0.5), rand_buf(be, H * DH, seed + 5, 1.0), rand_buf(be, H * DR, seed + 6, 1.0), rand_buf(be, DR * E, seed + 7, 0.25)];
    let (kr, vr) = (rand_buf(be, RR * ckv, seed + 8, 1.0), rand_buf(be, RR * ckv, seed + 9, 1.0));
    let (kc, vc) = (rand_buf(be, HK * cr * DH, seed + 10, 1.5), rand_buf(be, HK * cr * DH, seed + 11, 1.0));
    let z = |n: usize| {
        let b = be.alloc(n * 2, MemKind::Device).unwrap();
        be.zero(&b, 0).unwrap();
        b
    };
    let (q, rel, k, v, o) = (z(t * H * DH), z(t * H * E), z(t * ckv), z(t * ckv), z(t * H * DH));
    let pro_in = vec![
        tr(&pq, &[t, nqw]),
        tr(&pk, &[t, pw]),
        tr(&pv, &[t, pw]),
        tr(&w[0], &[K, ckv]),
        tr(&w[1], &[K, ckv]),
        tr(&w[2], &[H, DH]),
        tr(&w[3], &[H, DR]),
        tr(&w[4], &[DR, E]),
    ];
    let pro = OpCall::new(
        Op::InkKvPrologue,
        pro_in.clone(),
        vec![tr(&q, &[t, H, DH]), tr(&rel, &[t, H, E]), tr(&k, &[t, HK, DH]), tr(&v, &[t, HK, DH]), tr(&kr, &[RR, ckv]), tr(&vr, &[RR, ckv])],
        a.clone(),
    );
    let mut att = OpCall::new(
        Op::RelSinkAttn,
        vec![tr(&q, &[t, H, DH]), tr(&k, &[t, HK, DH]), tr(&v, &[t, HK, DH]), tr(&kc, &[HK, cr, DH]), tr(&vc, &[HK, cr, DH]), tr(&rel, &[t, H, E]), tr(&pq, &[t, nqw])],
        vec![tr(&o, &[t, H * DH]), tr(&kc, &[HK, cr, DH]), tr(&vc, &[HK, cr, DH])],
        a.clone(),
    );
    att.scratch = scratch(be, "rel_sink_attn_bf16", &att);
    let fused = (t <= 16).then(|| {
        let mut inputs = pro_in;
        inputs.push(tr(&kc, &[HK, cr, DH]));
        inputs.push(tr(&vc, &[HK, cr, DH]));
        let mut f = OpCall::new(
            Op::InkAttnDecode,
            inputs,
            vec![tr(&o, &[t, H * DH]), tr(&kc, &[HK, cr, DH]), tr(&vc, &[HK, cr, DH]), tr(&kr, &[RR, ckv]), tr(&vr, &[RR, ckv])],
            a,
        );
        f.scratch = scratch(be, "ink_attn_decode_bf16", &f);
        f
    });
    Visit { pro, att, fused }
}

/// Microseconds per visit of `body` over all visits: (eager, graph replay).
fn time(be: &dyn Backend, visits: &[Visit], reps: usize, body: &dyn Fn(&Visit)) -> (f64, f64) {
    for v in visits {
        body(v);
    }
    be.synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..reps {
        for v in visits {
            body(v);
        }
    }
    be.synchronize(0).unwrap();
    let eager = t0.elapsed().as_secs_f64() * 1e6 / (reps * visits.len()) as f64;
    let g = be
        .graph_capture(0, &mut || {
            for v in visits {
                body(v);
            }
            Ok(())
        })
        .unwrap();
    be.graph_replay(&g, 0).unwrap();
    be.synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..reps {
        be.graph_replay(&g, 0).unwrap();
    }
    be.synchronize(0).unwrap();
    (eager, t0.elapsed().as_secs_f64() * 1e6 / (reps * visits.len()) as f64)
}

#[test]
fn ink_attn_timing() {
    if std::env::var("INTARSIA_INK_BENCH").map_or(true, |v| v != "1") {
        eprintln!("inkling2_attn_bench: set INTARSIA_INK_BENCH=1 to run");
        return;
    }
    let Some(be) = intarsia_backend_cuda::create(0).ok() else {
        common::skip("inkling2_attn_bench", "no usable CUDA device");
        return;
    };
    let be = be.as_ref();
    let launch = |name: &str, c: &OpCall| be.launch(name, c, 0).unwrap();
    let prefill = std::env::var("INTARSIA_INK_BENCH_PREFILL").is_ok_and(|v| v == "1");
    println!("layer ctx rows | prologue | attn | pair | fused   (us per visit, eager / graph) | KV MB per visit");
    let cases: Vec<(&str, usize, Vec<usize>)> = if prefill {
        vec![("swa", 1024, vec![0, 2048, 30720]), ("full", 0, vec![0, 2048, 30720])]
    } else {
        vec![("swa", 1024, vec![512, 4096]), ("full", 0, vec![512, 4096, 32768])]
    };
    let rows: &[usize] = if prefill { &[2048] } else { &[1, 4, 16] };
    for (name, window, ctxs) in cases {
        let ring = window > 0;
        for ctx in ctxs {
            for &t in rows {
                let cr = if ring { 1040 } else { ctx + t + 64 };
                let nv = if t >= 2048 { 2 } else { NV };
                let visits: Vec<Visit> = (0..nv).map(|i| visit(be, window, ring, cr, t, ctx, 1000 + i as u64 * 31)).collect();
                let reps = if t >= 2048 || ctx >= 32768 { 3 } else { 20 };
                let pro = time(be, &visits, reps, &|v| launch("ink_kv_prologue_bf16", &v.pro));
                // the kernel the registry picks: the flash kernel for prefill chunks (INTARSIA_INK_BENCH_ROWS_EXACT=1: the rows-exact one)
                let an = if t > 16 && std::env::var("INTARSIA_INK_BENCH_ROWS_EXACT").map_or(true, |v| v != "1") { "rel_sink_attn_bf16_flash" } else { "rel_sink_attn_bf16" };
                let att = time(be, &visits, reps, &|v| launch(an, &v.att));
                let pair = time(be, &visits, reps, &|v| {
                    launch("ink_kv_prologue_bf16", &v.pro);
                    launch(an, &v.att);
                });
                let fused = if visits[0].fused.is_some() { Some(time(be, &visits, reps, &|v| launch("ink_attn_decode_bf16", v.fused.as_ref().unwrap()))) } else { None };
                let vis = if ring { (ctx + t).min(window) } else { ctx + t };
                let kv_mb = (vis * HK * DH * 2 * 2) as f64 / 1e6;
                let f = fused.map_or("-".to_string(), |(e, g)| format!("{e:.1} / {g:.1}"));
                println!(
                    "{name} {ctx} {t} | {:.1} / {:.1} | {:.1} / {:.1} | {:.1} / {:.1} | {f} | {kv_mb:.1} (floor {:.1} us at 273 GB/s)",
                    pro.0,
                    pro.1,
                    att.0,
                    att.1,
                    pair.0,
                    pair.1,
                    kv_mb / 273e3 * 1e6
                );
            }
        }
    }
}
