//! Inkling-2-Mini attention on HIP (`ink_kv_prologue_bf16`, `rel_sink_attn_bf16`; src/inkattn.rs, kernels/hip/ink_attn.hip;
//! lane ink-hip-attn):
//! * `admission_full_sweep`: the full (non-quick) admission sweep of ops 70 / 71 against the f64 cpu reference (`ink_ref`) at the
//!   registered tolerances: small shapes (head_dim 16, d_rel 8) and the real model (24 / 8 heads, head_dim 128, d_rel 16, extent
//!   1024; sliding layers with window 1024 over a ring of 1040 rows and full layers with a linear cache; rows 1 / 4 / 16 / 2048).
//! * `rows_exact_real_shapes`: an R-row prologue + attention pair equals, bit for bit, R one-row pairs at positions pos0 ..
//!   pos0 + R made in order (q / rel / k / v / o and the final conv rings and K / V caches), host and device positions; T up to
//!   256 (the attention runs a call as 64-row slices). Port of the CUDA test (crates/intarsia-backend-cuda/tests/inkling2_attn.rs)
//!   without its op-75 arms.
//! * `bench_prefill` (ignored): one 256-row chunk at 4k context, split-KV vs flash (cache write included).
//! * `bench_decode` (ignored; run with --ignored): one-row prologue + attention at real shapes against the KV-read roofline.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_hip)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{run_admission, AdmissionOpts, SweepConfig};
use intarsia_core::inkling2::ops::nq;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, Buffer, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, ShapeInfo, TensorRef};
use std::sync::Arc;

const H: usize = 24;
const HK: usize = 8;
const DH: usize = 128;
const DR: usize = 16;
const E: usize = 1024;
const K: usize = 4;
const RR: usize = K - 1 + 16;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("ink_attn", &format!("no usable HIP device ({e})"));
            None
        }
    }
}

fn tr(b: &Buffer, off: usize, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: DType::Bf16, shape: shape.to_vec() }
}

fn rand_bf16(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    let v: Vec<f32> = if lo == hi { rng.normal_vec(n, lo) } else { (0..n).map(|_| lo + (hi - lo) * rng.next_f32()).collect() };
    HostTensor::from_f32_as(DType::Bf16, &[n], &v).data
}

fn upload(be: &dyn Backend, v: &[u8]) -> Buffer {
    let b = be.alloc(v.len().max(16), MemKind::Device).unwrap();
    be.upload(&b, 0, v, 0).unwrap();
    b
}

fn zeros(be: &dyn Backend, n: usize) -> Buffer {
    upload(be, &vec![0u8; n * 2])
}

fn read(be: &dyn Backend, b: &Buffer, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.synchronize(0).unwrap();
    be.download(b, 0, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

struct Layer {
    window: usize,
    ring: bool,
    cr: usize,
    dynamic: bool,
}

/// Everything one layer visit touches; `state` = (k ring, v ring, k cache, v cache).
struct Run {
    outs: Vec<Vec<u16>>,
    state: Vec<Vec<u16>>,
}

struct Inputs {
    pq: Vec<u8>,
    pk: Vec<u8>,
    pv: Vec<u8>,
    w: Vec<Vec<u8>>, // k taps, v taps, q scale, r scale, rel proj
    state0: Vec<Vec<u8>>,
}

fn make(l: &Layer, t: usize, seed: u64) -> Inputs {
    let ckv = HK * DH;
    let pw = if l.dynamic { ckv + HK * K } else { ckv };
    Inputs {
        pq: rand_bf16(t * nq(H, DH, DR), seed, 1.0, 1.0),
        pk: rand_bf16(t * pw, seed + 1, 1.0, 1.0),
        pv: rand_bf16(t * pw, seed + 2, 1.0, 1.0),
        w: vec![
            rand_bf16(K * ckv, seed + 3, 0.5, 0.5),
            rand_bf16(K * ckv, seed + 4, 0.5, 0.5),
            rand_bf16(H * DH, seed + 5, 0.5, 6.0),
            rand_bf16(H * DR, seed + 6, 0.5, 3.0),
            rand_bf16(DR * E, seed + 7, 0.25, 0.25),
        ],
        state0: vec![
            rand_bf16(RR * ckv, seed + 8, 1.0, 1.0),
            rand_bf16(RR * ckv, seed + 9, 1.0, 1.0),
            rand_bf16(HK * l.cr * DH, seed + 10, 1.5, 1.5),
            rand_bf16(HK * l.cr * DH, seed + 11, 1.0, 1.0),
        ],
    }
}

fn attrs(l: &Layer, pos0: usize) -> OpAttrs {
    let mut a = OpAttrs { pos0, n_heads: H, n_kv_heads: HK, head_dim: DH, window: l.window, scale: (DH as f32).powf(-0.5), eps: 1e-6, ..OpAttrs::default() };
    a.ink.qk_clip = 6.0;
    a.ink.rel_clip = 1.5;
    a.ink.d_rel = DR;
    a.ink.kv_ring = l.ring;
    a.ink.dyn_conv = l.dynamic;
    if l.dynamic {
        (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
    }
    a
}

fn scratch_for(be: &dyn Backend, name: &str, call: &OpCall) -> TensorRef {
    let need = be.kernels().iter().find(|k| k.name == name).unwrap().scratch_bytes(&ShapeInfo::from_call(call)).max(16);
    let sb = be.alloc(need, MemKind::Device).unwrap();
    TensorRef { buf: sb, offset: 0, dtype: DType::U8, shape: vec![need] }
}

/// Run the visit as calls of `rows` rows each (rows divides T); `dev`: device positions; `fused`: InkAttnDecode.
#[allow(clippy::too_many_arguments)]
fn run(be: &dyn Backend, l: &Layer, inp: &Inputs, t: usize, pos0: usize, rows: usize, dev: bool, fused: bool) -> Run {
    run_with(be, l, inp, t, pos0, rows, dev, fused, "rel_sink_attn_bf16")
}

/// `run` with the pair's attention kernel `ak`; the fused calls take the prefill kernel above 16 rows.
#[allow(clippy::too_many_arguments)]
fn run_with(be: &dyn Backend, l: &Layer, inp: &Inputs, t: usize, pos0: usize, rows: usize, dev: bool, fused: bool, ak: &str) -> Run {
    let ckv = HK * DH;
    let pw = inp.pk.len() / 2 / t;
    let nqw = nq(H, DH, DR);
    let (pq, pk, pv) = (upload(be, &inp.pq), upload(be, &inp.pk), upload(be, &inp.pv));
    let w: Vec<Buffer> = inp.w.iter().map(|v| upload(be, v)).collect();
    let st: Vec<Buffer> = inp.state0.iter().map(|v| upload(be, v)).collect();
    let (q, rel, k, v, o) = (zeros(be, t * H * DH), zeros(be, t * H * E), zeros(be, t * ckv), zeros(be, t * ckv), zeros(be, t * H * DH));
    let posb = be.alloc(16, MemKind::Device).unwrap();
    for c in 0..t / rows {
        let r0 = c * rows;
        let p0 = pos0 + r0;
        let mut a = attrs(l, p0);
        let pos = if dev {
            be.synchronize(0).unwrap();
            be.upload(&posb, 0, &(p0 as i32).to_le_bytes(), 0).unwrap();
            a.pos0 = 0; // must not be read
            Some(TensorRef { buf: posb.clone(), offset: 0, dtype: DType::I32, shape: vec![1] })
        } else {
            None
        };
        let pro_in = vec![
            tr(&pq, r0 * nqw * 2, &[rows, nqw]),
            tr(&pk, r0 * pw * 2, &[rows, pw]),
            tr(&pv, r0 * pw * 2, &[rows, pw]),
            tr(&w[0], 0, &[K, ckv]),
            tr(&w[1], 0, &[K, ckv]),
            tr(&w[2], 0, &[H, DH]),
            tr(&w[3], 0, &[H, DR]),
            tr(&w[4], 0, &[DR, E]),
        ];
        if fused {
            let mut inputs = pro_in;
            inputs.push(tr(&st[2], 0, &[HK, l.cr, DH]));
            inputs.push(tr(&st[3], 0, &[HK, l.cr, DH]));
            let mut f = OpCall::new(
                Op::InkAttnDecode,
                inputs,
                vec![
                    tr(&o, r0 * H * DH * 2, &[rows, H * DH]),
                    tr(&st[2], 0, &[HK, l.cr, DH]),
                    tr(&st[3], 0, &[HK, l.cr, DH]),
                    tr(&st[0], 0, &[RR, ckv]),
                    tr(&st[1], 0, &[RR, ckv]),
                ],
                a,
            );
            f.pos = pos;
            let fk = if rows > 16 { "ink_attn_decode_prefill_bf16" } else { "ink_attn_decode_bf16" };
            f.scratch = Some(scratch_for(be, fk, &f));
            be.launch(fk, &f, 0).unwrap();
            be.synchronize(0).unwrap();
            continue;
        }
        let mut pro = OpCall::new(
            Op::InkKvPrologue,
            pro_in,
            vec![
                tr(&q, r0 * H * DH * 2, &[rows, H, DH]),
                tr(&rel, r0 * H * E * 2, &[rows, H, E]),
                tr(&k, r0 * ckv * 2, &[rows, HK, DH]),
                tr(&v, r0 * ckv * 2, &[rows, HK, DH]),
                tr(&st[0], 0, &[RR, ckv]),
                tr(&st[1], 0, &[RR, ckv]),
            ],
            a.clone(),
        );
        pro.pos = pos.clone();
        be.launch("ink_kv_prologue_bf16", &pro, 0).unwrap();
        let mut att = OpCall::new(
            Op::RelSinkAttn,
            vec![
                tr(&q, r0 * H * DH * 2, &[rows, H, DH]),
                tr(&k, r0 * ckv * 2, &[rows, HK, DH]),
                tr(&v, r0 * ckv * 2, &[rows, HK, DH]),
                tr(&st[2], 0, &[HK, l.cr, DH]),
                tr(&st[3], 0, &[HK, l.cr, DH]),
                tr(&rel, r0 * H * E * 2, &[rows, H, E]),
                tr(&pq, r0 * nqw * 2, &[rows, nqw]),
            ],
            vec![tr(&o, r0 * H * DH * 2, &[rows, H * DH]), tr(&st[2], 0, &[HK, l.cr, DH]), tr(&st[3], 0, &[HK, l.cr, DH])],
            a,
        );
        att.pos = pos;
        att.scratch = Some(scratch_for(be, ak, &att));
        be.launch(ak, &att, 0).unwrap();
        be.synchronize(0).unwrap();
    }
    let outs = vec![read(be, &q, t * H * DH), read(be, &rel, t * H * E), read(be, &k, t * ckv), read(be, &v, t * ckv), read(be, &o, t * H * DH)];
    let state = st.iter().zip(&inp.state0).map(|(b, s)| read(be, b, s.len() / 2)).collect();
    Run { outs, state }
}

fn same(a: &Run, b: &Run, what: &str) {
    for (i, (x, y)) in a.outs.iter().zip(&b.outs).enumerate() {
        if i < 4 && (x.iter().all(|&v| v == 0) || y.iter().all(|&v| v == 0)) {
            continue; // a fused run has no q / rel / k / v
        }
        let bad = x.iter().zip(y).filter(|(p, q)| p != q).count();
        assert_eq!(bad, 0, "{what}: output {i} (q, rel, k, v, o) differs in {bad} of {} values", x.len());
    }
    for (i, (x, y)) in a.state.iter().zip(&b.state).enumerate() {
        let bad = x.iter().zip(y).filter(|(p, q)| p != q).count();
        assert_eq!(bad, 0, "{what}: state {i} (k ring, v ring, k cache, v cache) differs in {bad} of {} values", x.len());
    }
}

#[test]
fn admission_full_sweep() {
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::arc();
    let quick = std::env::var("INK_ATTN_QUICK").is_ok_and(|v| v == "1");
    let sweep = SweepConfig { quick, ..SweepConfig::default() };
    let opts = AdmissionOpts { sweep: Some(sweep), only_ops: Some(vec![Op::InkKvPrologue, Op::RelSinkAttn, Op::InkAttnDecode]), only_kernel: None };
    let report = run_admission(&*cpu, &*hip, &opts).unwrap();
    let text = report.render(true);
    println!("{text}");
    for name in ["ink_kv_prologue_bf16", "rel_sink_attn_bf16", "rel_sink_attn_bf16_flash", "ink_attn_decode_bf16"] {
        let k = report.kernels.iter().find(|k| k.kernel == name).unwrap_or_else(|| panic!("{name} not graded\n{text}"));
        assert!(k.cases > 0 && k.admitted, "{name}: {} cases, admitted {}\n{text}", k.cases, k.admitted);
    }
}

#[test]
fn rows_exact_real_shapes() {
    let Some(be) = device() else { return };
    let be = be.as_ref();
    let swa = Layer { window: 1024, ring: true, cr: 1040, dynamic: false };
    let full = Layer { window: 0, ring: false, cr: 4096, dynamic: true };
    // full at 8180..: 32 splits of 288 keys, the trailing ones empty
    let full_long = Layer { window: 0, ring: false, cr: 8300, dynamic: true };
    let mut n = 0;
    for (l, poss) in [(&swa, vec![0usize, 1030, 2100, 1020]), (&full, vec![3, 1018, 3000, 1500]), (&full_long, vec![7900])] {
        for &t in &[2usize, 4, 7, 16, 40, 100, 256] {
            for &pos0 in poss.iter() {
                if !l.ring && pos0 + t > l.cr {
                    continue;
                }
                let inp = make(l, t, (t * 977 + pos0) as u64);
                let multi = run(be, l, &inp, t, pos0, t, false, false);
                let one = run(be, l, &inp, t, pos0, 1, false, false);
                let what = format!("window {} ring {} T={t} pos0={pos0}", l.window, l.ring);
                same(&multi, &one, &format!("{what} (host positions)"));
                if t <= 40 {
                    let one_dev = run(be, l, &inp, t, pos0, 1, true, false);
                    same(&multi, &one_dev, &format!("{what} (device position)"));
                } else {
                    let multi_dev = run(be, l, &inp, t, pos0, t, true, false);
                    same(&multi, &multi_dev, &format!("{what} ({t}-row call, device position)"));
                }
                // InkAttnDecode (op 75, <= 16 rows): R rows, R x 1 row (host and device position) all bitwise the unfused pair
                if t <= 16 {
                    for (rows, dev) in [(t, false), (t, true), (1, false), (1, true)] {
                        let f = run(be, l, &inp, t, pos0, rows, dev, true);
                        same(&multi, &f, &format!("{what} (InkAttnDecode, {rows}-row calls, device position {dev})"));
                    }
                }
                // the attention output is not trivially zero
                assert!(multi.outs[4].iter().any(|&x| x & 0x7fff != 0), "{what}: zero output");
                n += 1;
            }
        }
    }
    eprintln!("{n} rows-exact cases bitwise");
}

/// One-row decode at the real shapes: prologue + attention, timed over many launches (host position, eager). KV bytes read per
/// call: 2 (K, V) x Hk x keys x Dh x 2 bytes; roofline 627 GB/s (the Neon review's stream rate, unverified).
#[test]
#[ignore]
fn bench_decode() {
    let Some(be) = device() else { return };
    let be = be.as_ref();
    let ckv = HK * DH;
    let nqw = nq(H, DH, DR);
    for (name, l, pos0) in [
        ("swa 1024 keys", Layer { window: 1024, ring: true, cr: 1040, dynamic: false }, 4000usize),
        ("full 4k keys", Layer { window: 0, ring: false, cr: 4200, dynamic: true }, 4095),
        ("full 8k keys", Layer { window: 0, ring: false, cr: 8300, dynamic: true }, 8191),
        ("full 16k keys", Layer { window: 0, ring: false, cr: 16500, dynamic: true }, 16383),
    ] {
        let inp = make(&l, 1, 7);
        let (pq, pk, pv) = (upload(be, &inp.pq), upload(be, &inp.pk), upload(be, &inp.pv));
        let w: Vec<Buffer> = inp.w.iter().map(|v| upload(be, v)).collect();
        let st: Vec<Buffer> = inp.state0.iter().map(|v| upload(be, v)).collect();
        let (q, rel, k, v, o) = (zeros(be, H * DH), zeros(be, H * E), zeros(be, ckv), zeros(be, ckv), zeros(be, H * DH));
        let pw = inp.pk.len() / 2;
        let a = attrs(&l, pos0);
        let pro = OpCall::new(
            Op::InkKvPrologue,
            vec![
                tr(&pq, 0, &[1, nqw]),
                tr(&pk, 0, &[1, pw]),
                tr(&pv, 0, &[1, pw]),
                tr(&w[0], 0, &[K, ckv]),
                tr(&w[1], 0, &[K, ckv]),
                tr(&w[2], 0, &[H, DH]),
                tr(&w[3], 0, &[H, DR]),
                tr(&w[4], 0, &[DR, E]),
            ],
            vec![tr(&q, 0, &[1, H, DH]), tr(&rel, 0, &[1, H, E]), tr(&k, 0, &[1, HK, DH]), tr(&v, 0, &[1, HK, DH]), tr(&st[0], 0, &[RR, ckv]), tr(&st[1], 0, &[RR, ckv])],
            a.clone(),
        );
        let mut att = OpCall::new(
            Op::RelSinkAttn,
            vec![
                tr(&q, 0, &[1, H, DH]),
                tr(&k, 0, &[1, HK, DH]),
                tr(&v, 0, &[1, HK, DH]),
                tr(&st[2], 0, &[HK, l.cr, DH]),
                tr(&st[3], 0, &[HK, l.cr, DH]),
                tr(&rel, 0, &[1, H, E]),
                tr(&pq, 0, &[1, nqw]),
            ],
            vec![tr(&o, 0, &[1, H * DH]), tr(&st[2], 0, &[HK, l.cr, DH]), tr(&st[3], 0, &[HK, l.cr, DH])],
            a,
        );
        att.scratch = Some(scratch_for(be, "rel_sink_attn_bf16", &att));
        let fa = att.attrs.clone(); // attrs() carries the prologue's and the attention's fields
        let mut dec = OpCall::new(
            Op::InkAttnDecode,
            pro.inputs.iter().cloned().chain([tr(&st[2], 0, &[HK, l.cr, DH]), tr(&st[3], 0, &[HK, l.cr, DH])]).collect(),
            vec![tr(&o, 0, &[1, H * DH]), tr(&st[2], 0, &[HK, l.cr, DH]), tr(&st[3], 0, &[HK, l.cr, DH]), tr(&st[0], 0, &[RR, ckv]), tr(&st[1], 0, &[RR, ckv])],
            fa,
        );
        dec.scratch = Some(scratch_for(be, "ink_attn_decode_bf16", &dec));
        let time = |f: &dyn Fn()| {
            for _ in 0..20 {
                f();
            }
            be.synchronize(0).unwrap();
            let n = 500;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            be.synchronize(0).unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        };
        let tp = time(&|| be.launch("ink_kv_prologue_bf16", &pro, 0).unwrap());
        let ta = time(&|| be.launch("rel_sink_attn_bf16", &att, 0).unwrap());
        let tf = time(&|| be.launch("ink_attn_decode_bf16", &dec, 0).unwrap());
        let keys = (pos0 + 1).min(if l.window > 0 { l.window } else { usize::MAX });
        let bytes = (2 * HK * keys * DH * 2) as f64;
        let roof = bytes / 627e9 * 1e6;
        println!(
            "BENCH {name}: prologue {tp:.1} us, attention {ta:.1} us, pair {:.1} us, fused op 75 {tf:.1} us ({:.0} GB/s of KV; roofline {roof:.1} us at 627 GB/s, {:.0}%)",
            tp + ta,
            bytes / (ta * 1e-6) / 1e9,
            roof / ta * 100.0
        );
    }
}

/// A prefill chunk of 256 rows ending at 4k context (sliding and full layer), the split-KV kernel vs the flash kernel.
#[test]
#[ignore]
fn bench_prefill() {
    let Some(be) = device() else { return };
    let be = be.as_ref();
    let t = 256usize;
    let ckv = HK * DH;
    let nqw = nq(H, DH, DR);
    for (name, l, pos0) in [
        ("swa, 256 rows at 3840", Layer { window: 1024, ring: true, cr: 1040, dynamic: false }, 3840usize),
        ("full, 256 rows at 3840", Layer { window: 0, ring: false, cr: 4200, dynamic: true }, 3840),
        ("full, 256 rows at 0", Layer { window: 0, ring: false, cr: 4200, dynamic: true }, 0),
    ] {
        let inp = make(&l, t, 11);
        let pq = upload(be, &inp.pq);
        let st: Vec<Buffer> = inp.state0.iter().map(|v| upload(be, v)).collect();
        let (q, rel, k, v, o) = (
            upload(be, &rand_bf16(t * H * DH, 1, 1.5, 1.5)),
            upload(be, &rand_bf16(t * H * E, 2, 1.0, 1.0)),
            upload(be, &rand_bf16(t * ckv, 3, 1.5, 1.5)),
            upload(be, &rand_bf16(t * ckv, 4, 1.0, 1.0)),
            zeros(be, t * H * DH),
        );
        let mut att = OpCall::new(
            Op::RelSinkAttn,
            vec![
                tr(&q, 0, &[t, H, DH]),
                tr(&k, 0, &[t, HK, DH]),
                tr(&v, 0, &[t, HK, DH]),
                tr(&st[2], 0, &[HK, l.cr, DH]),
                tr(&st[3], 0, &[HK, l.cr, DH]),
                tr(&rel, 0, &[t, H, E]),
                tr(&pq, 0, &[t, nqw]),
            ],
            vec![tr(&o, 0, &[t, H * DH]), tr(&st[2], 0, &[HK, l.cr, DH]), tr(&st[3], 0, &[HK, l.cr, DH])],
            attrs(&l, pos0),
        );
        att.scratch = Some(scratch_for(be, "rel_sink_attn_bf16", &att));
        let time = |kn: &str| {
            for _ in 0..3 {
                be.launch(kn, &att, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let n = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                be.launch(kn, &att, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        };
        let ts = time("rel_sink_attn_bf16");
        let tf = time("rel_sink_attn_bf16_flash");
        println!("BENCH prefill {name}: split-KV {ts:.0} us, flash {tf:.0} us ({:.1}x)", ts / tf);
    }
}
