//! Lane dw: `attn_decode_bf16_window_splitkv` (kernels/hip/attention_win_splitkv.hip, the DFlash2 draft's windowed block attention split over the
//! key range, `OpAttrs::win_split`) against an f64 host reference and against `attn_decode_bf16_window` (the kernel it replaces for the draft;
//! the target's Gemma sliding layers keep the old one), over the draft's heads, windows and positions; the device-position launch is bitwise the
//! host-position one; `WS_BENCH=1` prints microseconds a call, old and new, at the draft's 4k shapes.
//!
//! Needs a GPU: skips (passes) without one. `INTARSIA_HIP_DEVICE` picks the ordinal.
#![cfg(ik_hip)]

use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, ShapeInfo, TensorRef};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn rand_t(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(n, scale))
}

fn attrs(n: usize, nkv: usize, d: usize, cap: usize, p: usize, window: usize, win_split: bool) -> OpAttrs {
    OpAttrs {
        pos0: p,
        scale: 1.0 / (d as f32).sqrt(),
        n_heads: n,
        n_kv_heads: nkv,
        head_dim: d,
        kv_cap: cap,
        window,
        win_split,
        ..OpAttrs::default()
    }
}

/// One decode row at position `p` over a cache of `p + 1 + 70` slots (the 70 past the edge are random too).
fn case(rng: &mut SplitMix64, n: usize, nkv: usize, d: usize, p: usize, window: usize, qscale: f32) -> HostCall {
    let cap = p + 1 + 70;
    HostCall {
        op: Op::AttnDecode,
        inputs: vec![rand_t(rng, &[1, n, d], qscale), rand_t(rng, &[nkv, cap, d], 1.0), rand_t(rng, &[nkv, cap, d], 1.0)],
        outputs: vec![OutSpec::new(DType::Bf16, &[1, n, d])],
        attrs: attrs(n, nkv, d, cap, p, window, true),
    }
}

/// f64 reference of the windowed decode row: keys max(0, p + 1 - window) ..= p (window 0: all).
fn reference(hc: &HostCall) -> Vec<f64> {
    let a = &hc.attrs;
    let (n, nkv, d, p) = (a.n_heads, a.n_kv_heads, a.head_dim, a.pos0);
    let (q, k, v) = (hc.inputs[0].to_f32(), hc.inputs[1].to_f32(), hc.inputs[2].to_f32());
    let cap = a.kv_cap;
    let first = if a.window > 0 && p + 1 > a.window { p + 1 - a.window } else { 0 };
    let mut out = vec![0f64; n * d];
    for h in 0..n {
        let kvh = h / (n / nkv);
        let s: Vec<f64> = (first..=p)
            .map(|j| (0..d).map(|e| q[h * d + e] as f64 * k[(kvh * cap + j) * d + e] as f64).sum::<f64>() * a.scale as f64)
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let l: f64 = w.iter().sum();
        for e in 0..d {
            out[h * d + e] = (first..=p).zip(&w).map(|(j, wj)| wj * v[(kvh * cap + j) * d + e] as f64).sum::<f64>() / l;
        }
    }
    out
}

fn err(got: &[f32], want: &[f64]) -> (f64, f64) {
    let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let e = *g as f64 - w;
        num += e * e;
        den += w * w;
        mx = mx.max(e.abs());
    }
    ((num / den.max(1e-30)).sqrt(), mx / scale)
}

#[test]
fn window_splitkv_matches_the_reference_and_the_window_kernel() {
    let Some(hip) = device() else { return };
    let mut rng = SplitMix64::new(4242);
    // the 27B draft whole (32 / 8 / 128) and split over tp2 (16 / 4 / 128), a group of 8 (16 / 2 / 256), a group of 3 at D = 32, the Gemma sliding shape
    let heads = [(32usize, 8usize, 128usize), (16, 4, 128), (16, 2, 256), (6, 2, 32), (16, 8, 256)];
    // positions: the first keys, around the window, a draft's 4k context (last = 4095 + 7), well past it
    let positions = [0usize, 7, 63, 64, 300, 2047, 2053, 2054, 2100, 2500, 4102, 6000];
    // windows: none, tiny, the draft's 2048 + 7 - r class, an odd one
    let windows = [0usize, 5, 64, 1037, 2048, 2049, 2051, 2054];
    let mut cases = 0usize;
    for &(n, nkv, d) in &heads {
        for &p in &positions {
            for &w in &windows {
                if p > 2600 && !matches!(w, 0 | 2051 | 2054) {
                    continue; // thin the long positions: keep the run short
                }
                let qscale = if (p + w) % 2 == 0 { 1.0 } else { 4.0 };
                let hc = case(&mut rng, n, nkv, d, p, w, qscale);
                let want = reference(&hc);
                let new = run_host(&*hip, "attn_decode_bf16_window_splitkv", &hc)
                    .unwrap_or_else(|e| panic!("new N={n} Nkv={nkv} D={d} pos0={p} window={w}: {e}"));
                let mut old_hc = hc.clone();
                old_hc.attrs.win_split = false;
                let old = run_host(&*hip, "attn_decode_bf16_window", &old_hc).unwrap();
                let (en, mn) = err(&new[0].to_f32(), &want);
                let (eo, mo) = err(&old[0].to_f32(), &want);
                // bf16 output: one rounding is ~4e-3 of the value; the split kernel adds f32 reassociation only
                assert!(en <= 8e-3.max(eo * 2.0) && mn <= 2e-2, "N={n} Nkv={nkv} D={d} pos0={p} window={w}: new rel_l2 {en:.2e} max {mn:.2e} (old {eo:.2e} / {mo:.2e})");
                assert!(new[0].to_f32().iter().any(|v| *v != 0.0), "N={n} pos0={p} window={w}: all zero");
                cases += 1;
            }
        }
    }
    println!("window_splitkv: {cases} cases against the f64 reference and the window kernel");
}

fn put(be: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    be.zero(&buf, 0).unwrap();
    be.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn get(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    let mut d = vec![0u8; t.nbytes()];
    be.synchronize(0).unwrap();
    be.download(&t.buf, t.offset, &mut d, 0).unwrap();
    d
}

/// The workspace `kernel` declares for this call (None: it uses none).
fn scratch_for(be: &dyn Backend, kernel: &str, inputs: &[TensorRef], out: &TensorRef, a: &OpAttrs) -> Option<TensorRef> {
    let call = OpCall::new(Op::AttnDecode, inputs.to_vec(), vec![out.clone()], a.clone());
    let k = be.kernels().into_iter().find(|k| k.name == kernel).expect("registered kernel");
    let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
    (need > 0).then(|| TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]))
}

fn launch(be: &dyn Backend, kernel: &str, inputs: &[TensorRef], out: &TensorRef, a: &OpAttrs, pos: Option<&TensorRef>, scratch: &Option<TensorRef>) {
    let mut call = OpCall::new(Op::AttnDecode, inputs.to_vec(), vec![out.clone()], a.clone());
    call.pos = pos.cloned();
    call.scratch = scratch.clone();
    be.launch(kernel, &call, 0).unwrap();
}

#[test]
fn window_splitkv_device_position_is_bitwise_the_host_position() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(77);
    for &(n, nkv, d) in &[(32usize, 8usize, 128usize), (16, 4, 128), (16, 2, 256)] {
        for &w in &[0usize, 5, 2051, 2054] {
            for &p in &[0usize, 63, 64, 2053, 2100, 4102] {
                let hc = case(&mut rng, n, nkv, d, p, w, 1.0);
                let ins: Vec<TensorRef> = hc.inputs.iter().map(|t| put(&*be, t)).collect();
                let run = |dev: bool| {
                    let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d]));
                    let mut a = hc.attrs.clone();
                    let pos = put(&*be, &HostTensor::from_i32(&[1], &[p as i32]));
                    if dev {
                        a.pos0 = 0;
                    }
                    let sc = scratch_for(&*be, "attn_decode_bf16_window_splitkv", &ins, &out, &a);
                    launch(&*be, "attn_decode_bf16_window_splitkv", &ins, &out, &a, dev.then_some(&pos), &sc);
                    get(&*be, &out)
                };
                assert!(run(false) == run(true), "N={n} Nkv={nkv} D={d} window={w} pos0={p}: device position differs from host position");
            }
        }
    }
}

/// `WS_BENCH=1 cargo test --release --test win_splitkv -- --nocapture`: microseconds a call, 40 back-to-back launches (a draft block's 5 layers x 8
/// rows) repeated; old = `attn_decode_bf16_window`, new = `attn_decode_bf16_window_splitkv`, at the draft's heads and a 4k context.
#[test]
fn window_splitkv_bench() {
    if std::env::var("WS_BENCH").ok().as_deref() != Some("1") {
        return;
    }
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(9);
    for &(n, nkv, d) in &[(32usize, 8usize, 128usize), (16, 4, 128)] {
        for &(p, w) in &[(4102usize, 2054usize), (4102, 2048), (2500, 2054)] {
            let hc = case(&mut rng, n, nkv, d, p, w, 1.0);
            let ins: Vec<TensorRef> = hc.inputs.iter().map(|t| put(&*be, t)).collect();
            let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, n, d]));
            for kernel in ["attn_decode_bf16_window", "attn_decode_bf16_window_splitkv"] {
                let mut a = hc.attrs.clone();
                a.win_split = kernel.ends_with("splitkv");
                let sc = scratch_for(&*be, kernel, &ins, &out, &a);
                for _ in 0..20 {
                    launch(&*be, kernel, &ins, &out, &a, None, &sc);
                }
                be.synchronize(0).unwrap();
                let reps = 50usize;
                let t0 = std::time::Instant::now();
                for _ in 0..reps * 40 {
                    launch(&*be, kernel, &ins, &out, &a, None, &sc);
                }
                be.synchronize(0).unwrap();
                let us = t0.elapsed().as_secs_f64() * 1e6 / (reps * 40) as f64;
                println!("WSBENCH N={n} Nkv={nkv} D={d} pos0={p} window={w} {kernel}: {us:.1} us a call");
            }
        }
    }
}
