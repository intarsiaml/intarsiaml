//! The screened greedy head on CUDA (src/inkscreen.rs, kernels/cuda/ink_screen.cu, the gather in inkling2_p12.cu; contract
//! `intarsia_core::screen`), on a P12 head with near-ties planted at the top:
//!
//! * HeadScreen is bitwise the core's `screen_row` on the device screen values (candidates and count);
//! * a near-tie of more rows than the cap sends the step through the full-row fallback (count > cap), the gather then writes
//!   every logit bitwise the serving head's (`matmul_p12_logits_gemv`), and ScreenArgmax returns the full head's argmax
//!   (lowest index of the maximum);
//! * a near-tie within the cap takes the candidate path: every gathered logit is the head's logit of that row, bit for bit,
//!   every maximiser is a candidate, and the token is the full head's;
//! * +inf bounds (the model's forced fallback) mark the row non-finite and the token is still the full head's.
//!
//! The screen values come from the fp8 GEMV the model runs for the screen (`matmul_fp8_gemv_rows`).
//!
//! Needs a GPU: without one the test prints SKIPPED (and fails under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::quant::head::{quantize_host, HeadQuant};
use intarsia_core::quant::p12;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, MemKind, Op, OpAttrs, OpCall, TensorRef};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("inkling2_screen", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn up(be: &dyn Backend, dt: DType, shape: &[usize], data: &[u8]) -> TensorRef {
    let b = be.alloc(data.len().max(16), MemKind::Device).unwrap();
    if !data.is_empty() {
        be.upload(&b, 0, data, 0).unwrap();
    }
    TensorRef { buf: b, offset: 0, dtype: dt, shape: shape.to_vec() }
}

fn down(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    let mut v = vec![0u8; t.numel() * t.dtype.size_bytes()];
    be.synchronize(0).unwrap();
    be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    v
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn launch(be: &dyn Backend, kernel: &str, op: Op, ins: Vec<TensorRef>, outs: Vec<TensorRef>) {
    let mut call = OpCall::new(op, ins, outs, OpAttrs::default());
    call.exact_rows = true;
    call.check().unwrap_or_else(|e| panic!("{kernel}: {e}"));
    be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
}

/// The full head's argmax: the lowest index of the maximum (strict `>`), as `Argmax`.
fn argmax(y: &[f32]) -> i32 {
    let (mut bv, mut bi) = (f32::NEG_INFINITY, 0usize);
    for (n, &v) in y.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = n;
        }
    }
    bi as i32
}

struct Run {
    head: Vec<f32>,
    cand: Vec<i32>,
    cnt: i32,
    y: Vec<f32>,
    tok: i32,
}

/// One screened step on CUDA against the full head: `ties` rows (from row `t0`) copy the top row, every other one of them with
/// its first element one bf16 ulp up (a near-tie), the rest random and far below.
fn run(be: &dyn Backend, v: usize, d: usize, t0: usize, ties: usize, cap: usize, inf_beta: bool) -> Run {
    let mut rng = SplitMix64::new(7 + ties as u64);
    let h: Vec<f32> = rng.normal_vec(d, 1.0).iter().map(|x| half::bf16::from_f32(*x).to_f32()).collect();
    let mut w: Vec<u16> = rng.normal_vec(v * d, 0.3 * (d as f32).powf(-0.5)).iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
    let top: Vec<u16> = h.iter().map(|x| half::bf16::from_f32(x * 0.05).to_bits()).collect();
    for r in t0..t0 + ties {
        w[r * d..(r + 1) * d].copy_from_slice(&top);
        if (r - t0) % 2 == 1 {
            w[r * d] = w[r * d].wrapping_add(1); // one ulp of the first element: a logit within the screen's bound
        }
    }
    let wb: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let hb: Vec<u8> = h.iter().flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes()).collect();
    let x = up(be, DType::Bf16, &[1, d], &hb);
    // the serving head: P12 codes, the f32-out GEMV the model runs at one row
    let p = p12::pack(&w, v, d, 0).unwrap();
    let aux: Vec<u8> = p.aux.iter().flat_map(|a| a.to_le_bytes()).collect();
    let codes = up(be, DType::U8, &[v, p12::row_bytes(d)], &p.codes);
    let auxt = up(be, DType::I32, &[p.aux.len()], &aux);
    let headt = up(be, DType::F32, &[1, v], &vec![0u8; v * 4]);
    let mut c = OpCall::new(Op::MatmulP12, vec![x.clone(), codes.clone(), auxt.clone()], vec![headt.clone()], OpAttrs::default());
    c.check().unwrap();
    be.launch("matmul_p12_logits_gemv", &c, 0).unwrap();
    let head = f32s(&down(be, &headt));
    // the screen: fp8 copy, bounds, the fp8 GEMV, HeadScreen, the gather, ScreenArgmax
    let hq = quantize_host(HeadQuant::Fp8, DType::Bf16, &wb, v, d, 0, v).unwrap();
    let s = hq.parts[1].to_f32()[0];
    let beta = intarsia_core::screen::row_bounds(DType::Bf16, &wb, &hq.parts[0].data, s, v, d).unwrap();
    let beta: Vec<f32> = if inf_beta { vec![f32::INFINITY; v] } else { beta };
    let q = hq.upload(be).unwrap();
    let a = up(be, DType::F32, &[1, v], &vec![0u8; v * 4]);
    launch(be, "matmul_fp8_gemv_rows", Op::MatmulFp8, vec![x.clone(), q.parts[0].clone(), q.parts[1].clone()], vec![a.clone()]);
    let betat = up(be, DType::F32, &[v], &beta.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>());
    let cand = up(be, DType::I32, &[1, cap], &vec![0xAAu8; cap * 4]);
    let cnt = up(be, DType::I32, &[1], &[0u8; 4]);
    launch(be, "head_screen_f32", Op::HeadScreen, vec![a.clone(), x.clone(), betat], vec![cand.clone(), cnt.clone()]);
    let yt = up(be, DType::F32, &[1, v], &vec![0xFFu8; v * 4]); // NaN everywhere: unwritten slots stay visible
    launch(be, "logits_gather_p12", Op::LogitsGather, vec![x.clone(), codes, cand.clone(), cnt.clone(), auxt], vec![yt.clone()]);
    let tok = up(be, DType::I32, &[1], &[0u8; 4]);
    launch(be, "screen_argmax_f32", Op::ScreenArgmax, vec![cand.clone(), cnt.clone(), yt.clone()], vec![tok.clone()]);
    let (cand_v, cnt_v) = (i32s(&down(be, &cand)), i32s(&down(be, &cnt))[0]);
    // HeadScreen is bitwise the core's screen_row on the device screen values
    let mut want_cand = vec![0i32; cap];
    let want_cnt = intarsia_core::screen::screen_row(&f32s(&down(be, &a)), &h, &beta, &mut want_cand);
    assert_eq!((cnt_v, &cand_v), (want_cnt, &want_cand), "HeadScreen vs screen_row (ties {ties}, cap {cap})");
    Run { head, cand: cand_v, cnt: cnt_v, y: f32s(&down(be, &yt)), tok: i32s(&down(be, &tok))[0] }
}

#[test]
fn screened_head_near_ties_fall_back_and_match_the_full_head() {
    let Some(be) = device() else { return };
    let (v, d) = (4096, 256);
    // more near-tied rows than the cap: the full-row fallback
    let r = run(&*be, v, d, 1000, 40, 8, false);
    assert!(r.cnt > 8, "40 near-tied rows must exceed the cap of 8 (cnt {})", r.cnt);
    for n in 0..v {
        assert_eq!(r.y[n].to_bits(), r.head[n].to_bits(), "fallback gather row {n} vs the serving head");
    }
    assert_eq!(r.tok, argmax(&r.head), "fallback token vs the full head's argmax");
    assert_eq!(r.tok, 1000, "exact ties: the lowest index of the maximum");
    // the forced fallback (+inf bounds): non-finite row, every logit, the full head's token
    let r = run(&*be, v, d, 1000, 40, 8, true);
    assert_eq!(r.cnt, i32::MAX, "+inf bounds mark the row non-finite");
    assert!(r.cand.iter().all(|&c| c == -1));
    assert!(r.y.iter().zip(&r.head).all(|(a, b)| a.to_bits() == b.to_bits()), "forced fallback gather vs the serving head");
    assert_eq!(r.tok, argmax(&r.head));
}

#[test]
fn screened_head_near_ties_within_the_cap_take_the_candidate_path() {
    let Some(be) = device() else { return };
    let (v, d) = (4096, 256);
    let r = run(&*be, v, d, 2000, 3, 64, false);
    assert!((1..=64).contains(&r.cnt), "3 near-tied rows: the candidate path (cnt {})", r.cnt);
    let c = r.cnt as usize;
    assert!(r.cand[..c].windows(2).all(|p| p[0] < p[1]), "candidates ascending");
    assert!(r.cand[c..].iter().all(|&x| x == -1), "slots past the candidates are -1");
    for j in 0..c {
        let n = r.cand[j] as usize;
        assert_eq!(r.y[j].to_bits(), r.head[n].to_bits(), "gathered slot {j} (row {n}) vs the serving head");
    }
    let best = r.head.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for (n, &x) in r.head.iter().enumerate() {
        if x == best {
            assert!(r.cand[..c].contains(&(n as i32)), "maximiser {n} is a candidate");
        }
    }
    assert!((2000..2003).all(|n| r.cand[..c].contains(&n)), "the planted near-tie rows are candidates");
    assert_eq!(r.tok, argmax(&r.head), "candidate-path token vs the full head's argmax");
}
