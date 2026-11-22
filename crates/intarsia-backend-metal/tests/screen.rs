#![cfg(target_os = "macos")]
//! The screened greedy head on Metal (lane ink-fp8, `intarsia_core::screen`; brief 14 option B): the gather is bitwise the
//! serving 1-row head for every gathered weight row, in the full-row fallback and on candidate lists, at the real
//! Inkling-2-Mini width (D 2304): `logits_gather_f32o_v` against `logits_bf16_gemv` (bf16 head), `logits_gather_p12` against
//! `matmul_p12_logits_gemv` (P12 head); and the whole screen picks `Argmax` of the full head.

mod common;
use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::screen::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::p12;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, Op, OpAttrs, OpCall};

fn up(be: &MetalBackend, dt: DType, shape: &[usize], data: &[u8]) -> TensorRef {
    let b = be.alloc(data.len().max(16), MemKind::Device).unwrap();
    be.upload(&b, 0, data, 0).unwrap();
    TensorRef { buf: b, offset: 0, dtype: dt, shape: shape.to_vec() }
}

fn zeros(be: &MetalBackend, dt: DType, shape: &[usize]) -> TensorRef {
    let n = shape.iter().product::<usize>() * dt.size_bytes();
    up(be, dt, shape, &vec![0u8; n])
}

fn down(be: &MetalBackend, t: &TensorRef) -> Vec<u8> {
    let mut v = vec![0u8; t.numel() * t.dtype.size_bytes()];
    be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    v
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn launch(be: &MetalBackend, kernel: &str, op: Op, ins: Vec<TensorRef>, outs: Vec<TensorRef>, exact: bool) {
    let mut call = OpCall::new(op, ins, outs, OpAttrs::default());
    call.exact_rows = exact;
    call.check().unwrap();
    be.launch(kernel, &call, 0).unwrap();
    be.synchronize(0).unwrap();
}

/// The 1-row bf16 head's own logits (`logits_bf16_gemv`, the kernel the serving head selects for one row).
fn head_logits(be: &MetalBackend, x: &TensorRef, w: &TensorRef, v: usize) -> Vec<f32> {
    let y = zeros(be, DType::F32, &[1, v]);
    launch(be, intarsia_backend_metal::kernels::LOGITS_BF16_GEMV, Op::Logits, vec![x.clone(), w.clone()], vec![y.clone()], false);
    f32s(&down(be, &y))
}

/// Candidate lists of every shape the gather sees: empty (count 0: fallback), over the cap (fallback), one, a few, the cap,
/// ascending random rows including row 0 and the last row.
fn cand_cases(rng: &mut Rng, v: usize, cap: usize) -> Vec<(Vec<i32>, i32)> {
    let mut out = vec![(vec![0i32; cap], 0), (vec![0i32; cap], cap as i32 + 1), (vec![-1i32; cap], -1)];
    for k in [1usize, 2, 7, cap] {
        let mut cand = vec![0i32; cap];
        let mut n = if k == 2 { 0 } else { (rng.next_u64() % 50) as usize };
        for j in 0..k {
            cand[j] = n.min(v - 1) as i32;
            n += 1 + (rng.next_u64() % (v as u64 / cap as u64)) as usize;
        }
        if k == 2 {
            cand[1] = (v - 1) as i32;
        }
        out.push((cand, k as i32));
    }
    out
}

/// One gather against the full head row: `full` in the fallback, the candidates' entries otherwise, bit for bit; the rest of
/// `y` is left as it was (a sentinel).
fn check_gather(be: &MetalBackend, kernel: &str, ins: (&TensorRef, &TensorRef, Option<&TensorRef>), full: &[f32], cand: &[i32], cnt: i32, what: &str) {
    let (v, cap) = (full.len(), cand.len());
    let ct = up(be, DType::I32, &[1, cap], &i32b(cand));
    let nt = up(be, DType::I32, &[1], &i32b(&[cnt]));
    let sentinel = f32::from_bits(0x7fc0_dead);
    let y = up(be, DType::F32, &[1, v], &vec![sentinel; v].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let mut inputs = vec![ins.0.clone(), ins.1.clone(), ct, nt];
    if let Some(a) = ins.2 {
        inputs.push(a.clone());
    }
    launch(be, kernel, Op::LogitsGather, inputs, vec![y.clone()], false);
    let got = f32s(&down(be, &y));
    if intarsia_core::screen::candidate_path(cnt, cap) {
        for j in 0..cnt as usize {
            let n = cand[j] as usize;
            assert_eq!(got[j].to_bits(), full[n].to_bits(), "{what}: cand {j} (n {n})");
        }
        assert!(got[cnt as usize..].iter().all(|x| x.to_bits() == sentinel.to_bits()), "{what}: y past the candidates untouched");
    } else {
        for n in 0..v {
            assert_eq!(got[n].to_bits(), full[n].to_bits(), "{what}: full n {n}");
        }
    }
}

#[test]
fn gather_is_bitwise_the_head_logits() {
    let be = backend();
    let mut rng = Rng(0x6a7e);
    let (d, v, cap) = (2304usize, 50021usize, 64usize);
    let wb: Vec<u8> = (0..v * d).flat_map(|_| f32_to_bf16(rng.unit() * 0.03).to_le_bytes()).collect();
    let w = up(&be, DType::Bf16, &[v, d], &wb);
    for trial in 0..4 {
        let xb: Vec<u8> = (0..d).flat_map(|_| f32_to_bf16(rng.unit() * 2.0).to_le_bytes()).collect();
        let x = up(&be, DType::Bf16, &[1, d], &xb);
        let full = head_logits(&be, &x, &w, v);
        for (c, (cand, cnt)) in cand_cases(&mut rng, v, cap).into_iter().enumerate() {
            check_gather(&be, LOGITS_GATHER_F32O_V, (&x, &w, None), &full, &cand, cnt, &format!("bf16 trial {trial} case {c} cnt {cnt}"));
        }
    }
}

/// The P12 head (default-on on ink/metal): `logits_gather_p12` is bitwise `matmul_p12_logits_gemv`, which is bitwise the bf16
/// head (tests/p12.rs), on weight-like rows with escapes and adversarial rows, at the real unembed width.
#[test]
fn gather_is_bitwise_the_p12_head_logits() {
    let be = backend();
    let mut rng = Rng(0x9e12);
    let mut srng = SplitMix64::new(0x512);
    let (d, v, cap) = (2304usize, 20011usize, 64usize);
    for adversarial in [false, true] {
        let wf = p12::random_weight(&mut srng, v, d, (d as f32).powf(-0.5), adversarial);
        let pk = p12::pack(&wf, v, d, 0).unwrap();
        let codes = up(&be, DType::U8, &[v, p12::row_bytes(d)], &pk.codes);
        let aux = up(&be, DType::I32, &[pk.aux.len()], &pk.aux.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let wbf = up(&be, DType::Bf16, &[v, d], &wf.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        for trial in 0..3 {
            let xb: Vec<u8> = (0..d).flat_map(|_| f32_to_bf16(rng.unit() * 2.0).to_le_bytes()).collect();
            let x = up(&be, DType::Bf16, &[1, d], &xb);
            let y = zeros(&be, DType::F32, &[1, v]);
            let head = intarsia_backend_metal::p12::MATMUL_P12_LOGITS_GEMV;
            launch(&be, head, Op::MatmulP12, vec![x.clone(), codes.clone(), aux.clone()], vec![y.clone()], false);
            let full = f32s(&down(&be, &y));
            // and the P12 head is the bf16 head (so either gather serves the same tokens)
            let bf = head_logits(&be, &x, &wbf, v);
            assert!(full.iter().zip(&bf).all(|(a, b)| a.to_bits() == b.to_bits()), "P12 head vs bf16 head");
            for (c, (cand, cnt)) in cand_cases(&mut rng, v, cap).into_iter().enumerate() {
                let what = format!("p12 adv {adversarial} trial {trial} case {c} cnt {cnt}");
                check_gather(&be, LOGITS_GATHER_P12, (&x, &codes, Some(&aux)), &full, &cand, cnt, &what);
            }
        }
    }
}

/// The gathers refuse more than one activation row (the screen serves 1-row heads; the multi-row head is never screened).
#[test]
fn gather_refuses_multi_row() {
    let be = backend();
    let (d, v, cap) = (64usize, 40usize, 8usize);
    let x = zeros(&be, DType::Bf16, &[2, d]);
    let w = zeros(&be, DType::Bf16, &[v, d]);
    let call = OpCall::new(Op::LogitsGather, vec![x, w, zeros(&be, DType::I32, &[2, cap]), zeros(&be, DType::I32, &[2])], vec![zeros(&be, DType::F32, &[2, v])], OpAttrs::default());
    call.check().unwrap();
    assert!(be.launch(LOGITS_GATHER_F32O_V, &call, 0).is_err());
}

/// End to end on Metal: MatmulFp8 screen values, HeadScreen, LogitsGather, ScreenArgmax = Argmax of the full head, with an
/// fp8 copy and bounds built exactly as the model builds them; rows with a planted exact tie at the maximum.
#[test]
fn screened_argmax_is_the_full_argmax() {
    let be = backend();
    let mut rng = Rng(0x5c4e);
    let (d, v, cap) = (2304usize, 30011usize, 64usize);
    let mut wf: Vec<f32> = (0..v * d).map(|_| rng.unit() * 0.03).collect();
    // duplicate rows: exact ties of whole logits
    for r in (5..v).step_by(997) {
        let src = (r * 13) % v;
        let row: Vec<f32> = wf[src * d..(src + 1) * d].to_vec();
        wf[r * d..(r + 1) * d].copy_from_slice(&row);
    }
    let wb: Vec<u8> = wf.iter().flat_map(|x| f32_to_bf16(*x).to_le_bytes()).collect();
    let hq = intarsia_core::quant::head::quantize_host(intarsia_core::quant::head::HeadQuant::Fp8, DType::Bf16, &wb, v, d, 0, v).unwrap();
    let s = hq.parts[1].to_f32()[0];
    let beta = intarsia_core::screen::row_bounds(DType::Bf16, &wb, &hq.parts[0].data, s, v, d).unwrap();
    let q = hq.upload(&be).unwrap();
    let w = up(&be, DType::Bf16, &[v, d], &wb);
    let bt = up(&be, DType::F32, &[v], &beta.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let (mut cand_rows, mut fallbacks) = (0usize, 0usize);
    for trial in 0..32 {
        let t = 1usize;
        // activations aligned with a few weight rows (peaked logits, as a trained head), the rest noise
        let mut xf: Vec<f32> = (0..t * d).map(|_| rng.unit()).collect();
        for r in 0..t {
            let n = (rng.next_u64() % v as u64) as usize;
            for c in 0..d {
                xf[r * d + c] += 20.0 * wf[n * d + c];
            }
        }
        let xb: Vec<u8> = xf.iter().flat_map(|x| f32_to_bf16(*x).to_le_bytes()).collect();
        let x = up(&be, DType::Bf16, &[t, d], &xb);
        let full = head_logits(&be, &x, &w, v);
        let want: Vec<i32> = (0..t).map(|r| intarsia_core::sampler::argmax(&full[r * v..(r + 1) * v]) as i32).collect();
        let a = zeros(&be, DType::F32, &[t, v]);
        let sc = q.call(x.clone(), a.clone());
        let k = intarsia_backend_metal::fp8::MATMUL_FP8_GEMV;
        let call = OpCall::new(sc.op, sc.inputs, sc.outputs, sc.attrs);
        be.launch(k, &call, 0).unwrap();
        let ct = zeros(&be, DType::I32, &[t, cap]);
        let nt = zeros(&be, DType::I32, &[t]);
        launch(&be, HEAD_SCREEN_F32, Op::HeadScreen, vec![a.clone(), x.clone(), bt.clone()], vec![ct.clone(), nt.clone()], false);
        let y = zeros(&be, DType::F32, &[t, v]);
        launch(&be, LOGITS_GATHER_F32O_V, Op::LogitsGather, vec![x.clone(), w.clone(), ct.clone(), nt.clone()], vec![y.clone()], false);
        let tok = zeros(&be, DType::I32, &[t]);
        launch(&be, SCREEN_ARGMAX_F32, Op::ScreenArgmax, vec![ct.clone(), nt.clone(), y], vec![tok.clone()], false);
        let got: Vec<i32> = down(&be, &tok).chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let cnts: Vec<i32> = down(&be, &nt).chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // the screen's candidates and counts are the cpu reference's, bit for bit
        let af = f32s(&down(&be, &a));
        let xs: Vec<f32> = xb.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
        let cands: Vec<i32> = down(&be, &ct).chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        for r in 0..t {
            let mut cr = vec![0i32; cap];
            let cn = intarsia_core::screen::screen_row(&af[r * v..(r + 1) * v], &xs[r * d..(r + 1) * d], &beta, &mut cr);
            assert_eq!(cnts[r], cn, "trial {trial} row {r}: count vs cpu");
            if intarsia_core::screen::candidate_path(cn, cap) {
                assert_eq!(&cands[r * cap..r * cap + cn as usize], &cr[..cn as usize], "trial {trial} row {r}: candidates vs cpu");
                cand_rows += 1;
            } else {
                fallbacks += 1;
            }
        }
        assert_eq!(got, want, "trial {trial} T={t} counts {cnts:?}");
    }
    eprintln!("screened rows {cand_rows}, full-row fallbacks {fallbacks}");
    assert!(cand_rows > 0);
}
