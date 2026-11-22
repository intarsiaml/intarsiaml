#![cfg(target_os = "macos")]
//! The screened greedy head on Metal (lane ink-fp8, `intarsia_core::screen`): `logits_gather_f32o_v` is bitwise the serving
//! head's `Logits` (`logits_bf16_gemv` for 1 row, `logits_bf16_gemv_rows` for 2..=16 rows under `exact_rows`) for every
//! gathered (row, weight row), in the full-row fallback and on candidate lists, at the real Inkling-2-Mini width (D 2304);
//! and the whole screen picks `Argmax` of the full head.

mod common;
use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::screen::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
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

/// The head's own logits of `t` rows, by the kernel the serving head selects for that row count.
fn head_logits(be: &MetalBackend, x: &TensorRef, w: &TensorRef, t: usize, v: usize) -> Vec<f32> {
    let y = zeros(be, DType::F32, &[t, v]);
    let k = if t == 1 { intarsia_backend_metal::kernels::LOGITS_BF16_GEMV } else { intarsia_backend_metal::multirow::LOGITS_BF16_GEMV_ROWS };
    launch(be, k, Op::Logits, vec![x.clone(), w.clone()], vec![y.clone()], t > 1);
    f32s(&down(be, &y))
}

#[test]
fn gather_is_bitwise_the_head_logits() {
    let be = backend();
    let mut rng = Rng(0x6a7e);
    let (d, v, cap) = (2304usize, 50021usize, 64usize);
    let wb: Vec<u8> = (0..v * d).flat_map(|_| f32_to_bf16(rng.unit() * 0.03).to_le_bytes()).collect();
    let w = up(&be, DType::Bf16, &[v, d], &wb);
    for t in [1usize, 2, 3, 4, 7, 8, 16] {
        let xb: Vec<u8> = (0..t * d).flat_map(|_| f32_to_bf16(rng.unit() * 2.0).to_le_bytes()).collect();
        let x = up(&be, DType::Bf16, &[t, d], &xb);
        let full = head_logits(&be, &x, &w, t, v);
        // row r: the full-row fallback for even r, a candidate list (ascending, random) for odd r
        let mut cand = vec![0i32; t * cap];
        let mut cnt = vec![0i32; t];
        for r in 0..t {
            if r % 2 == 0 {
                cnt[r] = if r % 4 == 0 { 0 } else { cap as i32 + 1 };
            } else {
                let k = 1 + (rng.next_u64() % cap as u64) as usize;
                let mut n = (rng.next_u64() % 50) as usize;
                for j in 0..k {
                    cand[r * cap + j] = n.min(v - 1) as i32;
                    n += 1 + (rng.next_u64() % 700) as usize;
                }
                cnt[r] = k as i32;
            }
        }
        let ct = up(&be, DType::I32, &[t, cap], &i32b(&cand));
        let nt = up(&be, DType::I32, &[t], &i32b(&cnt));
        let y = zeros(&be, DType::F32, &[t, v]);
        launch(&be, LOGITS_GATHER_F32O_V, Op::LogitsGather, vec![x.clone(), w.clone(), ct, nt], vec![y.clone()], t > 1);
        let got = f32s(&down(&be, &y));
        for r in 0..t {
            if r % 2 == 0 {
                for n in 0..v {
                    assert_eq!(got[r * v + n].to_bits(), full[r * v + n].to_bits(), "T={t} row {r} full n {n}");
                }
            } else {
                for j in 0..cnt[r] as usize {
                    let n = cand[r * cap + j] as usize;
                    assert_eq!(got[r * v + j].to_bits(), full[r * v + n].to_bits(), "T={t} row {r} cand {j} (n {n})");
                }
            }
        }
    }
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
    for trial in 0..24 {
        let t = [1usize, 4, 16][trial % 3];
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
        let full = head_logits(&be, &x, &w, t, v);
        let want: Vec<i32> = (0..t).map(|r| intarsia_core::sampler::argmax(&full[r * v..(r + 1) * v]) as i32).collect();
        let a = zeros(&be, DType::F32, &[t, v]);
        let sc = q.call(x.clone(), a.clone());
        let k = if t == 1 { intarsia_backend_metal::fp8::MATMUL_FP8_GEMV } else { intarsia_backend_metal::fp8::MATMUL_FP8_GEMV_ROWS };
        let mut call = OpCall::new(sc.op, sc.inputs, sc.outputs, sc.attrs);
        call.exact_rows = t > 1;
        be.launch(k, &call, 0).unwrap();
        let ct = zeros(&be, DType::I32, &[t, cap]);
        let nt = zeros(&be, DType::I32, &[t]);
        launch(&be, HEAD_SCREEN_F32, Op::HeadScreen, vec![a.clone(), x.clone(), bt.clone()], vec![ct.clone(), nt.clone()], t > 1);
        let y = zeros(&be, DType::F32, &[t, v]);
        launch(&be, LOGITS_GATHER_F32O_V, Op::LogitsGather, vec![x.clone(), w.clone(), ct.clone(), nt.clone()], vec![y.clone()], t > 1);
        let tok = zeros(&be, DType::I32, &[t]);
        launch(&be, SCREEN_ARGMAX_F32, Op::ScreenArgmax, vec![ct.clone(), nt.clone(), y], vec![tok.clone()], t > 1);
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
