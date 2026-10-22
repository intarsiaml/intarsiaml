//! `moe_experts_bf16` against the cpu reference math (`intarsia_core::quant::reference::moe_experts_token`) at the
//! Qwen3.6-35B-A3B expert shapes (256 experts, top-8, H 2048, I 512): the decode path (T*K <= 16: T = 1, 2) and the grouped
//! prefill path (T = 3, 300), out-of-range ids, run-to-run bitwise output of the grouped path (atomic scatter), and expert
//! weights bound at 8 mod 16 inside a larger buffer (how safetensors shards place most expert tensors).
#![cfg(target_os = "macos")]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::cases::{random_experts, random_routing, HostExperts};
use intarsia_core::quant::reference::{moe_experts_token, ExpertsRef};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use intarsia_backend_metal::MetalBackend;

const E: usize = 256;
const K: usize = 8;
const H: usize = 2048;
const I: usize = 512;

/// Upload `t` at byte `off` of a fresh buffer.
fn put(be: &MetalBackend, t: &HostTensor, off: usize) -> TensorRef {
    let b = be.alloc((off + t.nbytes()).max(16), MemKind::Device).unwrap();
    be.upload(&b, off, &t.data, 0).unwrap();
    TensorRef { buf: b, offset: off, dtype: t.dtype, shape: t.shape.clone() }
}

/// Run `moe_experts_bf16` with the weights at byte `woff` and the block scales at byte `soff` (down scales at `soff + 2`,
/// so their 4-byte-aligned skips differ) of their buffers; output of `odt`, as f32.
#[allow(clippy::too_many_arguments)]
fn run_with(
    be: &MetalBackend,
    x: &HostTensor,
    ids: &HostTensor,
    ws: &HostTensor,
    ex: &HostExperts,
    woff: usize,
    soff: usize,
    odt: DType,
) -> intarsia_core::Result<Vec<f32>> {
    let t = x.shape[0];
    let mut inputs = vec![put(be, x, 0), put(be, ids, 0), put(be, ws, 0)];
    inputs.push(put(be, &ex.gate_up_w, woff));
    inputs.push(put(be, &ex.gate_up_s, soff));
    inputs.push(put(be, &ex.down_w, woff));
    inputs.push(put(be, &ex.down_s, if soff == 0 { 0 } else { soff + 2 }));
    inputs.push(put(be, &ex.gscales, 0));
    let out = put(be, &HostTensor::zeros(odt, &[t, H]), 0);
    let mut call = OpCall::new(Op::MoeExperts, inputs, vec![out.clone()], OpAttrs { top_k: K, ..OpAttrs::default() });
    call.check()?;
    let ks = be.kernels();
    let k = ks.iter().find(|k| k.name == "moe_experts_bf16").expect("registered");
    let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
    let sb = be.alloc(need.max(16), MemKind::Device)?;
    call.scratch = Some(TensorRef::new(sb, DType::U8, &[need]));
    be.launch("moe_experts_bf16", &call, 0)?;
    be.synchronize(0)?;
    let mut data = vec![0u8; out.nbytes()];
    be.download(&out.buf, out.offset, &mut data, 0)?;
    Ok(HostTensor::new(odt, vec![t, H], data)?.to_f32())
}

/// `run_with` at bf16 output, aligned block scales.
fn run(be: &MetalBackend, x: &HostTensor, ids: &HostTensor, ws: &HostTensor, ex: &HostExperts, woff: usize) -> intarsia_core::Result<Vec<f32>> {
    run_with(be, x, ids, ws, ex, woff, 0, DType::Bf16)
}

fn reference(x: &HostTensor, ids: &[i32], ws: &[f32], ex: &HostExperts, odt: DType) -> Vec<f64> {
    let t = x.shape[0];
    let xf = x.to_f32();
    let gs = ex.gscales.to_f32();
    let er = ExpertsRef {
        e: ex.gate_up_w.shape[0],
        h: H,
        i: I,
        gate_up_w: &ex.gate_up_w.data,
        gate_up_s: &ex.gate_up_s.data,
        down_w: &ex.down_w.data,
        down_s: &ex.down_s.data,
        gscales: &gs,
    };
    let threads = std::thread::available_parallelism().map_or(4, |p| p.get());
    let mut out = vec![0f64; t * H];
    let chunk = t.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (c, oc) in out.chunks_mut(chunk * H).enumerate() {
            let (xf, er) = (&xf, &er);
            s.spawn(move || {
                for (j, o) in oc.chunks_mut(H).enumerate() {
                    let ti = c * chunk + j;
                    let r = moe_experts_token(er, &xf[ti * H..(ti + 1) * H], &ids[ti * K..(ti + 1) * K], &ws[ti * K..(ti + 1) * K], DType::Bf16, odt);
                    o.copy_from_slice(&r);
                }
            });
        }
    });
    out
}

fn grade(got: &[f32], want: &[f64], what: &str) {
    let (mut d2, mut r2, mut max_abs, mut scale) = (0f64, 0f64, 0f64, 0f64);
    for (g, r) in got.iter().zip(want) {
        let d = *g as f64 - r;
        d2 += d * d;
        r2 += r * r;
        max_abs = max_abs.max(d.abs());
        scale = scale.max(r.abs());
    }
    let rel = (d2 / r2.max(1e-30)).sqrt();
    eprintln!("{what}: max_abs {max_abs:.3e} (scale {scale:.3e}) rel_l2 {rel:.3e}");
    assert!(rel <= 1e-3 && max_abs <= 1.5e-2 * scale.max(1.0), "{what}: max_abs {max_abs:.3e} rel_l2 {rel:.3e}");
}

#[test]
fn moe_experts_matches_the_reference_on_both_paths() {
    let be = common::backend();
    let mut rng = SplitMix64::new(0x35a3b);
    let ex = random_experts(&mut rng, E, H, I);
    for t in [1usize, 2, 3, 300] {
        let x = HostTensor::from_f32_as(DType::Bf16, &[t, H], &rng.normal_vec(t * H, 1.0));
        let (ids, ws) = random_routing(&mut rng, t, E, K);
        let mut idv = ids.to_i32().unwrap();
        // out-of-range ids are skipped: one -1 and one E (in the last token, so both paths see them)
        if t >= 2 {
            idv[(t - 1) * K + 1] = -1;
            idv[(t - 1) * K + 5] = E as i32;
        }
        let ids = HostTensor::from_i32(&[t, K], &idv);
        let want = reference(&x, &idv, &ws.to_f32(), &ex, DType::Bf16);
        let path = if t * K <= 16 { "decode" } else { "grouped" };
        let got = run(&be, &x, &ids, &ws, &ex, 0).unwrap();
        grade(&got, &want, &format!("T={t} ({path})"));
        if t == 300 {
            // the grouped path's slot order comes from atomics; the values must not depend on it
            let again = run(&be, &x, &ids, &ws, &ex, 0).unwrap();
            assert!(got.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "T=300: not run-to-run bitwise");
            // expert tensors at 8 mod 16 (bitwise the aligned binding)
            let mis = run(&be, &x, &ids, &ws, &ex, 8).unwrap();
            assert!(got.iter().zip(&mis).all(|(a, b)| a.to_bits() == b.to_bits()), "T=300: weights at 8 mod 16 differ");
        }
        if t == 1 {
            let mis = run(&be, &x, &ids, &ws, &ex, 8).unwrap();
            assert!(got.iter().zip(&mis).all(|(a, b)| a.to_bits() == b.to_bits()), "T=1: weights at 8 mod 16 differ");
            // 2 mod 4 cannot take the packed 4-byte loads: a clear error, never a misread
            let err = run(&be, &x, &ids, &ws, &ex, 2).unwrap_err();
            assert!(err.to_string().contains("4-byte aligned"), "{err}");
        }
    }
}

/// Skewed routing: T = 300 tokens over 12 hot experts (of 16) is ~200 rows per expert, so every expert runs several 32-row
/// tiles of the grouped path (the uniform T = 300 case above puts ~9 rows on each expert: one tile each).
#[test]
fn moe_experts_skewed_routing_runs_several_tiles_per_expert() {
    let be = common::backend();
    let mut rng = SplitMix64::new(0x35a3c);
    let (e, hot, t) = (16usize, 12usize, 300usize);
    let ex = random_experts(&mut rng, e, H, I);
    let x = HostTensor::from_f32_as(DType::Bf16, &[t, H], &rng.normal_vec(t * H, 1.0));
    let (_, ws) = random_routing(&mut rng, t, e, K);
    let mut idv = Vec::with_capacity(t * K);
    for _ in 0..t {
        // K distinct experts out of the first `hot`
        let mut pool: Vec<i32> = (0..hot as i32).collect();
        for j in 0..K {
            let r = j + (rng.next_u64() as usize) % (hot - j);
            pool.swap(j, r);
            idv.push(pool[j]);
        }
    }
    let mut rows = vec![0usize; e];
    idv.iter().for_each(|&i| rows[i as usize] += 1);
    assert!(rows[..hot].iter().all(|&r| r > 2 * 32), "rows per hot expert {rows:?}");
    let ids = HostTensor::from_i32(&[t, K], &idv);
    let want = reference(&x, &idv, &ws.to_f32(), &ex, DType::Bf16);
    let got = run(&be, &x, &ids, &ws, &ex, 0).unwrap();
    grade(&got, &want, &format!("T={t} skewed (rows/expert {}..{})", rows[..hot].iter().min().unwrap(), rows[..hot].iter().max().unwrap()));
}

/// Block-scale tensors at odd byte offsets (gate_up at 5, down at 7: 4-byte skips 1 and 3; weights at 8 mod 16) are bitwise
/// the aligned binding on both paths, and the f32 output (the other `MoeExperts` dtype) matches the reference.
#[test]
fn moe_experts_odd_scale_offsets_and_f32_output() {
    let be = common::backend();
    let mut rng = SplitMix64::new(0x35a3d);
    let e = 16usize;
    let ex = random_experts(&mut rng, e, H, I);
    for t in [2usize, 40] {
        let x = HostTensor::from_f32_as(DType::Bf16, &[t, H], &rng.normal_vec(t * H, 1.0));
        let (ids, ws) = random_routing(&mut rng, t, e, K);
        let path = if t * K <= 16 { "decode" } else { "grouped" };
        let want = reference(&x, &ids.to_i32().unwrap(), &ws.to_f32(), &ex, DType::Bf16);
        let got = run_with(&be, &x, &ids, &ws, &ex, 0, 0, DType::Bf16).unwrap();
        grade(&got, &want, &format!("T={t} ({path}) aligned"));
        let odd = run_with(&be, &x, &ids, &ws, &ex, 8, 5, DType::Bf16).unwrap();
        assert!(got.iter().zip(&odd).all(|(a, b)| a.to_bits() == b.to_bits()), "T={t} ({path}): odd scale offsets differ");
        let want32 = reference(&x, &ids.to_i32().unwrap(), &ws.to_f32(), &ex, DType::F32);
        let got32 = run_with(&be, &x, &ids, &ws, &ex, 0, 0, DType::F32).unwrap();
        grade(&got32, &want32, &format!("T={t} ({path}) f32 out"));
        let odd32 = run_with(&be, &x, &ids, &ws, &ex, 8, 5, DType::F32).unwrap();
        assert!(got32.iter().zip(&odd32).all(|(a, b)| a.to_bits() == b.to_bits()), "T={t} ({path}): f32 out, odd scale offsets differ");
    }
}

/// `moe_topk_bf16` (bf16 and f32 logits) against `reference::moe_topk`: ids exactly (the logits are drawn from six values so
/// every row is full of ties; the lowest index must win), weights within one rounding of the logits dtype.
#[test]
fn moe_topk_matches_the_reference_with_ties() {
    let be = common::backend();
    let mut rng = SplitMix64::new(0x70b4);
    // (T, E, K): 35b decode / prefill, a K = E row, the E and K limits, T off the 4-token group
    for (t, e, k) in [(1usize, 256usize, 8usize), (7, 256, 8), (33, 256, 8), (5, 16, 16), (3, 512, 32), (6, 64, 1)] {
        for dt in [DType::Bf16, DType::F32] {
            for norm in [false, true] {
                let vals: Vec<f32> = (0..t * e).map(|_| (rng.next_u64() % 6) as f32 * 0.5 - 1.0).collect();
                let lg = HostTensor::from_f32_as(dt, &[t, e], &vals);
                let (ww, wi) = intarsia_core::quant::reference::moe_topk(&lg.to_f32(), t, e, k, norm, dt);
                let (wo, io) = (put(&be, &HostTensor::zeros(DType::F32, &[t, k]), 0), put(&be, &HostTensor::zeros(DType::I32, &[t, k]), 0));
                let call = OpCall::new(Op::MoeTopk, vec![put(&be, &lg, 0)], vec![wo.clone(), io.clone()], OpAttrs { top_k: k, norm_topk: norm, ..OpAttrs::default() });
                call.check().unwrap();
                be.launch("moe_topk_bf16", &call, 0).unwrap();
                be.synchronize(0).unwrap();
                let (mut wb, mut ib) = (vec![0u8; wo.nbytes()], vec![0u8; io.nbytes()]);
                be.download(&wo.buf, 0, &mut wb, 0).unwrap();
                be.download(&io.buf, 0, &mut ib, 0).unwrap();
                let what = format!("T={t} E={e} K={k} {dt} norm={norm}");
                let gi = HostTensor::new(DType::I32, vec![t, k], ib).unwrap().to_i32().unwrap();
                assert_eq!(gi, wi, "{what}: ids");
                let gw = HostTensor::new(DType::F32, vec![t, k], wb).unwrap().to_f32();
                // one rounding step of the logits dtype (bf16: 2^-8 relative), f32: the exp() of the shader vs f64
                let rel = if dt == DType::Bf16 { 8e-3 } else { 1e-5 };
                for (g, w) in gw.iter().zip(&ww) {
                    assert!((g - w).abs() <= rel * w.abs() + 1e-7, "{what}: weight {g} vs {w}");
                }
            }
        }
    }
}

/// `moe_combine_bf16` against `reference::moe_combine`; the f32 output is refused (the kernel is bf16 in/out only).
#[test]
fn moe_combine_matches_the_reference_and_refuses_f32_output() {
    let be = common::backend();
    let mut rng = SplitMix64::new(0xc0b1);
    for (t, h) in [(1usize, 2048usize), (5, 2048), (300, 2048), (3, 17)] {
        let routed = HostTensor::from_f32_as(DType::Bf16, &[t, h], &rng.normal_vec(t * h, 1.0));
        let shared = HostTensor::from_f32_as(DType::Bf16, &[t, h], &rng.normal_vec(t * h, 1.0));
        let gate = HostTensor::from_f32_as(DType::Bf16, &[t, 1], &rng.normal_vec(t, 2.0));
        let want = intarsia_core::quant::reference::moe_combine(&routed.to_f32(), &shared.to_f32(), &gate.to_f32(), t, h, DType::Bf16);
        let mk = |odt: DType, ooff: usize| {
            let out = put(&be, &HostTensor::zeros(odt, &[t, h]), ooff);
            let call = OpCall::new(Op::MoeCombine, vec![put(&be, &routed, 0), put(&be, &shared, 2), put(&be, &gate, 6)], vec![out.clone()], OpAttrs::default());
            (call, out)
        };
        let (call, out) = mk(DType::Bf16, 2);
        call.check().unwrap();
        be.launch("moe_combine_bf16", &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut data = vec![0u8; out.nbytes()];
        be.download(&out.buf, out.offset, &mut data, 0).unwrap();
        let got = HostTensor::new(DType::Bf16, vec![t, h], data).unwrap().to_f32();
        grade(&got, &want, &format!("combine T={t} H={h}"));
        let (call, _) = mk(DType::F32, 0);
        if call.check().is_ok() {
            let err = be.launch("moe_combine_bf16", &call, 0).unwrap_err();
            assert!(err.to_string().contains("expected bf16"), "{err}");
        }
    }
}
