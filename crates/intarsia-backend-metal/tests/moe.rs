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

/// Run `moe_experts_bf16` with the five expert tensors at byte `woff` of their buffers; bf16 output as f32.
fn run(be: &MetalBackend, x: &HostTensor, ids: &HostTensor, ws: &HostTensor, ex: &HostExperts, woff: usize) -> intarsia_core::Result<Vec<f32>> {
    let t = x.shape[0];
    let mut inputs = vec![put(be, x, 0), put(be, ids, 0), put(be, ws, 0)];
    for w in [&ex.gate_up_w, &ex.gate_up_s, &ex.down_w, &ex.down_s] {
        inputs.push(put(be, w, woff));
    }
    inputs.push(put(be, &ex.gscales, 0));
    let out = put(be, &HostTensor::zeros(DType::Bf16, &[t, H]), 0);
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
    Ok(HostTensor::new(DType::Bf16, vec![t, H], data)?.to_f32())
}

fn reference(x: &HostTensor, ids: &[i32], ws: &[f32], ex: &HostExperts) -> Vec<f64> {
    let t = x.shape[0];
    let xf = x.to_f32();
    let gs = ex.gscales.to_f32();
    let er = ExpertsRef {
        e: E,
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
                    let r = moe_experts_token(er, &xf[ti * H..(ti + 1) * H], &ids[ti * K..(ti + 1) * K], &ws[ti * K..(ti + 1) * K], DType::Bf16, DType::Bf16);
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
        let want = reference(&x, &idv, &ws.to_f32(), &ex);
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
