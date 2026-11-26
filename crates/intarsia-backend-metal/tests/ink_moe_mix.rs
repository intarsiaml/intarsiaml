#![cfg(target_os = "macos")]
//! Lane ink-moepf2: the mixed-tile prefill experts (`INTARSIA_INK_MOE_MIX`, inkmoe.rs `Mix`: an expert's last rows on a 16-
//! or 32-row tile instead of a zero-padded 64-row tile) must give the bits of the uniform 64-row form. Real shape (H 2304,
//! I 1152, 128 experts, top-8, SiTU), NVFP4 and MXFP8, on a random routing at 2048 and 600 rows and on a routing with every
//! remainder class (experts of 0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, ... rows).

use intarsia_backend_metal::bench::Rng;
use intarsia_backend_metal::inkmoe::{mix_served, set_mix};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};

const E: usize = 128;
const K: usize = 8;
const H: usize = 2304;
const I: usize = 1152;

fn tr(b: &Buffer, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: 0, dtype: dt, shape: shape.to_vec() }
}

fn fill(be: &MetalBackend, bytes: usize, seed: u64, f: &dyn Fn(u64) -> u8) -> Buffer {
    let mut r = Rng(seed | 1);
    let v: Vec<u8> = (0..bytes).map(|_| f(r.next_u64())).collect();
    let b = be.alloc(bytes, MemKind::Device).unwrap();
    be.upload(&b, 0, &v, 0).unwrap();
    b
}

/// Rows per expert with every remainder class, summing to t * K (each at most t).
fn class_counts(t: usize) -> Vec<usize> {
    let mut c: Vec<usize> = vec![0, 1, 2, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 72, 80, 81, 96, 97, 127, 128, 129];
    c.extend([144, 145, 160, 161, 192, 193, 208, 209, 224, 256, 257, 272, 273, 288, 0, 0, 3, 5, 11, 13, 19, 23, 29]);
    c.resize(E, 0);
    let mut left = t * K - c.iter().sum::<usize>();
    // the rest over the tail experts, in uneven shares
    for e in (c.len() - 80..E).cycle() {
        if left == 0 {
            break;
        }
        let add = left.min(37 + (e * 53) % 91).min(t - c[e]);
        c[e] += add;
        left -= add;
    }
    c
}

/// Top-8 ids with `cnt` rows per expert (slot-major, so an expert never lands twice in one row).
fn ids_from_counts(cnt: &[usize], t: usize) -> Vec<i32> {
    let flat: Vec<i32> = cnt.iter().enumerate().flat_map(|(e, &c)| std::iter::repeat_n(e as i32, c)).collect();
    assert_eq!(flat.len(), t * K);
    let mut ids = vec![0i32; t * K];
    for (i, &e) in flat.iter().enumerate() {
        ids[(i % t) * K + i / t] = e;
    }
    ids
}

#[test]
fn mixed_tiles_are_bitwise_the_64_row_tiles() {
    let be = MetalBackend::new().expect("metal");
    if !be.has_metal4() {
        eprintln!("no metal4.0 library: skipped");
        return;
    }
    let nib = |r: u64| (r >> 40) as u8;
    let nvsc = |r: u64| 0x28 + ((r >> 40) % 8) as u8;
    let e4 = |r: u64| ((r >> 33) & 0x80) as u8 | ((4 + (r >> 41) % 7) as u8) << 3 | ((r >> 50) & 7) as u8;
    let e8 = |r: u64| 116 + ((r >> 40) % 4) as u8;
    let mut failures = Vec::new();
    for mx in [false, true] {
        let w: Vec<TensorRef> = if mx {
            vec![
                tr(&fill(&be, E * 2 * I * H, 1, &e4), DType::Fp8E4m3, &[E, 2 * I, H]),
                tr(&fill(&be, E * 2 * I * H / 32, 2, &e8), DType::U8, &[E, 2 * I, H / 32]),
                tr(&fill(&be, E * H * I, 3, &e4), DType::Fp8E4m3, &[E, H, I]),
                tr(&fill(&be, E * H * I / 32, 4, &e8), DType::U8, &[E, H, I / 32]),
            ]
        } else {
            let gs = be.alloc(E * 12, MemKind::Device).unwrap();
            be.upload(&gs, 0, &vec![0.05f32; E * 3].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), 0).unwrap();
            vec![
                tr(&fill(&be, E * 2 * I * H / 2, 1, &nib), DType::U8, &[E, 2 * I, H / 2]),
                tr(&fill(&be, E * 2 * I * H / 16, 2, &nvsc), DType::Fp8E4m3, &[E, 2 * I, H / 16]),
                tr(&fill(&be, E * H * I / 2, 3, &nib), DType::U8, &[E, H, I / 2]),
                tr(&fill(&be, E * H * I / 16, 4, &nvsc), DType::Fp8E4m3, &[E, H, I / 16]),
                tr(&gs, DType::F32, &[E, 3]),
            ]
        };
        let kname = if mx { "moe_experts_mxfp8_bf16" } else { "moe_experts_situ_bf16" };
        let kern = be.kernels().into_iter().find(|k| k.name == kname).unwrap();
        for (case, t) in [("random", 2048usize), ("classes", 2048), ("random", 600)] {
            let mut rng = intarsia_core::rng::SplitMix64::new(0x3c0e ^ t as u64);
            let xb = be.alloc(t * H * 2, MemKind::Device).unwrap();
            be.upload(&xb, 0, &HostTensor::from_f32_as(DType::Bf16, &[t, H], &rng.normal_vec(t * H, 1.0)).data, 0).unwrap();
            let (mut ids, ws) = intarsia_core::quant::cases::random_routing(&mut rng, t, E, K);
            if case == "classes" {
                ids = HostTensor::from_i32(&[t, K], &ids_from_counts(&class_counts(t), t));
            }
            let ib = be.alloc(t * K * 4, MemKind::Device).unwrap();
            be.upload(&ib, 0, &ids.data, 0).unwrap();
            let wb = be.alloc(t * K * 4, MemKind::Device).unwrap();
            let w8: Vec<f32> = ws.to_f32().iter().map(|v| v * 8.0).collect();
            be.upload(&wb, 0, &HostTensor::from_f32(&[t, K], &w8).data, 0).unwrap();
            let out = be.alloc(t * H * 2, MemKind::Device).unwrap();
            let mut ins = vec![tr(&xb, DType::Bf16, &[t, H]), tr(&ib, DType::I32, &[t, K]), tr(&wb, DType::F32, &[t, K])];
            ins.extend(w.iter().cloned());
            let mut a = OpAttrs { top_k: K, ..OpAttrs::default() };
            (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
            let op = if mx { Op::MoeExpertsMxfp8 } else { Op::MoeExperts };
            let mut call = OpCall::new(op, ins, vec![tr(&out, DType::Bf16, &[t, H])], a);
            call.check().unwrap();
            let need = kern.scratch_bytes(&ShapeInfo::from_call(&call));
            call.scratch = Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
            let run = |form: Option<&str>| -> Vec<u8> {
                set_mix(form);
                // poison the output first (bf16 0xFFFF = NaN), so a form that writes nothing cannot pass as the last result
                be.upload(&out, 0, &vec![0xFFu8; t * H * 2], 0).unwrap();
                be.launch(kname, &call, 0).unwrap();
                be.synchronize(0).unwrap();
                let mut raw = vec![0u8; t * H * 2];
                be.download(&out, 0, &mut raw, 0).unwrap();
                raw
            };
            let s0 = mix_served();
            let base = run(None);
            assert_eq!(mix_served(), s0, "the base run took the mixed form");
            assert!(base.chunks_exact(2).all(|c| c != [0xFF, 0xFF]), "the base run left poisoned elements");
            for form in ["16", "32", "16+32", "16k64", "32k64", "16+32k64"] {
                let s1 = mix_served();
                let y = run(Some(form));
                let ran = mix_served() == s1 + 1;
                let diff = y.chunks_exact(2).zip(base.chunks_exact(2)).filter(|(a, b)| a != b).count();
                let fmt = if mx { "mx" } else { "nv" };
                eprintln!("{fmt} {case:8} T={t:4} mix {form:9} differing elements {diff} of {} ran {ran}", t * H);
                if diff != 0 || !ran {
                    failures.push(format!("{fmt} {case} T={t} {form}: {diff} differ, ran {ran}"));
                }
            }
        }
    }
    set_mix(None);
    assert!(failures.is_empty(), "mixed tiles off the 64-row tiles: {failures:?}");
}
