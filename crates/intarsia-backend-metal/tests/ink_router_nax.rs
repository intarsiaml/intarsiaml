#![cfg(target_os = "macos")]
//! Lane ink-prefill: `logits_bf16_nax` (Op::Logits for M > 16 on the accelerators, the Inkling-2 router of prefill chunks)
//! against an f64 reference of the bf16 operands, at the Logits f32 budget (max_abs 1e-3, rel_l2 2e-4, kernels.rs `F32_TOL`),
//! and that the registry selects it for M > 16 and `logits_bf16` / the GEMV forms below. Router shape (N 128, K 2304) at
//! 17 / 40 / 300 / 2048 rows, plus an N / M edge shape (N 200, K 1000, M 77) and a 64 x 64-tile shape (N 1024).

use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};

fn up(be: &MetalBackend, h: &HostTensor) -> TensorRef {
    let b = be.alloc(h.data.len().max(16), MemKind::Device).unwrap();
    be.upload(&b, 0, &h.data, 0).unwrap();
    TensorRef { buf: b, offset: 0, dtype: h.dtype, shape: h.shape.clone() }
}

#[test]
fn logits_nax_matches_f64_and_is_selected_for_prefill_rows() {
    let be = MetalBackend::new().expect("metal");
    if !be.has_metal4() {
        eprintln!("no metal4.0 library: skipped");
        return;
    }
    let ks = be.kernels();
    let nax = ks.iter().find(|k| k.name == "logits_bf16_nax").expect("registered");
    let mut rng = intarsia_core::rng::SplitMix64::new(0x40e7);
    for (m, n, k) in [(17usize, 128usize, 2304usize), (40, 128, 2304), (300, 128, 2304), (2048, 128, 2304), (77, 200, 1000), (64, 1024, 512)] {
        let xh = HostTensor::from_f32_as(DType::Bf16, &[m, k], &rng.normal_vec(m * k, 1.0));
        let wh = HostTensor::from_f32_as(DType::Bf16, &[n, k], &rng.normal_vec(n * k, 1.0 / (k as f32).sqrt()));
        let (x, w) = (up(&be, &xh), up(&be, &wh));
        let y = up(&be, &HostTensor::from_f32(&[m, n], &vec![f32::NAN; m * n]));
        let call = OpCall::new(Op::Logits, vec![x, w], vec![y.clone()], OpAttrs::default());
        call.check().unwrap();
        let s = ShapeInfo::from_call(&call);
        assert!(nax.accepts(&s), "M={m}: logits_bf16_nax must accept");
        let pick = be.kernels().into_iter().filter(|k| k.op == Op::Logits && k.accepts(&s)).max_by_key(|k| k.priority).unwrap();
        assert_eq!(pick.name, "logits_bf16_nax", "M={m}: the highest-priority Logits kernel");
        be.launch("logits_bf16_nax", &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut raw = vec![0u8; m * n * 4];
        be.download(&y.buf, 0, &mut raw, 0).unwrap();
        let got: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let (xf, wf) = (xh.to_f32(), wh.to_f32());
        let (mut max_abs, mut d2, mut r2) = (0f64, 0f64, 0f64);
        for i in 0..m {
            for j in 0..n {
                let r: f64 = (0..k).map(|l| xf[i * k + l] as f64 * wf[j * k + l] as f64).sum();
                let d = (got[i * n + j] as f64 - r).abs();
                assert!(d.is_finite(), "M={m} N={n}: non-finite at ({i},{j})");
                max_abs = max_abs.max(d);
                d2 += d * d;
                r2 += r * r;
            }
        }
        let rel = (d2 / r2).sqrt();
        eprintln!("M={m:5} N={n:5} K={k:5} max_abs {max_abs:.3e} rel_l2 {rel:.3e}");
        assert!(max_abs <= 1e-3 && rel <= 2e-4, "M={m} N={n} K={k}: max_abs {max_abs:.3e} rel_l2 {rel:.3e}");
    }
    // at M <= 16 it must not be picked
    let x = up(&be, &HostTensor::from_f32_as(DType::Bf16, &[16, 2304], &vec![0.0; 16 * 2304]));
    let w = up(&be, &HostTensor::from_f32_as(DType::Bf16, &[128, 2304], &vec![0.0; 128 * 2304]));
    let y = up(&be, &HostTensor::from_f32(&[16, 128], &vec![0.0; 16 * 128]));
    let call = OpCall::new(Op::Logits, vec![x, w], vec![y], OpAttrs::default());
    assert!(!nax.accepts(&ShapeInfo::from_call(&call)), "M=16 stays off the accelerator kernel");
}
