#![cfg(target_os = "macos")]
//! The metal4.0 fallback of the controller-fusion kernels (inkctrl.rs, lane ink-dfuse): with `INTARSIA_METAL_NO_METAL4=1`
//! `ink_route_bf16` is not registered (so `Registry::select` of an `InkRoute` call fails and the model graph keeps `Logits` +
//! `MoeRoute`) and refuses to launch, and the blend tail stays off even when forced on (LoopBlend gives the three-dispatch
//! bits). The test sets the process environment, so it is the only test in this file.

use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::registry::Registry;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

fn dev(be: &MetalBackend, dt: DType, shape: &[usize], bytes: &[u8]) -> TensorRef {
    let need = shape.iter().product::<usize>() * dt.size_bytes();
    let b = be.alloc(need.max(bytes.len()).max(16), MemKind::Device).unwrap();
    if !bytes.is_empty() {
        be.upload(&b, 0, bytes, 0).unwrap();
    }
    TensorRef { buf: b, offset: 0, dtype: dt, shape: shape.to_vec() }
}

fn bf(n: usize, scale: f32, rng: &mut Rng) -> Vec<u8> {
    (0..n).flat_map(|_| f32_to_bf16(rng.unit() * scale).to_le_bytes()).collect()
}

fn f32s(n: usize, scale: f32, rng: &mut Rng) -> Vec<u8> {
    (0..n).flat_map(|_| (rng.unit() * scale).to_le_bytes()).collect()
}

#[test]
fn without_metal4_the_fused_paths_fall_back() {
    std::env::set_var("INTARSIA_METAL_NO_METAL4", "1");
    let be = MetalBackend::new().expect("Metal backend");
    assert!(!be.has_metal4());
    assert!(!be.kernels().iter().any(|k| k.op == Op::InkRoute), "ink_route_bf16 registered without the metal4.0 library");
    let mut rng = Rng(0xfa11);
    let (t, d, e, k) = (1usize, 256usize, 128usize, 8usize);
    let mut a = OpAttrs { top_k: k, scale: 8.0, norm_topk: true, temperature: 1.0, ..OpAttrs::default() };
    a.dsv.score_func = intarsia_core::dsv41::ops::ScoreFunc::Sigmoid;
    let call = OpCall::new(
        Op::InkRoute,
        vec![dev(&be, DType::Bf16, &[t, d], &bf(t * d, 1.0, &mut rng)), dev(&be, DType::Bf16, &[e, d], &bf(e * d, 0.06, &mut rng)), dev(&be, DType::F32, &[e], &f32s(e, 0.1, &mut rng))],
        vec![dev(&be, DType::F32, &[t, k], &[]), dev(&be, DType::I32, &[t, k], &[]), dev(&be, DType::F32, &[t, e], &[])],
        a,
    );
    call.check().unwrap();
    assert!(Registry::new("metal", &be.info(), be.kernels()).select(&call).is_err(), "an InkRoute call must find no kernel");
    assert!(be.launch(intarsia_backend_metal::inkctrl::INK_ROUTE_BF16, &call, 0).is_err(), "ink_route_bf16 must refuse to launch");
    // LoopBlend with the tail forced on == forced off
    let (g, s, r) = (64usize, 3usize, 2usize);
    let ins: Vec<(DType, Vec<usize>, Vec<u8>)> = vec![
        (DType::F32, vec![1, r, d], f32s(r * d, 1.0, &mut rng)),
        (DType::F32, vec![s, r, d], f32s(s * r * d, 1.0, &mut rng)),
        (DType::Bf16, vec![s, g, 2 * d], bf(s * g * 2 * d, 0.03, &mut rng)),
        (DType::Bf16, vec![s, d, g], bf(s * d * g, 0.25, &mut rng)),
        (DType::Bf16, vec![s, d, g], bf(s * d * g, 0.25, &mut rng)),
        (DType::Bf16, vec![s, d], bf(s * d, 1.0, &mut rng)),
    ];
    let run = |tail: bool| -> Vec<u8> {
        intarsia_backend_metal::inkctrl::set_blend_tail(Some(tail));
        let it: Vec<TensorRef> = ins.iter().map(|(dt, sh, b)| dev(&be, *dt, sh, b)).collect();
        let out = dev(&be, DType::Bf16, &[s, r, d], &vec![0u8; s * r * d * 2]);
        let mut c = OpCall::new(Op::LoopBlend, it, vec![out.clone()], OpAttrs { eps: 1e-6, ..OpAttrs::default() });
        c.exact_rows = true;
        c.check().unwrap();
        let n = intarsia_backend_metal::inkmix::blend_scratch(&ShapeInfo::from_call(&c));
        c.scratch = Some(dev(&be, DType::U8, &[n], &[]));
        be.launch(intarsia_backend_metal::inkmix::LOOP_BLEND_BF16, &c, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut v = vec![0u8; s * r * d * 2];
        be.download(&out.buf, 0, &mut v, 0).unwrap();
        v
    };
    let (on, off) = (run(true), run(false));
    intarsia_backend_metal::inkctrl::set_blend_tail(None);
    assert_eq!(on, off, "the blend tail ran without the metal4.0 library");
}
