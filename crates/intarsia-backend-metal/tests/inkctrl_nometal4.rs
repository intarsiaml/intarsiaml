#![cfg(target_os = "macos")]
//! The metal4.0 fallback of the controller-fusion kernels (inkctrl.rs, lane ink-dfuse): with `INTARSIA_METAL_NO_METAL4=1`
//! `ink_route_bf16` is not registered (so `Registry::select` of an `InkRoute` call fails and the model graph keeps `Logits` +
//! `MoeRoute`) and refuses to launch. The test sets the process environment, so it is the only test in this file.

use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::registry::Registry;
use intarsia_core::{DType, Op, OpAttrs, OpCall};

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
}
