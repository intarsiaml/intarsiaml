//! Lane ink-prefill: µs per call of the small Inkling-2-Mini prefill ops at a 2048-row chunk that the kernel lanes' benches do
//! not cover: the router (`Logits` [T, 2304] x [128, 2304] -> f32, the Matmul the graph turns into `Logits` on Metal) and
//! `MoeRoute` (sigmoid, E 128, top-8). Wall time of a synced batch, several operand copies rotating.
//!
//!   cargo run --release -p intarsia-backend-metal --example ink_prefill_bench [-- rows ...]
//!
//! Run it inside a fleetq slot (`--quiet gpu,cpu`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ink_prefill_bench needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, MemKind, TensorRef};
    use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall};
    use std::time::Instant;

    const D: usize = 2304;
    const E: usize = 128;
    const K: usize = 8;
    let rows: Vec<usize> = {
        let r: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
        if r.is_empty() {
            vec![2048]
        } else {
            r
        }
    };
    let be = MetalBackend::new().expect("metal backend");
    let names: Vec<String> = be.kernels().iter().map(|k| k.name.to_string()).collect();
    let mut rng = intarsia_core::rng::SplitMix64::new(7);
    let mut up = |dt: DType, shape: &[usize], scale: f32| -> TensorRef {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = rng.normal_vec(n, scale);
        let h = HostTensor::from_f32_as(dt, shape, &v);
        let b = be.alloc(h.data.len().max(16), MemKind::Device).unwrap();
        be.upload(&b, 0, &h.data, 0).unwrap();
        TensorRef { buf: b, offset: 0, dtype: dt, shape: shape.to_vec() }
    };
    let time = |kname: &str, calls: &[OpCall]| -> f64 {
        let run = |reps: usize| {
            for r in 0..reps {
                be.launch(kname, &calls[r % calls.len()], 0).unwrap();
            }
            be.synchronize(0).unwrap();
        };
        run(calls.len());
        let mut v = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            run(48);
            v.push(t0.elapsed().as_secs_f64() / 48.0 * 1e6);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[2]
    };
    for &t in &rows {
        let ws: Vec<TensorRef> = (0..8).map(|_| up(DType::Bf16, &[E, D], 0.02)).collect();
        let x = up(DType::Bf16, &[t, D], 1.0);
        let out = up(DType::F32, &[t, E], 0.0);
        for kname in ["logits_bf16", "logits_bf16_gemv", "logits_bf16_gemv_rows"] {
            if !names.iter().any(|n| n == kname) {
                continue;
            }
            let calls: Vec<OpCall> = ws
                .iter()
                .map(|w| OpCall::new(Op::Logits, vec![x.clone(), w.clone()], vec![out.clone()], OpAttrs::default()))
                .collect();
            if calls[0].check().is_err() {
                continue;
            }
            let us = time(kname, &calls);
            let fl = 2.0 * (t * E * D) as f64;
            println!("router {kname:24} T={t:5} {us:9.1} us/call  {:6.2} TFLOP/s", fl / (us * 1e-6) / 1e12);
        }
        let bias = up(DType::F32, &[E], 0.1);
        let logits = up(DType::F32, &[t, E], 1.0);
        let rw = up(DType::F32, &[t, K], 0.0);
        let ri = TensorRef { dtype: DType::I32, ..up(DType::F32, &[t, K], 0.0) };
        let mut ra = OpAttrs { top_k: K, scale: 2.5, norm_topk: true, temperature: 1.0, ..OpAttrs::default() };
        ra.dsv.score_func = intarsia_core::dsv41::ops::ScoreFunc::Sigmoid;
        let call = OpCall::new(Op::MoeRoute, vec![logits, bias], vec![rw, ri], ra);
        call.check().unwrap();
        let us = time("moe_route_f32", &[call]);
        println!("route  moe_route_f32            T={t:5} {us:9.1} us/call");
    }
}
