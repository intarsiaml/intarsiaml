//! `qk_norm_rope_kv_bf16` (Op::QkNormRopeKv, kernels/hip/elementwise.hip) must be bitwise the unfused chain the model
//! runs without it: `qk_norm_bf16` on q and on k, `rope_bf16` on both, `kv_append_bf16` of the roped k and of v. Compared:
//! the normalised and roped q, and both whole caches (rows before and after the appended ones must be untouched). Cases:
//! the 35B's attention (16 / 2 heads of 256, a quarter rotary, zero-centred norm weights), the 8B's (32 / 8 of 128, full
//! rotary), a GQA group of one and of eight, a head of 32, decode, verify and prefill-chunk row counts, positions near
//! the cache end, bf16 and fp8 caches, the position on the host and on the device.
#![cfg(ik_hip)]

use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, TensorRef};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn put(be: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    be.zero(&buf, 0).unwrap();
    be.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn get(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    let mut d = vec![0u8; t.nbytes()];
    be.synchronize(0).unwrap();
    be.download(&t.buf, t.offset, &mut d, 0).unwrap();
    d
}

fn launch(be: &dyn Backend, kernel: &str, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: &OpAttrs, pos: Option<&TensorRef>) {
    let mut call = OpCall::new(op, inputs, outputs, attrs.clone());
    call.pos = pos.cloned();
    call.check().unwrap();
    be.launch(kernel, &call, 0).unwrap();
}

struct Case {
    q: HostTensor,
    k: HostTensor,
    v: HostTensor,
    qw: HostTensor,
    kw: HostTensor,
    kc: HostTensor,
    vc: HostTensor,
    attrs: OpAttrs,
}

#[allow(clippy::too_many_arguments)]
fn make(rng: &mut SplitMix64, t: usize, n: usize, nkv: usize, d: usize, rot: usize, woff: f32, p: usize, cache: DType) -> Case {
    let cap = p + t + 3;
    let r = |rng: &mut SplitMix64, shape: &[usize], s: f32| HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), s));
    let w = |rng: &mut SplitMix64| {
        // zero-centred weights (woff 1) are small around 0, plain ones around 1
        let v: Vec<f32> = (0..d).map(|_| (1.0 - woff) + 0.25 * rng.normal()).collect();
        HostTensor::from_f32_as(DType::Bf16, &[d], &v)
    };
    let cache_init = |rng: &mut SplitMix64| HostTensor::from_f32_as(cache, &[nkv, cap, d], &rng.normal_vec(nkv * cap * d, 1.0));
    Case {
        q: r(rng, &[t, n, d], 2.0),
        k: r(rng, &[t, nkv, d], 2.0),
        v: r(rng, &[t, nkv, d], 1.0),
        qw: w(rng),
        kw: w(rng),
        kc: cache_init(rng),
        vc: cache_init(rng),
        attrs: OpAttrs {
            eps: 1e-6,
            w_offset: woff,
            pos0: p,
            theta: if d == 256 { 1e7 } else { 1e6 },
            head_dim: d,
            rotary_dim: rot,
            n_heads: n,
            n_kv_heads: nkv,
            kv_cap: cap,
            ..Default::default()
        },
    }
}

/// (q out, k cache, v cache) of the six unfused launches.
fn unfused(be: &dyn Backend, c: &Case, dev: Option<&TensorRef>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (q, k, v, qw, kw) = (put(be, &c.q), put(be, &c.k), put(be, &c.v), put(be, &c.qw), put(be, &c.kw));
    let (kc, vc) = (put(be, &c.kc), put(be, &c.vc));
    let qn = put(be, &HostTensor::zeros(DType::Bf16, &c.q.shape));
    let qr = put(be, &HostTensor::zeros(DType::Bf16, &c.q.shape));
    let kn = put(be, &HostTensor::zeros(DType::Bf16, &c.k.shape));
    let kr = put(be, &HostTensor::zeros(DType::Bf16, &c.k.shape));
    let a = &c.attrs;
    launch(be, "qk_norm_bf16", Op::QkNorm, vec![q, qw], vec![qn.clone()], a, None);
    launch(be, "rope_bf16", Op::Rope, vec![qn], vec![qr.clone()], a, dev);
    launch(be, "qk_norm_bf16", Op::QkNorm, vec![k, kw], vec![kn.clone()], a, None);
    launch(be, "rope_bf16", Op::Rope, vec![kn], vec![kr.clone()], a, dev);
    launch(be, "kv_append_bf16", Op::KvAppend, vec![kr], vec![kc.clone()], a, dev);
    launch(be, "kv_append_bf16", Op::KvAppend, vec![v], vec![vc.clone()], a, dev);
    (get(be, &qr), get(be, &kc), get(be, &vc))
}

fn fused(be: &dyn Backend, c: &Case, dev: Option<&TensorRef>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (q, k, v, qw, kw) = (put(be, &c.q), put(be, &c.k), put(be, &c.v), put(be, &c.qw), put(be, &c.kw));
    let (kc, vc) = (put(be, &c.kc), put(be, &c.vc));
    let qo = put(be, &HostTensor::zeros(DType::Bf16, &c.q.shape));
    launch(be, "qk_norm_rope_kv_bf16", Op::QkNormRopeKv, vec![q, k, v, qw, kw], vec![qo.clone(), kc.clone(), vc.clone()], &c.attrs, dev);
    (get(be, &qo), get(be, &kc), get(be, &vc))
}

#[test]
fn qk_norm_rope_kv_is_bitwise_the_unfused_chain() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(31);
    // (heads, kv heads, head_dim, rotary_dim (0 = all), w_offset)
    let dims = [(16usize, 2usize, 256usize, 64usize, 1.0f32), (32, 8, 128, 0, 0.0), (4, 2, 64, 16, 1.0), (8, 8, 64, 0, 0.0), (24, 3, 32, 8, 0.0)];
    let mut n_cases = 0;
    for &(n, nkv, d, rot, woff) in &dims {
        for t in [1usize, 2, 5, 17] {
            for p in [0usize, 37, 4090] {
                for cache in [DType::Bf16, DType::Fp8E4m3] {
                    let c = make(&mut rng, t, n, nkv, d, rot, woff, p, cache);
                    let what = format!("t={t} n={n} nkv={nkv} d={d} rot={rot} woff={woff} p={p} cache={cache}");
                    let pos = put(&*be, &HostTensor::from_i32(&[1], &[p as i32]));
                    for dev in [None, Some(&pos)] {
                        let a = unfused(&*be, &c, dev);
                        let b = fused(&*be, &c, dev);
                        let w = if dev.is_some() { "device pos" } else { "host pos" };
                        assert!(a.0 == b.0, "{what} {w}: q differs");
                        assert!(a.1 == b.1, "{what} {w}: k cache differs");
                        assert!(a.2 == b.2, "{what} {w}: v cache differs");
                        assert!(a.1 != c.kc.data && a.2 != c.vc.data, "{what}: nothing appended");
                        n_cases += 1;
                    }
                }
            }
        }
    }
    println!("qk_norm_rope_kv: {n_cases} cases bitwise");
}

/// A one-ulp f32 difference (another fma contraction, say) shows in the bf16 output only when it crosses a bf16 rounding
/// boundary, a few times per 10^5 values: so many draws of the 35B's prefill-chunk shape at positions up to 16k.
#[test]
fn qk_norm_rope_kv_many_draws_bitwise() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(97);
    let draws = 60;
    for i in 0..draws {
        let p = (rng.next_u64() % 16000) as usize;
        let cache = if i % 2 == 0 { DType::Bf16 } else { DType::Fp8E4m3 };
        let c = make(&mut rng, 17, 16, 2, 256, 64, 1.0, p, cache);
        let a = unfused(&*be, &c, None);
        let b = fused(&*be, &c, None);
        assert!(a.0 == b.0 && a.1 == b.1 && a.2 == b.2, "draw {i} p={p} cache={cache}: fused differs");
    }
    println!("qk_norm_rope_kv: {draws} draws of 17 rows bitwise");
}
