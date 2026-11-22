//! Lane ink-m-moe: µs per call and effective GB/s of the Inkling-2-Mini expert kernels at the real shape (H 2304, I 1152,
//! 128 experts, top-8): NVFP4 situ (`moe_experts_situ_bf16`) and MXFP8 situ (`moe_experts_mxfp8_bf16`), decode rows 1 / 4 /
//! 16 under each decode policy, and a 2048-row prefill chunk. Random routing (distinct top-8 per token, uniform over the
//! experts): the expected union is 8 / 29.1 / 82.4 experts at 1 / 4 / 16 rows; the bytes counted are the call's actual
//! union x the expert bytes (NVFP4 4.479 MB, MXFP8 8.211 MB). Several layer copies of the weights rotate (> SLC) and every
//! call has its own routing; wall time of a synced batch (host encoding overlaps the GPU).
//!
//!   cargo run --release -p intarsia-backend-metal --example inkmoe_bench [-- nv|mx] [rows ...]
//!
//! Run it inside a fleetq slot (`fleetq run --host endor --quiet gpu,cpu --mem 8 ...`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("inkmoe_bench needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::bench::Rng;
    use intarsia_backend_metal::inkmoe::{set_dnf_threads, set_policy, set_probe, Policy};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
    use std::time::Instant;

    const E: usize = 128;
    const K: usize = 8;
    const H: usize = 2304;
    const I: usize = 1152;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fmts: Vec<&str> = if args.iter().any(|a| a == "nv" || a == "mx") {
        args.iter().filter(|a| *a == "nv" || *a == "mx").map(|s| s.as_str()).collect()
    } else {
        vec!["nv", "mx"]
    };
    let rows: Vec<usize> = {
        let r: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
        if r.is_empty() {
            vec![1, 4, 16, 2048]
        } else {
            r
        }
    };
    if let Some(n) = std::env::var("INKMOE_DNF_THREADS").ok().and_then(|v| v.parse().ok()) {
        set_dnf_threads(n);
    }
    let be = MetalBackend::new().expect("metal backend");
    let ks = be.kernels();
    let t = |b: &Buffer, dt: DType, shape: &[usize]| TensorRef {
        buf: b.clone(),
        offset: 0,
        dtype: dt,
        shape: shape.to_vec(),
    };
    let fill = |bytes: usize, seed: u64, f: &dyn Fn(u64) -> u8| -> Buffer {
        let mut r = Rng(seed | 1);
        let v: Vec<u8> = (0..bytes).map(|_| f(r.next_u64())).collect();
        let b = be.alloc(bytes, MemKind::Device).unwrap();
        be.upload(&b, 0, &v, 0).unwrap();
        b
    };
    // e2m1 nibbles (no NaN issue), e4m3 scales around 2^-4, e4m3 weights with exponents 4..=10, E8M0 scales 116..=119
    let nib = |r: u64| (r >> 40) as u8;
    let nvsc = |r: u64| 0x28 + ((r >> 40) % 8) as u8;
    let e4 = |r: u64| ((r >> 33) & 0x80) as u8 | ((4 + (r >> 41) % 7) as u8) << 3 | ((r >> 50) & 7) as u8;
    let e8 = |r: u64| 116 + ((r >> 40) % 4) as u8;
    for fmt in fmts {
        let mx = fmt == "mx";
        let layers = if mx { 3 } else { 5 };
        let ebytes = if mx { 8_211_456usize } else { 4_478_976 };
        println!(
            "== {} experts, {layers} layer copies of {:.0} MB",
            if mx { "MXFP8" } else { "NVFP4" },
            (E * ebytes) as f64 / 1e6
        );
        let w: Vec<Vec<TensorRef>> = (0..layers)
            .map(|l| {
                let s = 1000 * l as u64;
                if mx {
                    vec![
                        t(&fill(E * 2 * I * H, s + 1, &e4), DType::Fp8E4m3, &[E, 2 * I, H]),
                        t(&fill(E * 2 * I * H / 32, s + 2, &e8), DType::U8, &[E, 2 * I, H / 32]),
                        t(&fill(E * H * I, s + 3, &e4), DType::Fp8E4m3, &[E, H, I]),
                        t(&fill(E * H * I / 32, s + 4, &e8), DType::U8, &[E, H, I / 32]),
                    ]
                } else {
                    let gs = be.alloc(E * 12, MemKind::Device).unwrap();
                    be.upload(
                        &gs,
                        0,
                        &vec![0.05f32; E * 3].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
                        0,
                    )
                    .unwrap();
                    vec![
                        t(&fill(E * 2 * I * H / 2, s + 1, &nib), DType::U8, &[E, 2 * I, H / 2]),
                        t(&fill(E * 2 * I * H / 16, s + 2, &nvsc), DType::Fp8E4m3, &[E, 2 * I, H / 16]),
                        t(&fill(E * H * I / 2, s + 3, &nib), DType::U8, &[E, H, I / 2]),
                        t(&fill(E * H * I / 16, s + 4, &nvsc), DType::Fp8E4m3, &[E, H, I / 16]),
                        t(&gs, DType::F32, &[E, 3]),
                    ]
                }
            })
            .collect();
        let kname = if mx { "moe_experts_mxfp8_bf16" } else { "moe_experts_situ_bf16" };
        let kern = ks.iter().find(|k| k.name == kname).unwrap().clone();
        for &tr in &rows {
            let ncalls = if tr > 64 { 6 } else { 48 };
            let mut rng = intarsia_core::rng::SplitMix64::new(0x1a4b ^ tr as u64);
            let xb = be.alloc(tr * H * 2, MemKind::Device).unwrap();
            let xv: Vec<f32> = rng.normal_vec(tr * H, 1.0);
            be.upload(&xb, 0, &HostTensor::from_f32_as(DType::Bf16, &[tr, H], &xv).data, 0).unwrap();
            let out = be.alloc(tr * H * 2, MemKind::Device).unwrap();
            let mut unions = Vec::new();
            let calls: Vec<OpCall> = (0..ncalls)
                .map(|c| {
                    let (ids, ws) = intarsia_core::quant::cases::random_routing(&mut rng, tr, E, K);
                    let mut seen = [false; E];
                    for &id in &ids.to_i32().unwrap() {
                        seen[id as usize] = true;
                    }
                    unions.push(seen.iter().filter(|&&s| s).count());
                    let ib = be.alloc(tr * K * 4, MemKind::Device).unwrap();
                    be.upload(&ib, 0, &ids.data, 0).unwrap();
                    let wb = be.alloc(tr * K * 4, MemKind::Device).unwrap();
                    let w8: Vec<f32> = ws.to_f32().iter().map(|v| v * 8.0).collect();
                    be.upload(&wb, 0, &HostTensor::from_f32(&[tr, K], &w8).data, 0).unwrap();
                    let mut ins =
                        vec![t(&xb, DType::Bf16, &[tr, H]), t(&ib, DType::I32, &[tr, K]), t(&wb, DType::F32, &[tr, K])];
                    ins.extend(w[c % layers].iter().cloned());
                    let mut a = OpAttrs { top_k: K, ..OpAttrs::default() };
                    (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
                    let op = if mx { Op::MoeExpertsMxfp8 } else { Op::MoeExperts };
                    let mut call = OpCall::new(op, ins, vec![t(&out, DType::Bf16, &[tr, H])], a);
                    call.check().unwrap();
                    let need = kern.scratch_bytes(&ShapeInfo::from_call(&call));
                    call.scratch =
                        Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
                    call
                })
                .collect();
            let mean_union = unions.iter().sum::<usize>() as f64 / unions.len() as f64;
            let pols: &[Policy] = if tr == 1 {
                &[Policy::Auto, Policy::Union, Policy::Pair, Policy::Fused]
            } else if tr <= 16 {
                &[Policy::Auto, Policy::Union, Policy::Pair, Policy::Fused, Policy::UnionFused]
            } else {
                &[Policy::Auto]
            };
            let probes: &[u8] = if std::env::var("INKMOE_PROBE").is_ok() && tr <= 16 { &[0, 1, 2] } else { &[0] };
            for &pol in pols {
                for &probe in probes {
                    set_policy(pol);
                    set_probe(probe);
                    let run = |reps: usize| {
                        for r in 0..reps {
                            be.launch(kname, &calls[r % ncalls], 0).unwrap();
                        }
                        be.synchronize(0).unwrap();
                    };
                    run(ncalls.min(8));
                    let reps = if tr > 64 { 12 } else { 240 };
                    let mut v = Vec::new();
                    for _ in 0..5 {
                        let t0 = Instant::now();
                        run(reps);
                        v.push(t0.elapsed().as_secs_f64() / reps as f64 * 1e6);
                    }
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let us = v[2];
                    let gbs = mean_union * ebytes as f64 / (us * 1e-6) / 1e9;
                    let what = ["all", "gate+up", "down"][probe as usize];
                    let gbs = gbs * [1.0, 0.6667, 0.3333][probe as usize];
                    println!(
                    "{fmt} T={tr:5} {:12} {what:8} {us:9.1} us/call (min {:8.1})  union {mean_union:5.1} experts  {gbs:6.1} GB/s",
                    format!("{pol:?}"),
                    v[0]
                );
                }
            }
            set_probe(0);
            set_policy(Policy::Auto);
        }
    }
}
