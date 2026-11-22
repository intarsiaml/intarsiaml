//! Lane ink-cuda-mixmoe: timing of the Inkling-2-Mini CUDA kernels at the real shapes (D / H 2304, I 1152, 128 experts,
//! top-8): InkInProj ([6552 + 1056 + 1056, 2304] bf16) against the existing `matmul_bf16_gemv_rows` over the same rows; the
//! experts (NVFP4 situ, MXFP8 situ) at 1 / 4 / 16 rows with random routing (distinct top-8 per token) and a 2048-row chunk;
//! ConvMix / SconvRes / AttnResMix / LoopBlend at decode rows. Weight copies rotate (more bytes than L2) and every expert call
//! has its own routing. Time = wall time of a synced batch of back-to-back launches on one stream (median of 5), so a
//! kernel shorter than the host's launch cost reads as that cost (the decode graph removes most of it).
//!
//!   cargo run --release --features ... -p intarsia-backend-cuda --example ink_bench [-- proj|moe|mix] [rows ...]
//!
//! Run it inside a fleetq slot (`fleetq run --host spark2 --quiet gpu,cpu ...`).

use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, HostTensor, KernelImpl, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::Arc;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn t(b: &Buffer, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: 0, dtype: dt, shape: shape.to_vec() }
}

fn fill(be: &dyn Backend, bytes: usize, seed: u64, f: &dyn Fn(u64) -> u8) -> Buffer {
    let mut r = Rng(seed | 1);
    let v: Vec<u8> = (0..bytes).map(|_| f(r.next_u64())).collect();
    let b = be.alloc(bytes.max(16), MemKind::Device).unwrap();
    be.upload(&b, 0, &v, 0).unwrap();
    b
}

fn bf16_buf(be: &dyn Backend, shape: &[usize], seed: u64, scale: f32) -> Buffer {
    let n: usize = shape.iter().product();
    let mut rng = intarsia_core::rng::SplitMix64::new(seed);
    let v = rng.normal_vec(n, scale);
    let b = be.alloc((n * 2).max(16), MemKind::Device).unwrap();
    be.upload(&b, 0, &HostTensor::from_f32_as(DType::Bf16, shape, &v).data, 0).unwrap();
    b
}

fn f32_buf(be: &dyn Backend, shape: &[usize], seed: u64, scale: f32) -> Buffer {
    let n: usize = shape.iter().product();
    let mut rng = intarsia_core::rng::SplitMix64::new(seed);
    let v = rng.normal_vec(n, scale);
    let b = be.alloc((n * 4).max(16), MemKind::Device).unwrap();
    be.upload(&b, 0, &HostTensor::from_f32(shape, &v).data, 0).unwrap();
    b
}

fn with_scratch(be: &dyn Backend, k: &KernelImpl, mut c: OpCall) -> OpCall {
    c.check().unwrap();
    let need = k.scratch_bytes(&ShapeInfo::from_call(&c));
    if need > 0 {
        c.scratch = Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
    }
    c
}

/// median µs per call over 5 batches of `reps` launches (calls rotate)
fn time(be: &dyn Backend, kname: &str, calls: &[OpCall], reps: usize) -> (f64, f64) {
    let run = |n: usize| {
        for r in 0..n {
            be.launch(kname, &calls[r % calls.len()], 0).unwrap();
        }
        be.synchronize(0).unwrap();
    };
    run(calls.len().min(8));
    let mut v = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        run(reps);
        v.push(t0.elapsed().as_secs_f64() / reps as f64 * 1e6);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[2], v[0])
}

fn kern(ks: &[KernelImpl], name: &str) -> KernelImpl {
    ks.iter().find(|k| k.name == name).unwrap_or_else(|| panic!("no kernel {name}")).clone()
}

fn bench_proj(be: &dyn Backend, ks: &[KernelImpl], rows: &[usize]) {
    const D: usize = 2304;
    let ns = [6552usize, 1056, 1056];
    let n: usize = ns.iter().sum();
    let copies = 4; // 4 x 40 MB
    let ws: Vec<Buffer> = (0..copies).map(|c| bf16_buf(be, &[n, D], 100 + c as u64, (D as f32).powf(-0.5))).collect();
    let wbytes = (n * D * 2) as f64;
    println!("== InkInProj [{n}, {D}] bf16, {copies} weight copies of {:.1} MB", wbytes / 1e6);
    for &tr in rows {
        for s in [1usize, 3] {
            let xb = bf16_buf(be, &[s, tr, D], 7 + tr as u64, 1.0);
            let outs: Vec<Buffer> = ns.iter().map(|&nn| be.alloc(tr * nn * 2, MemKind::Device).unwrap()).collect();
            let names: Vec<&str> = if tr <= 16 { vec!["ink_in_proj_bf16_mma"] } else { vec!["ink_in_proj_bf16_cublas", "ink_in_proj_bf16_mma"] };
            for kname in names {
                let k = kern(ks, kname);
                let calls: Vec<OpCall> = (0..copies)
                    .map(|c| {
                        let call = OpCall::new(
                            Op::InkInProj,
                            vec![t(&xb, DType::Bf16, &[s, tr, D]), t(&ws[c], DType::Bf16, &[n, D])],
                            ns.iter().zip(&outs).map(|(&nn, b)| t(b, DType::Bf16, &[tr, nn])).collect(),
                            OpAttrs::default(),
                        );
                        with_scratch(be, &k, call)
                    })
                    .collect();
                let reps = if tr > 64 { 20 } else { 200 };
                let (us, mn) = time(be, kname, &calls, reps);
                println!("proj T={tr:5} S={s} {kname:26} {us:9.1} us (min {mn:8.1})  {:6.1} GB/s weights", wbytes / (us * 1e-6) / 1e9);
            }
            if s == 1 && tr <= 16 {
                // the existing best bf16 GEMV over the stacked weight (one launch, `rows_exact` twin) for comparison
                let kname = "matmul_bf16_gemv_rows";
                let ob = be.alloc(tr * n * 2, MemKind::Device).unwrap();
                let calls: Vec<OpCall> = (0..copies)
                    .map(|c| {
                        OpCall::new(Op::Matmul, vec![t(&xb, DType::Bf16, &[tr, D]), t(&ws[c], DType::Bf16, &[n, D])], vec![t(&ob, DType::Bf16, &[tr, n])], OpAttrs::default())
                    })
                    .collect();
                let (us, mn) = time(be, kname, &calls, 200);
                println!("proj T={tr:5} S=1 {kname:26} {us:9.1} us (min {mn:8.1})  {:6.1} GB/s weights", wbytes / (us * 1e-6) / 1e9);
            }
        }
    }
}

fn bench_moe(be: &dyn Backend, ks: &[KernelImpl], rows: &[usize]) {
    const E: usize = 128;
    const K: usize = 8;
    const H: usize = 2304;
    const I: usize = 1152;
    let nib = |r: u64| (r >> 40) as u8;
    let nvsc = |r: u64| 0x28 + ((r >> 40) % 8) as u8;
    let e4 = |r: u64| ((r >> 33) & 0x80) as u8 | ((4 + (r >> 41) % 7) as u8) << 3 | ((r >> 50) & 7) as u8;
    let e8 = |r: u64| 116 + ((r >> 40) % 4) as u8;
    let probe_env = std::env::var("INKMOE_PROBE").is_ok();
    for mx in [false, true] {
        let layers = if mx { 3 } else { 5 };
        let ebytes = if mx { 8_211_456usize } else { 4_478_976 };
        println!("== {} experts, {layers} layer copies of {:.0} MB", if mx { "MXFP8" } else { "NVFP4" }, (E * ebytes) as f64 / 1e6);
        let w: Vec<Vec<TensorRef>> = (0..layers)
            .map(|l| {
                let s = 1000 * l as u64;
                if mx {
                    vec![
                        t(&fill(be, E * 2 * I * H, s + 1, &e4), DType::Fp8E4m3, &[E, 2 * I, H]),
                        t(&fill(be, E * 2 * I * H / 32, s + 2, &e8), DType::U8, &[E, 2 * I, H / 32]),
                        t(&fill(be, E * H * I, s + 3, &e4), DType::Fp8E4m3, &[E, H, I]),
                        t(&fill(be, E * H * I / 32, s + 4, &e8), DType::U8, &[E, H, I / 32]),
                    ]
                } else {
                    let gs = be.alloc(E * 12, MemKind::Device).unwrap();
                    be.upload(&gs, 0, &vec![0.05f32; E * 3].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), 0).unwrap();
                    vec![
                        t(&fill(be, E * 2 * I * H / 2, s + 1, &nib), DType::U8, &[E, 2 * I, H / 2]),
                        t(&fill(be, E * 2 * I * H / 16, s + 2, &nvsc), DType::Fp8E4m3, &[E, 2 * I, H / 16]),
                        t(&fill(be, E * H * I / 2, s + 3, &nib), DType::U8, &[E, H, I / 2]),
                        t(&fill(be, E * H * I / 16, s + 4, &nvsc), DType::Fp8E4m3, &[E, H, I / 16]),
                        t(&gs, DType::F32, &[E, 3]),
                    ]
                }
            })
            .collect();
        let kname = if mx { "moe_experts_mxfp8_bf16" } else { "moe_experts_situ_bf16" };
        let k = kern(ks, kname);
        for &tr in rows {
            let ncalls = if tr > 64 { 6 } else { 48 };
            let mut rng = intarsia_core::rng::SplitMix64::new(0x1a4b ^ tr as u64);
            let xb = bf16_buf(be, &[tr, H], 3 + tr as u64, 1.0);
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
                    let mut ins = vec![t(&xb, DType::Bf16, &[tr, H]), t(&ib, DType::I32, &[tr, K]), t(&wb, DType::F32, &[tr, K])];
                    ins.extend(w[c % layers].iter().cloned());
                    let mut a = OpAttrs { top_k: K, ..OpAttrs::default() };
                    (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
                    let op = if mx { Op::MoeExpertsMxfp8 } else { Op::MoeExperts };
                    with_scratch(be, &k, OpCall::new(op, ins, vec![t(&out, DType::Bf16, &[tr, H])], a))
                })
                .collect();
            let mean_union = unions.iter().sum::<usize>() as f64 / unions.len() as f64;
            let probes: &[i32] = if probe_env && tr <= 16 { &[0, 1, 2] } else { &[0] };
            for &probe in probes {
                intarsia_backend_cuda::inkmoe::set_probe(probe);
                let reps = if tr > 64 { 6 } else { 240 };
                let (us, mn) = time(be, kname, &calls, reps);
                let what = ["all", "gate+up", "down"][probe as usize];
                let gbs = mean_union * ebytes as f64 / (us * 1e-6) / 1e9 * [1.0, 0.6667, 0.3333][probe as usize];
                println!("moe {} T={tr:5} {what:8} {us:9.1} us (min {mn:8.1})  union {mean_union:5.1}  {gbs:6.1} GB/s", if mx { "mx" } else { "nv" });
            }
            intarsia_backend_cuda::inkmoe::set_probe(0);
        }
    }
}

fn bench_mix(be: &dyn Backend, ks: &[KernelImpl], rows: &[usize]) {
    const D: usize = 2304;
    const KT: usize = 4;
    const G: usize = 64;
    let rr = KT - 1 + 16;
    let bf = DType::Bf16;
    let none = be.alloc(16, MemKind::Device).unwrap();
    let nt = |dt: DType| t(&none, dt, &[0]);
    println!("== mix ops, D {D}, ring {rr}");
    for &tr in rows {
        let x = bf16_buf(be, &[tr, D], 1, 1.0);
        let taps = bf16_buf(be, &[KT, D], 2, 0.5);
        let res = bf16_buf(be, &[tr, D], 3, 1.0);
        let add = bf16_buf(be, &[tr, D], 4, 1.0);
        let ring = bf16_buf(be, &[rr, D], 5, 1.0);
        let y = be.alloc(tr * D * 2, MemKind::Device).unwrap();
        let sum = be.alloc(tr * D * 2, MemKind::Device).unwrap();
        let a = OpAttrs { eps: 1e-6, pos0: 29, ..OpAttrs::default() };
        // (label, N, Q, norm, S)
        for &(n, q, norm, s) in &[(4usize, 1usize, true, 0usize), (7, 3, true, 0), (0, 0, false, 0), (4, 1, false, 3), (4, 1, false, 1)] {
            let src = bf16_buf(be, &[n.max(1), tr, D], 6, 1.5);
            let qs = f32_buf(be, &[q.max(1), D], 7, 0.2 / (D as f32).sqrt());
            let nrows = if s > 0 { s } else { q };
            let nw = bf16_buf(be, &[nrows.max(1), D], 8, 0.3);
            let msg = f32_buf(be, &[s.max(1), tr, D], 9, 1.0);
            let down = bf16_buf(be, &[s.max(1), G, 2 * D], 10, (2.0 * D as f32).powf(-0.5));
            let gain = bf16_buf(be, &[s.max(1), D, G], 11, 0.2);
            let forget = bf16_buf(be, &[s.max(1), D, G], 12, 0.2);
            let mrows = if s > 0 { s } else { q };
            let mix = be.alloc((mrows.max(1) * tr * D * 4).max(16), MemKind::Device).unwrap();
            let k = kern(ks, "conv_mix_bf16");
            let ins = vec![
                t(&x, bf, &[tr, D]),
                t(&taps, bf, &[KT, D]),
                t(&res, bf, &[tr, D]),
                t(&add, bf, &[tr, D]),
                t(&src, bf, &[n, tr, D]),
                t(&qs, DType::F32, &[q, D]),
                if norm || s > 0 { t(&nw, bf, &[nrows, D]) } else { nt(bf) },
                if s > 0 { t(&msg, DType::F32, &[s, tr, D]) } else { nt(bf) },
                if s > 0 { t(&down, bf, &[s, G, 2 * D]) } else { nt(bf) },
                if s > 0 { t(&gain, bf, &[s, D, G]) } else { nt(bf) },
                if s > 0 { t(&forget, bf, &[s, D, G]) } else { nt(bf) },
            ];
            let mixo = if q == 0 { nt(bf) } else if s > 0 { t(&mix, bf, &[s, tr, D]) } else if norm { t(&mix, bf, &[q, tr, D]) } else { t(&mix, DType::F32, &[q, tr, D]) };
            let outs = vec![t(&y, bf, &[tr, D]), t(&ring, bf, &[rr, D]), t(&sum, bf, &[tr, D]), mixo];
            let call = with_scratch(be, &k, OpCall::new(Op::ConvMix, ins, outs, a.clone()));
            let reps = if tr > 64 { 20 } else { 400 };
            let (us, mn) = time(be, "conv_mix_bf16", &[call], reps);
            println!("conv_mix T={tr:5} N={n} Q={q} norm={norm} S={s}: {us:8.1} us (min {mn:7.1})");
        }
        // SconvRes alone and AttnResMix (N 4, Q 1, norm)
        {
            let k = kern(ks, "sconv_res_bf16");
            let ins = vec![t(&x, bf, &[tr, D]), t(&taps, bf, &[KT, D]), t(&res, bf, &[tr, D]), t(&add, bf, &[tr, D])];
            let outs = vec![t(&y, bf, &[tr, D]), t(&ring, bf, &[rr, D]), t(&sum, bf, &[tr, D])];
            let call = with_scratch(be, &k, OpCall::new(Op::SconvRes, ins, outs, a.clone()));
            let (us, mn) = time(be, "sconv_res_bf16", &[call], if tr > 64 { 20 } else { 400 });
            println!("sconv_res T={tr:5}: {us:8.1} us (min {mn:7.1})");
            let src = bf16_buf(be, &[4, tr, D], 6, 1.5);
            let qs = f32_buf(be, &[1, D], 7, 0.01);
            let nw = bf16_buf(be, &[1, D], 8, 0.3);
            let o = be.alloc(tr * D * 2, MemKind::Device).unwrap();
            let k = kern(ks, "attn_res_mix_bf16");
            let ins = vec![t(&src, bf, &[4, tr, D]), t(&res, bf, &[tr, D]), t(&add, bf, &[tr, D]), t(&qs, DType::F32, &[1, D]), t(&nw, bf, &[1, D])];
            let call = with_scratch(be, &k, OpCall::new(Op::AttnResMix, ins, vec![t(&o, bf, &[1, tr, D])], a.clone()));
            let (us, mn) = time(be, "attn_res_mix_bf16", &[call], if tr > 64 { 20 } else { 400 });
            println!("attn_res_mix T={tr:5} N=4 Q=1 norm: {us:8.1} us (min {mn:7.1})");
            for s in [1usize, 3] {
                let dest = f32_buf(be, &[1, tr, D], 20, 1.0);
                let msg = f32_buf(be, &[s, tr, D], 21, 1.0);
                let down = bf16_buf(be, &[s, G, 2 * D], 22, (2.0 * D as f32).powf(-0.5));
                let gain = bf16_buf(be, &[s, D, G], 23, 0.2);
                let forget = bf16_buf(be, &[s, D, G], 24, 0.2);
                let nw = bf16_buf(be, &[s, D], 25, 0.3);
                let o = be.alloc(s * tr * D * 2, MemKind::Device).unwrap();
                let k = kern(ks, "loop_blend_bf16");
                let ins = vec![
                    t(&dest, DType::F32, &[1, tr, D]),
                    t(&msg, DType::F32, &[s, tr, D]),
                    t(&down, bf, &[s, G, 2 * D]),
                    t(&gain, bf, &[s, D, G]),
                    t(&forget, bf, &[s, D, G]),
                    t(&nw, bf, &[s, D]),
                ];
                let call = with_scratch(be, &k, OpCall::new(Op::LoopBlend, ins, vec![t(&o, bf, &[s, tr, D])], a.clone()));
                let (us, mn) = time(be, "loop_blend_bf16", &[call], if tr > 64 { 20 } else { 400 });
                println!("loop_blend T={tr:5} S={s}: {us:8.1} us (min {mn:7.1})");
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let what: Vec<&str> = {
        let w: Vec<&str> = args.iter().map(|s| s.as_str()).filter(|a| matches!(*a, "proj" | "moe" | "mix")).collect();
        if w.is_empty() {
            vec!["proj", "moe", "mix"]
        } else {
            w
        }
    };
    let rows: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let be: Arc<dyn Backend> = intarsia_backend_cuda::create(0).expect("cuda backend");
    let ks = be.kernels();
    for w in what {
        match w {
            "proj" => bench_proj(&*be, &ks, if rows.is_empty() { &[1, 2, 4, 8, 16, 2048] } else { &rows }),
            "moe" => bench_moe(&*be, &ks, if rows.is_empty() { &[1, 4, 16, 2048] } else { &rows }),
            "mix" => bench_mix(&*be, &ks, if rows.is_empty() { &[1, 4, 16, 2048] } else { &rows }),
            _ => {}
        }
    }
}
