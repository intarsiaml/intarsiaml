//! Microbenchmark of the Inkling-2 decode controller fusion (lane ink-dfuse, docs/results/2026-11-21-inkling2-dfuse.md): the
//! fused router `ink_route_bf16` (op 76) against the pair it replaces (`logits_bf16_gemv` / `_rows` + `moe_route_f32`), the
//! blend (LoopBlend, S = 3 and 1) and the two ConvMix call sites with the load-order switches on and off, at the real shapes
//! (D 2304, 128 experts top-8, rank 64).
//! GPU time per unit (a unit = the launches one call site makes) from the backend's `INTARSIA_METAL_PROFILE=cb:<file>`
//! trace, many dependent units per command buffer, each on its own operand set (48 sets: the weights rotate through more than
//! the system cache holds, like the 48 visits of a decode step).
//!
//!   cargo run --release -p intarsia-backend-metal --example ink_ctrl_bench -- [--reps N] [--rows 1,2,3,4,16] [--trace F]
//!
//! Run inside a fleetq slot (`--quiet gpu,cpu`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ink_ctrl_bench needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{f32_to_bf16, stats, Rng};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, MemKind, TensorRef};
    use intarsia_core::dsv41::ops::ScoreFunc;
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

    const D: usize = 2304;
    const E: usize = 128;
    const K: usize = 8;
    const G: usize = 64;
    const SETS: usize = 48;

    fn buf(be: &MetalBackend, dt: DType, shape: &[usize], scale: f32, rng: &mut Rng) -> TensorRef {
        let n: usize = shape.iter().product();
        let bytes: Vec<u8> = match dt {
            DType::Bf16 => (0..n).flat_map(|_| f32_to_bf16(rng.unit() * scale).to_le_bytes()).collect(),
            _ => (0..n).flat_map(|_| (rng.unit() * scale).to_le_bytes()).collect(),
        };
        let b = be.alloc(bytes.len().max(16) + 64, MemKind::Device).unwrap();
        if !bytes.is_empty() {
            be.upload(&b, 0, &bytes, 0).unwrap();
        }
        TensorRef { buf: b, offset: 0, dtype: dt, shape: shape.to_vec() }
    }

    fn zeros(be: &MetalBackend, dt: DType, shape: &[usize]) -> TensorRef {
        let n: usize = shape.iter().product::<usize>() * dt.size_bytes();
        TensorRef { buf: be.alloc(n.max(16) + 64, MemKind::Device).unwrap(), offset: 0, dtype: dt, shape: shape.to_vec() }
    }

    struct L {
        kernel: &'static str,
        call: OpCall,
    }

    /// Median / p10 / p90 GPU us per unit: each rep runs the unit list `inner` times (about 30 ms of GPU work), 3 warm-ups.
    fn bench(be: &MetalBackend, trace: &std::path::Path, units: &[Vec<L>], reps: usize, before: &dyn Fn()) -> (f64, f64, f64) {
        use std::io::{Read, Seek, SeekFrom};
        before();
        let mut f = std::fs::OpenOptions::new().read(true).open(trace).ok();
        let mut pos = f.as_mut().map_or(0, |f| f.seek(SeekFrom::End(0)).unwrap());
        let mut inner = 1usize;
        let mut samples = Vec::new();
        for rep in 0..reps + 3 {
            for _ in 0..inner {
                for u in units {
                    for l in u {
                        be.launch(l.kernel, &l.call, 0).unwrap();
                    }
                }
            }
            be.synchronize(0).unwrap();
            if f.is_none() {
                f = std::fs::OpenOptions::new().read(true).open(trace).ok();
            }
            let Some(fh) = f.as_mut() else { continue };
            fh.seek(SeekFrom::Start(pos)).unwrap();
            let mut text = String::new();
            fh.read_to_string(&mut text).unwrap();
            pos += text.len() as u64;
            let mut tot = 0.0;
            for l in text.lines() {
                let r: serde_json::Value = serde_json::from_str(l).unwrap();
                tot += r["e"].as_f64().unwrap() - r["s"].as_f64().unwrap();
            }
            let per = tot / (units.len() * inner) as f64;
            if rep == 0 {
                inner = ((0.03 / (per * units.len() as f64).max(1e-7)) as usize).clamp(1, 4000);
            } else if rep >= 3 {
                samples.push(per * 1e6);
            }
        }
        let st = stats(&mut samples);
        (st.median, st.p10, st.p90)
    }

    fn route_attrs() -> OpAttrs {
        let mut a = OpAttrs { top_k: K, scale: 8.0, norm_topk: true, temperature: 1.0, ..OpAttrs::default() };
        a.dsv.score_func = ScoreFunc::Sigmoid;
        a
    }

    /// (fused units, pair units) of the router at `t` rows.
    fn route_units(be: &MetalBackend, rng: &mut Rng, t: usize) -> (Vec<Vec<L>>, Vec<Vec<L>>) {
        let (mut fu, mut pu) = (Vec::new(), Vec::new());
        for _ in 0..SETS {
            let x = buf(be, DType::Bf16, &[t, D], 1.0, rng);
            let w = buf(be, DType::Bf16, &[E, D], (D as f32).powf(-0.5), rng);
            let bias = buf(be, DType::F32, &[E], 0.1, rng);
            let (wo, ids, lg) = (zeros(be, DType::F32, &[t, K]), zeros(be, DType::I32, &[t, K]), zeros(be, DType::F32, &[t, E]));
            // fused, logits to the call's output (as the graph runs it, for the dumps)
            let mut fc = OpCall::new(Op::InkRoute, vec![x.clone(), w.clone(), bias.clone()], vec![wo.clone(), ids.clone(), lg.clone()], route_attrs());
            fc.exact_rows = t > 1;
            fc.check().unwrap();
            fu.push(vec![L { kernel: intarsia_backend_metal::inkctrl::INK_ROUTE_BF16, call: fc }]);
            let mut lc = OpCall::new(Op::Logits, vec![x, w], vec![lg.clone()], OpAttrs::default());
            lc.exact_rows = t > 1;
            lc.check().unwrap();
            let rc = OpCall::new(Op::MoeRoute, vec![lg, bias], vec![wo, ids], route_attrs());
            rc.check().unwrap();
            let lk = if t == 1 { "logits_bf16_gemv" } else { "logits_bf16_gemv_rows" };
            pu.push(vec![L { kernel: lk, call: lc }, L { kernel: "moe_route_f32", call: rc }]);
        }
        (fu, pu)
    }

    /// LoopBlend units with S streams at `t` rows (dest shared, as the graph's attention-side blend).
    fn blend_units(be: &MetalBackend, rng: &mut Rng, t: usize, s: usize) -> Vec<Vec<L>> {
        (0..SETS)
            .map(|_| {
                let ins = vec![
                    buf(be, DType::F32, &[1, t, D], 1.0, rng),
                    buf(be, DType::F32, &[s, t, D], 1.0, rng),
                    buf(be, DType::Bf16, &[s, G, 2 * D], (2.0 * D as f32).powf(-0.5), rng),
                    buf(be, DType::Bf16, &[s, D, G], 0.25, rng),
                    buf(be, DType::Bf16, &[s, D, G], 0.25, rng),
                    buf(be, DType::Bf16, &[s, D], 1.0, rng),
                ];
                let out = zeros(be, DType::Bf16, &[s, t, D]);
                let mut call = OpCall::new(Op::LoopBlend, ins, vec![out], OpAttrs { eps: 1e-6, ..OpAttrs::default() });
                call.exact_rows = t > 1;
                call.check().unwrap();
                let n = intarsia_backend_metal::inkmix::blend_scratch(&ShapeInfo::from_call(&call));
                call.scratch = Some(zeros(be, DType::U8, &[n]));
                vec![L { kernel: intarsia_backend_metal::inkmix::LOOP_BLEND_BF16, call }]
            })
            .collect()
    }

    /// ConvMix units at `t` rows like the graph's two call sites (K 4, ring 19, N = 4 stored sources): the attention side
    /// (`res`, Q 1, blend S 1 on looped visits) and the MLP side (`res` + `add` -> sum, Q 3, blend S 3 on looped visits);
    /// `s` = 0 is a plain visit (the norm instead of the blend).
    fn conv_units(be: &MetalBackend, rng: &mut Rng, t: usize, mlp: bool, s: usize) -> Vec<Vec<L>> {
        let (n, q, rr) = (4usize, if mlp { 3usize } else { 1 }, 19usize);
        let bf = DType::Bf16;
        let none = |be: &MetalBackend| zeros(be, bf, &[0]);
        (0..SETS)
            .map(|_| {
                let mut ins = vec![
                    buf(be, bf, &[t, D], 1.0, rng),
                    buf(be, bf, &[4, D], 0.5, rng),
                    buf(be, bf, &[t, D], 1.0, rng),
                    if mlp { buf(be, bf, &[t, D], 1.0, rng) } else { none(be) },
                    buf(be, bf, &[n, t, D], 1.5, rng),
                    buf(be, DType::F32, &[q, D], 0.2, rng),
                    buf(be, bf, &[if s > 0 { s } else { q }, D], 1.0, rng),
                ];
                if s > 0 {
                    ins.push(buf(be, DType::F32, &[s, t, D], 1.0, rng));
                    ins.push(buf(be, bf, &[s, G, 2 * D], (2.0 * D as f32).powf(-0.5), rng));
                    ins.push(buf(be, bf, &[s, D, G], 0.25, rng));
                    ins.push(buf(be, bf, &[s, D, G], 0.25, rng));
                } else {
                    ins.extend([none(be), none(be), none(be), none(be)]);
                }
                let outs = vec![
                    zeros(be, bf, &[t, D]),
                    buf(be, bf, &[rr, D], 1.0, rng),
                    if mlp { zeros(be, bf, &[t, D]) } else { none(be) },
                    zeros(be, bf, &[if s > 0 { s } else { q }, t, D]),
                ];
                let mut call = OpCall::new(Op::ConvMix, ins, outs, OpAttrs { pos0: 100, eps: 1e-6, ..OpAttrs::default() });
                call.exact_rows = t > 1;
                call.check().unwrap();
                let nb = intarsia_backend_metal::inkmix::blend_scratch(&ShapeInfo::from_call(&call));
                call.scratch = Some(zeros(be, DType::U8, &[nb]));
                vec![L { kernel: intarsia_backend_metal::inkmix::CONV_MIX_BF16, call }]
            })
            .collect()
    }

    pub fn main() {
        let mut reps = 40usize;
        let mut rows = vec![1usize, 2, 3, 4, 16];
        let mut trace = std::path::PathBuf::from(format!("target/ink_ctrl_bench-{}.jsonl", std::process::id()));
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--reps" => reps = args.next().unwrap().parse().unwrap(),
                "--rows" => rows = args.next().unwrap().split(',').map(|v| v.parse().unwrap()).collect(),
                "--trace" => trace = args.next().unwrap().into(),
                o => panic!("unknown argument {o}"),
            }
        }
        std::env::set_var("INTARSIA_METAL_PROFILE", format!("cb:{}", trace.display()));
        let be = MetalBackend::new().expect("metal backend");
        assert!(be.has_metal4(), "ink_ctrl_bench needs the metal4.0 library");
        use intarsia_backend_metal::inkctrl::{BLEND_NORM_C, MIX_HOIST};
        let mut rng = Rng(0xc7_1be4c4);
        let none = || {};
        println!("| what | T | us per call (median) | p10 | p90 |");
        println!("|---|---:|---:|---:|---:|");
        for &t in &rows {
            let (fu, pu) = route_units(&be, &mut rng, t);
            let (m, lo, hi) = bench(&be, &trace, &pu, reps, &none);
            println!("| router pair (logits_bf16_gemv{} + moe_route_f32) | {t} | {m:.2} | {lo:.2} | {hi:.2} |", if t == 1 { "" } else { "_rows" });
            let (m, lo, hi) = bench(&be, &trace, &fu, reps, &none);
            println!("| ink_route_bf16 (fused) | {t} | {m:.2} | {lo:.2} | {hi:.2} |");
            drop((fu, pu));
            for s in [3usize, 1] {
                let bu = blend_units(&be, &mut rng, t, s);
                for (c, name) in [(false, "off"), (true, "norm_c")] {
                    let (m, lo, hi) = bench(&be, &trace, &bu, reps, &move || BLEND_NORM_C.set(Some(c)));
                    println!("| loop_blend S={s} (4 dispatches): {name} | {t} | {m:.2} | {lo:.2} | {hi:.2} |");
                }
            }
            BLEND_NORM_C.set(None);
            // the load-order switches (INTARSIA_INK_MIX_HOIST, INTARSIA_INK_BLEND_NORM_C) on the two ConvMix call sites
            let sw = |h: bool, c: bool| {
                move || {
                    MIX_HOIST.set(Some(h));
                    BLEND_NORM_C.set(Some(c));
                }
            };
            for (mlp, s, what) in [(false, 0usize, "conv_mix attn side, plain"), (false, 1, "conv_mix attn side, blend S=1"), (true, 0, "conv_mix mlp side, plain (Q 3)"), (true, 3, "conv_mix mlp side, blend S=3")] {
                let cu = conv_units(&be, &mut rng, t, mlp, s);
                let cfgs: &[(bool, bool, &str)] = if s > 0 { &[(false, false, "off"), (true, false, "hoist"), (false, true, "norm_c"), (true, true, "hoist + norm_c")] } else { &[(false, false, "off"), (true, false, "hoist")] };
                for &(h, c, name) in cfgs {
                    let (m, lo, hi) = bench(&be, &trace, &cu, reps, &sw(h, c));
                    println!("| {what}: {name} | {t} | {m:.2} | {lo:.2} | {hi:.2} |");
                }
            }
            MIX_HOIST.set(None);
            BLEND_NORM_C.set(None);
        }
        let _ = std::fs::remove_file(&trace);
    }
}
