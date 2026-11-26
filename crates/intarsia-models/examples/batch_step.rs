//! Batched decode step vs B solo steps (Qwen3-8B BF16, Metal; any model, HIP): the speed measurement of `Model::forward_batch`.
//!
//!   cargo run --release -p intarsia-models --example batch_step -- [--ctx 1024] [--rounds 8] [--steps 16] [--batches 1,2,4,8]
//!       [--prompts text|random] [--digest] [--backend metal|hip] [--model DIR] [--mem-budget GIB] [--batched-only]
//!       [--prefill-chunk N]
//!   env: INTARSIA_METAL_TEST_MODEL (default ~/models/Qwen3-8B; `--model` wins)
//!   `--backend hip` (lane ink-hip-batch; the default on Linux): one R9700, card 0 of HIP_VISIBLE_DEVICES.
//!   `--batched-only`: no solo states or steps (half the prefill, no token check): long-context speed rows only. The last column is
//!   the p99 of the batched steps (every stream's inter-token latency in a pure decode batch).
//!
//! Prompts: `--prompts text` (default; distinct slices of a fixed real-text corpus, examples/common/prompts.rs) or
//! `random` (the old uniform-random ids, whose rows share experts far more than real text). `--digest` prints a digest of
//! each row's batched greedy tokens per B (for comparing arms bitwise).
//!
//! For each batch size B the first B of the prefilled sequences (distinct prompts of `--ctx` tokens) are stepped
//! `--steps` tokens by B solo steps in turn, and the same by one batched step (two independent state sets fed the same
//! tokens, greedy on-device argmax so every step ends on one synchronised download). The two variants alternate round
//! by round (A B A B ...) so clock drift hits both alike; the table gives the median per-step time of each and the
//! aggregate token rate. The result is checked for equality of the sampled tokens while it runs. Run inside a fleetq
//! slot (exclusive on endor): it needs the whole GPU.

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn main() {
    eprintln!("batch_step needs macOS / Metal or Linux / HIP");
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn main() {
    imp::main()
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[path = "common/prompts.rs"]
mod prompts;

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod imp {
    use super::prompts::{digest, Prompts};
    use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
    use intarsia_core::state::SequenceState;
    use intarsia_core::Telemetry;
    use intarsia_models::load::SchemedSource;
    use intarsia_models::{read_model_spec, EngineOptions};
    use std::sync::Arc;
    use std::time::Instant;

    fn arg(args: &[String], name: &str, default: &str) -> String {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.into())
    }

    struct Stats {
        median: f64,
        p99: f64,
    }

    /// The median and p99 of the samples (interpolated, as `intarsia_backend_metal::bench::stats`).
    fn stats(v: &mut [f64]) -> Stats {
        if v.is_empty() {
            return Stats { median: f64::NAN, p99: f64::NAN };
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| {
            let r = p * (v.len() - 1) as f64;
            let (lo, hi) = (r.floor() as usize, r.ceil() as usize);
            v[lo] + (v[hi] - v[lo]) * (r - lo as f64)
        };
        Stats { median: q(0.5), p99: q(0.99) }
    }

    fn backend(name: &str) -> std::sync::Arc<dyn intarsia_core::Backend> {
        match name {
            #[cfg(target_os = "macos")]
            "metal" => intarsia_backend_metal::MetalBackend::arc().unwrap(),
            "hip" => intarsia_backend_hip::create(0).unwrap(),
            other => panic!("--backend {other}: metal (macOS) or hip"),
        }
    }

    fn tok(o: &StepOut) -> u32 {
        match o {
            StepOut::Token(t) => *t,
            _ => panic!("no token"),
        }
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let ctx: usize = arg(&args, "--ctx", "1024").parse().unwrap();
        let rounds: usize = arg(&args, "--rounds", "8").parse().unwrap();
        let steps: usize = arg(&args, "--steps", "16").parse().unwrap();
        // sequence i starts at ctx + i * spread (mixed positions; 0: all at ctx, as before)
        let spread: usize = arg(&args, "--ctx-spread", "0").parse().unwrap();
        let batches: Vec<usize> = arg(&args, "--batches", "1,2,4,8").split(',').map(|x| x.parse().unwrap()).collect();
        let want_digest = args.iter().any(|a| a == "--digest");
        let batched_only = args.iter().any(|a| a == "--batched-only");
        let bname = arg(&args, "--backend", if cfg!(target_os = "macos") { "metal" } else { "hip" });
        let dir = args
            .iter()
            .position(|a| a == "--model")
            .and_then(|i| args.get(i + 1))
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("INTARSIA_METAL_TEST_MODEL").map(std::path::PathBuf::from))
            .unwrap_or_else(|| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("models/Qwen3-8B"));
        let prompts = Prompts::new(&arg(&args, "--prompts", "text"), &dir);
        let backend = backend(&bname);
        let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
        let weights = SchemedSource::open(&dir).unwrap();
        let chunk = args
            .iter()
            .position(|a| a == "--prefill-chunk")
            .and_then(|i| args.get(i + 1))
            .map(|v| v.parse().unwrap())
            .unwrap_or_else(|| recipe.resolve_prefill_chunk(None, &bname).value);
        let top0 = *batches.iter().max().unwrap();
        let max_seq = ctx + spread * top0 + (rounds + 1) * steps * batches.len() + 64;
        let mem_budget = args.iter().position(|a| a == "--mem-budget").and_then(|i| args.get(i + 1)).map(|g| (g.parse::<f64>().unwrap() * (1u64 << 30) as f64) as u64);
        let opts = LoadOptions { max_seq, prefill_chunk: chunk, profile_every: 0, mem_budget, ..Default::default() };
        let mut m = Model::load(backend, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
        let top = *batches.iter().max().unwrap();
        if top > 1 {
            m.enable_batch(top.max(2)).unwrap();
        }
        m.configure_plan(&Default::default(), None).unwrap();
        let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        println!("# {name} {bname}, context {ctx} (+{spread} per sequence, +{} decoded), {rounds} rounds x {steps} steps, chunk {chunk}, prompts: {}", rounds * steps, prompts.describe());

        let mut solo: Vec<SequenceState> = Vec::new();
        let mut bat: Vec<SequenceState> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        for i in 0..top {
            let n = ctx + i * spread;
            let p = prompts.row(i, n);
            let mut b = m.new_state().unwrap();
            m.forward(&mut b, &p[..n - 1], HeadMode::None, None).unwrap();
            if !batched_only {
                let mut a = m.new_state().unwrap();
                m.forward(&mut a, &p[..n - 1], HeadMode::None, None).unwrap();
                solo.push(a);
            }
            bat.push(b);
            cur.push(p[n - 1]);
        }
        println!("| B | B solo steps ms | 1 batched step ms | speedup | solo tok/s | batched tok/s | design est. ms (1k) | batched step p99 ms |");
        println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
        let est = |b: usize| match b {
            _ if bname != "metal" => f64::NAN,
            1 => 27.3,
            2 => 28.0,
            4 => 29.3,
            6 => 31.1,
            8 => 36.6,
            _ => f64::NAN,
        };
        let mut digests: Vec<String> = Vec::new();
        for &b in &batches {
            let (mut ts, mut tb) = (Vec::new(), Vec::new());
            let mut rows: Vec<Vec<u32>> = vec![Vec::new(); b];
            for round in 0..=rounds {
                let batched_first = round % 2 == 1;
                let run_solo = |ts: &mut Vec<f64>, solo: &mut Vec<SequenceState>, cur: &mut Vec<u32>| {
                    let mut toks = Vec::new();
                    for _ in 0..steps {
                        let t0 = Instant::now();
                        for i in 0..b {
                            let o = m.forward(&mut solo[i], &[cur[i]], HeadMode::LastArgmax, None).unwrap();
                            cur[i] = tok(&o);
                        }
                        ts.push(t0.elapsed().as_secs_f64() * 1e3);
                        toks.push(cur[..b].to_vec());
                    }
                    toks
                };
                let run_batch = |tb: &mut Vec<f64>, bat: &mut Vec<SequenceState>, cur: &mut Vec<u32>| {
                    let mut toks = Vec::new();
                    for _ in 0..steps {
                        let t0 = Instant::now();
                        let ids: Vec<u32> = cur[..b].to_vec();
                        let modes = vec![HeadMode::LastArgmax; b];
                        let mut refs: Vec<&mut SequenceState> = bat[..b].iter_mut().collect();
                        let outs = if b == 1 {
                            vec![m.forward(refs[0], &ids, HeadMode::LastArgmax, None).unwrap()]
                        } else {
                            m.forward_batch(&mut refs, &ids, &modes).unwrap()
                        };
                        for i in 0..b {
                            cur[i] = tok(&outs[i]);
                        }
                        tb.push(t0.elapsed().as_secs_f64() * 1e3);
                        toks.push(cur[..b].to_vec());
                    }
                    toks
                };
                // both sides start every round from the same tokens and lengths: run A then B on separate copies of `cur`
                let (mut cs, mut cb) = (cur.clone(), cur.clone());
                let (a, bb);
                if batched_only {
                    bb = run_batch(&mut tb, &mut bat, &mut cb);
                    a = bb.clone();
                    cs = cb.clone();
                } else if batched_first {
                    bb = run_batch(&mut tb, &mut bat, &mut cb);
                    a = run_solo(&mut ts, &mut solo, &mut cs);
                } else {
                    a = run_solo(&mut ts, &mut solo, &mut cs);
                    bb = run_batch(&mut tb, &mut bat, &mut cb);
                }
                assert_eq!(a, bb, "batched tokens differ from solo tokens (B={b}, round {round})");
                for step in &bb {
                    for (i, t) in step.iter().enumerate() {
                        rows[i].push(*t);
                    }
                }
                cur = cs;
            }
            // the first round of each variant is warm-up
            ts.drain(..steps.min(ts.len()));
            tb.drain(..steps);
            let (s, t) = (stats(&mut ts), stats(&mut tb));
            println!(
                "| {b} | {:.2} | {:.2} | {:.2}x | {:.1} | {:.1} | {} | {:.2} |",
                s.median,
                t.median,
                s.median / t.median,
                b as f64 * 1e3 / s.median,
                b as f64 * 1e3 / t.median,
                if est(b).is_nan() { "-".into() } else { format!("{:.1}", est(b)) },
                t.p99
            );
            if want_digest {
                let per: Vec<String> = rows.iter().map(|r| digest(r)).collect();
                digests.push(format!("# B {b} batched-token digest {} rows {}", digest(&rows.concat()), per.join(" ")));
            }
        }
        for d in &digests {
            println!("{d}");
        }
    }
}
