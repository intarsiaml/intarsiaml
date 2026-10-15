//! Batched decode step vs B solo steps (Qwen3-8B BF16, Metal): the speed measurement of `Model::forward_batch`.
//!
//!   cargo run --release -p intarsia-models --example batch_step -- [--ctx 1024] [--rounds 8] [--steps 16] [--batches 1,2,4,8]
//!   env: INTARSIA_METAL_TEST_MODEL (default ~/models/Qwen3-8B)
//!
//! For each batch size B the first B of 8 prefilled sequences (distinct random prompts of `--ctx` tokens) are stepped
//! `--steps` tokens by B solo steps in turn, and the same by one batched step (two independent state sets fed the same
//! tokens, greedy on-device argmax so every step ends on one synchronised download). The two variants alternate round
//! by round (A B A B ...) so clock drift hits both alike; the table gives the median per-step time of each and the
//! aggregate token rate. The result is checked for equality of the sampled tokens while it runs. Run inside a fleetq
//! slot (exclusive on endor): it needs the whole GPU.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("batch_step needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::stats;
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
        let batches: Vec<usize> = arg(&args, "--batches", "1,2,4,8").split(',').map(|x| x.parse().unwrap()).collect();
        let dir = std::env::var_os("INTARSIA_METAL_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("models/Qwen3-8B"));
        let backend = intarsia_backend_metal::MetalBackend::arc().unwrap();
        let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
        let weights = SchemedSource::open(&dir).unwrap();
        let chunk = recipe.resolve_prefill_chunk(None, "metal").value;
        let max_seq = ctx + (rounds + 1) * steps * batches.len() + 64;
        let opts = LoadOptions { max_seq, prefill_chunk: chunk, ..Default::default() };
        let mut m = Model::load(backend, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
        let top = *batches.iter().max().unwrap();
        if top > 1 {
            m.enable_batch(top.max(2)).unwrap();
        }
        m.configure_plan(&Default::default(), None).unwrap();
        println!("# Qwen3-8B BF16 Metal, context {ctx} (+{} decoded), {rounds} rounds x {steps} steps, chunk {chunk}", rounds * steps);

        let mut solo: Vec<SequenceState> = Vec::new();
        let mut bat: Vec<SequenceState> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        for i in 0..top {
            let mut rng = intarsia_core::rng::SplitMix64::new(0xbe7c_0000 + i as u64);
            let p: Vec<u32> = (0..ctx).map(|_| 1000 + (rng.next_u64() % 29000) as u32).collect();
            let (mut a, mut b) = (m.new_state().unwrap(), m.new_state().unwrap());
            m.forward(&mut a, &p[..ctx - 1], HeadMode::None, None).unwrap();
            m.forward(&mut b, &p[..ctx - 1], HeadMode::None, None).unwrap();
            solo.push(a);
            bat.push(b);
            cur.push(p[ctx - 1]);
        }
        println!("| B | B solo steps ms | 1 batched step ms | speedup | solo tok/s | batched tok/s | design est. ms (1k) |");
        println!("|---:|---:|---:|---:|---:|---:|---:|");
        let est = |b: usize| match b {
            1 => 27.3,
            2 => 28.0,
            4 => 29.3,
            6 => 31.1,
            8 => 36.6,
            _ => f64::NAN,
        };
        for &b in &batches {
            let (mut ts, mut tb) = (Vec::new(), Vec::new());
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
                if batched_first {
                    bb = run_batch(&mut tb, &mut bat, &mut cb);
                    a = run_solo(&mut ts, &mut solo, &mut cs);
                } else {
                    a = run_solo(&mut ts, &mut solo, &mut cs);
                    bb = run_batch(&mut tb, &mut bat, &mut cb);
                }
                assert_eq!(a, bb, "batched tokens differ from solo tokens (B={b}, round {round})");
                cur = cs;
            }
            // the first round of each variant is warm-up
            ts.drain(..steps);
            tb.drain(..steps);
            let (s, t) = (stats(&mut ts), stats(&mut tb));
            println!(
                "| {b} | {:.2} | {:.2} | {:.2}x | {:.1} | {:.1} | {} |",
                s.median,
                t.median,
                s.median / t.median,
                b as f64 * 1e3 / s.median,
                b as f64 * 1e3 / t.median,
                if est(b).is_nan() { "-".into() } else { format!("{:.1}", est(b)) }
            );
        }
    }
}
