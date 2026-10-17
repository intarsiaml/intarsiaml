//! M3 stage 0 (go / no-go measurement, no engine change): the GPU times the batched-verify projection is built from.
//!
//!   cargo run --release -p intarsia-models --example m3_verify_cost -- [--ctx 512,1024,4096] [--rounds 10] [--reps 12]
//!   env: INTARSIA_METAL_TEST_MODEL (default ~/models/Qwen3-8B)
//!
//! Per context length, with 6 prefilled sequences (distinct random prompts of `ctx - 1` tokens), every pass below is timed
//! (host wall around the call; every pass ends on one synchronised download) and then rolled back with `truncate`, so
//! the context is the same for every sample:
//!   - `plain<B>`   B = 1..6: one batched decode step over B sequences (B = 1 is the solo step), greedy on-device argmax.
//!   - `verify<q>`  q = 2..6: `Model::forward_verify` of q rows on one sequence (the solo verify pass of 1 + k = q rows).
//!   - `iter_b<B>_k<k>`: today's iteration with one speculating sequence: a solo verify of 1 + k rows, then a batched step of
//!     the other B - 1 sequences (B = 2, 3; k = 2, 3, 4). Compared with verify<1+k> + plain<B-1> to see whether the costs add.
//!   - `iter_vv<N>_k4`: today's lockstep iteration with N speculating sequences (N = 2, 3): N solo verifies of 5 rows.
//! The kinds are interleaved round by round in rotated order (clock drift hits all alike). Output: `data,ctx,kind,round,rep,
//! ms,epoch_ms` lines; the epoch lets the analysis join the samples with a macmon GPU-clock trace. Run inside an exclusive
//! fleetq slot on endor.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("m3_verify_cost needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_core::model::{HeadMode, LoadOptions, Model};
    use intarsia_core::state::SequenceState;
    use intarsia_core::Telemetry;
    use intarsia_models::load::SchemedSource;
    use intarsia_models::{read_model_spec, EngineOptions};
    use std::sync::Arc;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    fn arg(args: &[String], name: &str, default: &str) -> String {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.into())
    }

    const NSEQ: usize = 6;

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let ctxs: Vec<usize> = arg(&args, "--ctx", "512,1024,4096").split(',').map(|x| x.parse().unwrap()).collect();
        let rounds: usize = arg(&args, "--rounds", "10").parse().unwrap();
        let reps: usize = arg(&args, "--reps", "12").parse().unwrap();
        let dir = std::env::var_os("INTARSIA_METAL_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("models/Qwen3-8B"));
        let backend = intarsia_backend_metal::MetalBackend::arc().unwrap();
        let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
        let weights = SchemedSource::open(&dir).unwrap();
        let chunk = recipe.resolve_prefill_chunk(None, "metal").value;
        let max_ctx = *ctxs.iter().max().unwrap();
        let opts = LoadOptions { max_seq: max_ctx + 64, prefill_chunk: chunk, profile_every: 0, ..Default::default() };
        let mut m = Model::load(backend.clone(), &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
        m.enable_batch(NSEQ).unwrap();
        m.enable_verify(NSEQ).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
        println!("# m3_verify_cost: Qwen3-8B BF16 Metal, ctx {ctxs:?}, {rounds} rounds x {reps} reps, chunk {chunk}");

        // the kinds, in a fixed order; each is (name, closure-id)
        let mut kinds: Vec<(String, Kind)> = Vec::new();
        for b in 1..=NSEQ {
            kinds.push((format!("plain{b}"), Kind::Plain(b)));
        }
        for q in 2..=NSEQ {
            kinds.push((format!("verify{q}"), Kind::Verify(q)));
        }
        for b in [2usize, 3] {
            for k in [2usize, 3, 4] {
                kinds.push((format!("iter_b{b}_k{k}"), Kind::Iter(b, k)));
            }
        }
        for n in [2usize, 3] {
            kinds.push((format!("iter_vv{n}_k4"), Kind::Vv(n, 4)));
        }

        for &ctx in &ctxs {
            let mut states: Vec<SequenceState> = Vec::new();
            let mut cur: Vec<u32> = Vec::new();
            for i in 0..NSEQ {
                let mut rng = intarsia_core::rng::SplitMix64::new(0x3c0d_0000 + (ctx * 16 + i) as u64);
                let p: Vec<u32> = (0..ctx).map(|_| 1000 + (rng.next_u64() % 29000) as u32).collect();
                let mut st = m.new_state().unwrap();
                m.forward(&mut st, &p[..ctx - 1], HeadMode::None, None).unwrap();
                states.push(st);
                cur.push(p[ctx - 1]);
            }
            let base = ctx - 1;
            for round in 0..=rounds {
                // rotate the start so no kind always follows the same neighbour; round 0 is warm-up (printed with round 0, dropped by the analysis)
                let n = kinds.len();
                for j in 0..n {
                    let (name, kind) = &kinds[(j + round * 5) % n];
                    for rep in 0..reps {
                        let t0 = Instant::now();
                        run(&m, &mut states, &cur, *kind);
                        let ms = t0.elapsed().as_secs_f64() * 1e3;
                        let epoch = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
                        println!("data,{ctx},{name},{round},{rep},{ms:.4},{epoch}");
                        for st in states.iter_mut() {
                            st.truncate(base, &*backend).unwrap();
                        }
                    }
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Kind {
        Plain(usize),
        Verify(usize),
        Iter(usize, usize),
        Vv(usize, usize),
    }

    fn batch(m: &Model, states: &mut [SequenceState], cur: &[u32], b: usize, first: usize) {
        if b == 1 {
            m.forward(&mut states[first], &cur[first..first + 1], HeadMode::LastArgmax, None).unwrap();
        } else {
            let ids: Vec<u32> = cur[first..first + b].to_vec();
            let modes = vec![HeadMode::LastArgmax; b];
            let mut refs: Vec<&mut SequenceState> = states[first..first + b].iter_mut().collect();
            m.forward_batch(&mut refs, &ids, &modes).unwrap();
        }
    }

    fn verify(m: &Model, st: &mut SequenceState, cur: u32, q: usize) {
        let ids: Vec<u32> = (0..q as u32).map(|i| if i == 0 { cur } else { 1000 + i * 37 }).collect();
        m.forward_verify(st, &ids).unwrap();
    }

    fn run(m: &Model, states: &mut [SequenceState], cur: &[u32], kind: Kind) {
        match kind {
            Kind::Plain(b) => batch(m, states, cur, b, 0),
            Kind::Verify(q) => verify(m, &mut states[0], cur[0], q),
            Kind::Iter(b, k) => {
                verify(m, &mut states[0], cur[0], 1 + k);
                batch(m, states, cur, b - 1, 1);
            }
            Kind::Vv(n, k) => {
                for i in 0..n {
                    verify(m, &mut states[i], cur[i], 1 + k);
                }
            }
        }
    }
}
