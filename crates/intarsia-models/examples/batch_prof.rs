//! Profile target for the batched decode step (Qwen3-8B BF16, Metal): B prefilled sequences stepped by
//! `Model::forward_batch`, with per-step wall times and step start / end marks on the clock the Metal command-buffer
//! trace uses (`CLOCK_UPTIME_RAW`), so `INTARSIA_METAL_PROFILE=cb:<file>` / `kernel:<file>` records can be cut into steps.
//!
//!   INTARSIA_METAL_PROFILE=kernel:k.jsonl cargo run --release -p intarsia-models --example batch_prof -- \
//!       --ctx 1024 --b 8 --steps 40 --warm 8 --marks marks.jsonl
//!
//! Measurement only; it uses the public model API and changes nothing in the engine. Run inside a fleetq slot.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("batch_prof needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
    use intarsia_core::state::SequenceState;
    use intarsia_core::Telemetry;
    use intarsia_models::load::SchemedSource;
    use intarsia_models::{read_model_spec, EngineOptions};
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Instant;

    extern "C" {
        fn clock_gettime_nsec_np(clock_id: u32) -> u64;
    }

    fn now() -> f64 {
        // SAFETY: plain libc call without pointers.
        unsafe { clock_gettime_nsec_np(8) as f64 * 1e-9 }
    }

    fn arg(args: &[String], name: &str, default: &str) -> String {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.into())
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let ctx: usize = arg(&args, "--ctx", "1024").parse().unwrap();
        let b: usize = arg(&args, "--b", "8").parse().unwrap();
        let steps: usize = arg(&args, "--steps", "40").parse().unwrap();
        let warm: usize = arg(&args, "--warm", "8").parse().unwrap();
        let marks = arg(&args, "--marks", "");
        let dir = std::env::var_os("INTARSIA_METAL_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("models/Qwen3-8B"));
        let backend = intarsia_backend_metal::MetalBackend::arc().unwrap();
        let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
        let weights = SchemedSource::open(&dir).unwrap();
        let chunk = recipe.resolve_prefill_chunk(None, "metal").value;
        let opts = LoadOptions { max_seq: ctx + warm + steps + 64, prefill_chunk: chunk, profile_every: 0, ..Default::default() };
        let mut m = Model::load(backend, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
        if b > 1 {
            m.enable_batch(b.max(2)).unwrap();
        }
        m.configure_plan(&Default::default(), None).unwrap();
        let mut states: Vec<SequenceState> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        for i in 0..b {
            let mut rng = intarsia_core::rng::SplitMix64::new(0xbe7c_0000 + i as u64);
            let p: Vec<u32> = (0..ctx).map(|_| 1000 + (rng.next_u64() % 29000) as u32).collect();
            let mut s = m.new_state().unwrap();
            m.forward(&mut s, &p[..ctx - 1], HeadMode::None, None).unwrap();
            states.push(s);
            cur.push(p[ctx - 1]);
        }
        let mut out = (!marks.is_empty()).then(|| std::fs::File::create(&marks).unwrap());
        let mut walls = Vec::new();
        for step in 0..warm + steps {
            let t0 = Instant::now();
            let s = now();
            let modes = vec![HeadMode::LastArgmax; b];
            let mut refs: Vec<&mut SequenceState> = states.iter_mut().collect();
            let outs = if b == 1 {
                vec![m.forward(refs[0], &cur[..1], HeadMode::LastArgmax, None).unwrap()]
            } else {
                m.forward_batch(&mut refs, &cur, &modes).unwrap()
            };
            for (c, o) in cur.iter_mut().zip(&outs) {
                match o {
                    StepOut::Token(t) => *c = *t,
                    _ => panic!("no token"),
                }
            }
            let e = now();
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            if step >= warm {
                walls.push(wall);
            }
            if let Some(f) = out.as_mut() {
                let _ = writeln!(f, "{{\"step\":{step},\"warm\":{},\"s\":{s:.9},\"e\":{e:.9},\"wall_ms\":{wall:.4}}}", step < warm);
            }
        }
        walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("ctx {ctx} B {b}: median step {:.3} ms (min {:.3}, max {:.3}, n {})", walls[walls.len() / 2], walls[0], walls[walls.len() - 1], walls.len());
    }
}
