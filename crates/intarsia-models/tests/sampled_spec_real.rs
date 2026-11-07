//! Sampled speculation (`INTARSIA_SPEC_SAMPLED`) on a real checkpoint with an MTP head, cpu backend: the statistical half of
//! the exactness gate (docs/results/2026-11-05-sampled-spec.md). Opt-in and driven by environment variables, one arm per
//! process (the switch is read once per process):
//!
//! * `INTARSIA_TEST_REAL_MODEL=1`, `INTARSIA_SS_MODEL=<dir>` (a model with `mtp.*` weights, e.g. Qwen3.5-0.8B-Base);
//! * `INTARSIA_SPEC_SAMPLED=0|reject|match` (the arm), `INTARSIA_SS_CFG=A|B` (A: temperature 1.0, full draft head;
//!   B: temperature 0.7, top-k 20, top-p 0.8, draft head reduced to the first 4096 ids);
//! * `INTARSIA_SS_SEED0`, `INTARSIA_SS_N` (seeds `seed0 .. seed0 + n`), `INTARSIA_SS_OUT` (one line per run: the generated ids).
//!
//! Each run is the fixed prompt, `max_tokens` k + 2 = 4 with `ignore_eos`, its own seed. Token 1 comes from prefill (the
//! plain sampler in every arm); tokens 2..4 are the speculative positions. `scripts/sampled-spec/gtest.py` compares the arms.

use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{GenEvent, GenRequest};
use std::io::Write;
use std::path::PathBuf;

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

#[test]
fn sampled_spec_real_model_runs() {
    if var("INTARSIA_TEST_REAL_MODEL").as_deref() != Some("1") {
        eprintln!("SKIP: real-model test (set INTARSIA_TEST_REAL_MODEL=1 and INTARSIA_SS_MODEL)");
        return;
    }
    let Some(dir) = var("INTARSIA_SS_MODEL").map(PathBuf::from) else {
        eprintln!("SKIP: INTARSIA_SS_MODEL unset");
        return;
    };
    let cfg = var("INTARSIA_SS_CFG").unwrap_or_else(|| "A".into());
    let (temperature, top_k, top_p, draft_vocab) = match cfg.as_str() {
        "A" => (1.0f32, 0usize, 1.0f32, 0usize),
        "B" => (0.7, 20, 0.8, 4096),
        other => panic!("INTARSIA_SS_CFG={other} (A or B)"),
    };
    let seed0: u64 = var("INTARSIA_SS_SEED0").map_or(0, |v| v.parse().unwrap());
    let n: u64 = var("INTARSIA_SS_N").map_or(10, |v| v.parse().unwrap());
    let out = var("INTARSIA_SS_OUT").map(PathBuf::from);
    let k = 2;
    let mut opts = intarsia_models::EngineOptions {
        spec: SpecConfig { mode: SpecMode::Mtp, k, k_min: None },
        max_batch: Some(1),
        ..Default::default()
    };
    opts.load.mtp = true;
    opts.load.mtp_draft_vocab = Some(draft_vocab);
    opts.load.max_seq = 256;
    let loaded = intarsia_models::load_engine(&dir, intarsia_backend_cpu::CpuBackend::arc(), &opts).unwrap();
    let e = loaded.engine;
    let prompt = e.tokenize("Once upon a time, in a small village by the sea, there lived", false).unwrap();
    let mut f = out.map(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap());
    let t0 = std::time::Instant::now();
    for seed in seed0..seed0 + n {
        let req = GenRequest {
            prompt: prompt.clone(),
            max_tokens: k + 2,
            ignore_eos: true,
            temperature,
            top_k,
            top_p,
            seed: Some(seed),
            ..Default::default()
        };
        let mut ids = Vec::new();
        e.generate(req, &mut |ev| match ev {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Error { message } => panic!("engine error: {message}"),
            _ => {}
        })
        .unwrap();
        // a held-back text tail goes out after the last token as one more `Token` event repeating the last id
        assert!(ids.len() == k + 2 || (ids.len() == k + 3 && ids[k + 2] == ids[k + 1]), "{ids:?}");
        ids.truncate(k + 2);
        let line = ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
        match f.as_mut() {
            Some(f) => writeln!(f, "{seed} {line}").unwrap(),
            None => println!("{seed} {line}"),
        }
    }
    let t = e.telemetry();
    eprintln!(
        "cfg {cfg} seeds {seed0}..{} in {:.1} s; spec steps {} proposed {} accepted {}",
        seed0 + n,
        t0.elapsed().as_secs_f64(),
        t.spec_steps,
        t.spec_proposed,
        t.spec_accepted
    );
}
