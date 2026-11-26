//! Lane ink-sampled (docs/results/2026-11-26-inkling2-sampled-dspark.md): sampled DSpark speculation on the real Inkling-2-Mini, the
//! real-model half of gates 1 and 2. Opt-in, one arm per process (`INTARSIA_SPEC_SAMPLED_DFLASH` is read once per process):
//!
//! * `INTARSIA_TEST_REAL_MODEL=1`, `INTARSIA_INK_MODEL=<checkpoint dir>` (the DSpark draft is its `mtp.safetensors`),
//!   `INTARSIA_INKLING2_BACKEND=metal|hip|cuda|cpu`;
//! * `INTARSIA_SPEC_SAMPLED_DFLASH=0|1` (the arm), `INTARSIA_SS_TEMPS` (temperatures, comma-separated, default `1.0,0.6`; top-k 0,
//!   top-p 1.0: what `serve` gives a request without sampling fields for this checkpoint);
//! * `INTARSIA_SS_SEEDS=<a>..<b>[,<c>..<d>...]` (seed ranges, default `0..20`), `INTARSIA_SS_TOKENS` (default 6), `INTARSIA_SS_OUT`
//!   (appends one line per request: `T seed ids...`).
//!
//! Every request is the same short prompt, `--spec dflash --spec-k 3`, `ignore_eos`. Token 1 comes from prefill; tokens 2.. are the
//! speculative positions. `docs/results/2026-11-26-inkling2-sampled-dspark/raw/law.py` compares the arms (seed equality on shared seeds,
//! chi-square on disjoint ones).

use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{GenEvent, GenRequest};
use std::io::Write;
use std::path::PathBuf;

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn backend() -> std::sync::Arc<dyn intarsia_core::Backend> {
    match var("INTARSIA_INKLING2_BACKEND").as_deref() {
        #[cfg(target_os = "macos")]
        Some("metal") => intarsia_backend_metal::MetalBackend::arc().unwrap(),
        Some("cuda") => intarsia_backend_cuda::create(0).unwrap(),
        Some("hip") => intarsia_backend_hip::create(0).unwrap(),
        Some("cpu") | None => intarsia_backend_cpu::CpuBackend::arc(),
        Some(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu, metal, cuda or hip"),
    }
}

#[test]
fn inkling2_sampled_dspark_real_model_runs() {
    if var("INTARSIA_TEST_REAL_MODEL").as_deref() != Some("1") {
        eprintln!("SKIP: real-model test (set INTARSIA_TEST_REAL_MODEL=1 and INTARSIA_INK_MODEL)");
        return;
    }
    let Some(dir) = var("INTARSIA_INK_MODEL").map(PathBuf::from) else {
        eprintln!("SKIP: INTARSIA_INK_MODEL unset");
        return;
    };
    let temps: Vec<f32> = var("INTARSIA_SS_TEMPS").unwrap_or_else(|| "1.0,0.6".into()).split(',').map(|t| t.trim().parse().unwrap()).collect();
    let seeds: Vec<(u64, u64)> = var("INTARSIA_SS_SEEDS")
        .unwrap_or_else(|| "0..20".into())
        .split(',')
        .map(|r| {
            let (a, b) = r.split_once("..").expect("a seed range a..b");
            (a.trim().parse().unwrap(), b.trim().parse().unwrap())
        })
        .collect();
    let tokens: usize = var("INTARSIA_SS_TOKENS").map_or(6, |v| v.parse().unwrap());
    let out = var("INTARSIA_SS_OUT").map(PathBuf::from);
    let opts = intarsia_models::EngineOptions {
        spec: SpecConfig { mode: SpecMode::Dflash, k: 3, k_min: None },
        spec_draft: Some(dir.clone()),
        spec_k_given: true,
        max_batch: Some(1),
        ..Default::default()
    };
    let t_load = std::time::Instant::now();
    let loaded = intarsia_models::load_engine(&dir, backend(), &opts).unwrap();
    let e = loaded.engine;
    eprintln!("loaded in {:.1} s, INTARSIA_SPEC_SAMPLED_DFLASH={:?}", t_load.elapsed().as_secs_f64(), var("INTARSIA_SPEC_SAMPLED_DFLASH"));
    let prompt = e.tokenize("Once upon a time, in a small village by the sea, there lived", false).unwrap();
    let mut f = out.map(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap());
    let t0 = std::time::Instant::now();
    let mut n = 0usize;
    for &t in &temps {
        for &(a, b) in &seeds {
            for seed in a..b {
                let req = GenRequest {
                    prompt: prompt.clone(),
                    max_tokens: tokens,
                    ignore_eos: true,
                    temperature: t,
                    top_k: 0,
                    top_p: 1.0,
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
                assert!(ids.len() == tokens || (ids.len() == tokens + 1 && ids[tokens] == ids[tokens - 1]), "{ids:?}");
                ids.truncate(tokens);
                let line = format!("{t} {seed} {}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
                match f.as_mut() {
                    Some(f) => writeln!(f, "{line}").unwrap(),
                    None => println!("{line}"),
                }
                n += 1;
            }
        }
    }
    let s = e.telemetry();
    eprintln!(
        "{n} requests in {:.1} s; spec verify passes {} proposed {} accepted {}",
        t0.elapsed().as_secs_f64(),
        s.spec_steps,
        s.spec_proposed,
        s.spec_accepted
    );
}
