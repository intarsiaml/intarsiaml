//! Lane ink-sampled (docs/results/2026-11-26-inkling2-sampled-dspark.md): sampled DSpark speculation on the real Inkling-2-Mini, the
//! real-model half of gates 1 and 2. Opt-in, one arm per process (`INTARSIA_SPEC_SAMPLED_DFLASH` is read once per process):
//!
//! * `INTARSIA_TEST_REAL_MODEL=1`, `INTARSIA_INK_MODEL=<checkpoint dir>` (the DSpark draft is its `mtp.safetensors`),
//!   `INTARSIA_INKLING2_BACKEND=metal|hip|cuda|cpu`;
//! * `INTARSIA_SPEC_SAMPLED_DFLASH=0|1` (the arm), `INTARSIA_SS_TEMPS` (temperatures, comma-separated, default `1.0,0.6`; top-k 0,
//!   top-p 1.0: what `serve` gives a request without sampling fields for this checkpoint);
//! * `INTARSIA_SS_SEEDS=<a>..<b>[,<c>..<d>...]` (seed ranges, default `0..20`; one request per seed and temperature, seeds outer),
//!   `INTARSIA_SS_TOKENS` (default 6), `INTARSIA_SS_OUT` (appends one line per request: `T seed ids...`), `INTARSIA_SS_BUDGET_S`
//!   (stop starting requests after this many seconds of generation; unset: no limit);
//! * `INTARSIA_SS_K` (lane ink-sampled-adapt): `--spec-k` (default 3); `INTARSIA_SPEC_SAMPLED_ADAPT` picks the draft length policy as in
//!   `serve` (with `INTARSIA_SPEC_TRACE=1` its `adapt:` lines count the plain stretches and re-entries).
//! * `INTARSIA_SS_DRAFT` (lane ink-hip-specflip): the `--spec-draft` directory (default `INTARSIA_INK_MODEL`, the DSpark draft; e.g. the
//!   DFlash2 drafter `Inkling-2-Mini-DFlash2`).
//! * `INTARSIA_SS_DEFAULT=1` (lane ink-hip-specflip, option (a)): no `--spec` at all, the recipe's speculation default as `serve` and
//!   `generate` apply it (Inkling-2 on HIP: the DSpark default, with the DFlash2 drafter beside it when found); `INTARSIA_SS_K` and
//!   `INTARSIA_SS_DRAFT` are then unused. `INTARSIA_SS_PROMPT_X=<n>` repeats the prompt n times (default 1; lane ink-hip-specflip's
//!   mixed-switch run: a reuse above `DFLASH_RESYNC_REUSE_MAX`). A temperature of 0 is a greedy request.
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
    let temps: Vec<f32> = var("INTARSIA_SS_TEMPS")
        .unwrap_or_else(|| "1.0,0.6".into())
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
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
    let budget = var("INTARSIA_SS_BUDGET_S").map(|v| std::time::Duration::from_secs_f64(v.parse().unwrap()));
    let spec_k: usize = var("INTARSIA_SS_K").map_or(3, |v| v.parse().unwrap());
    let opts = if var("INTARSIA_SS_DEFAULT").is_some_and(|v| v == "1") {
        // `spec_auto`: no `--spec` given, as the CLI sets it (`common.rs`); without it the loader applies no recipe default
        intarsia_models::EngineOptions { max_batch: Some(1), spec_auto: true, ..Default::default() }
    } else {
        intarsia_models::EngineOptions {
            spec: SpecConfig { mode: SpecMode::Dflash, k: spec_k, k_min: None },
            spec_draft: Some(var("INTARSIA_SS_DRAFT").map(PathBuf::from).unwrap_or_else(|| dir.clone())),
            spec_k_given: true,
            max_batch: Some(1),
            ..Default::default()
        }
    };
    let t_load = std::time::Instant::now();
    let be = backend();
    let loaded = intarsia_models::load_engine(&dir, be.clone(), &opts).unwrap();
    let e = loaded.engine;
    eprintln!(
        "loaded in {:.1} s, INTARSIA_SPEC_SAMPLED_DFLASH={:?}, draft {:?}",
        t_load.elapsed().as_secs_f64(),
        var("INTARSIA_SPEC_SAMPLED_DFLASH"),
        var("INTARSIA_SS_DRAFT")
    );
    let prompt = e.tokenize("Once upon a time, in a small village by the sea, there lived", false).unwrap();
    let prompt = prompt.repeat(var("INTARSIA_SS_PROMPT_X").map_or(1, |v| v.parse().unwrap()));
    let mut f = out.map(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap());
    // lane ink-hip-specflip (coordinator 01:03): per-request wall time and time to first token on stderr, and the device's free memory
    // before the first request and after the last (a draft swap frees and allocates a draft state: no creep)
    let free0 = be.info().free_mem;
    eprintln!("free_mem before {} MiB", free0 >> 20);
    let t0 = std::time::Instant::now();
    let mut n = 0usize;
    'all: for &(a, b) in &seeds {
        for seed in a..b {
            for &t in &temps {
                if budget.is_some_and(|b| t0.elapsed() >= b) {
                    eprintln!("budget reached at seed {seed}");
                    break 'all;
                }
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
                let tr = std::time::Instant::now();
                let mut ttft = None;
                e.generate(req, &mut |ev| match ev {
                    GenEvent::Token { id, .. } => {
                        ttft.get_or_insert_with(|| tr.elapsed());
                        ids.push(id)
                    }
                    GenEvent::Error { message } => panic!("engine error: {message}"),
                    _ => {}
                })
                .unwrap();
                // a held-back text tail goes out after the last token as one more `Token` event repeating the last id
                assert!(ids.len() == tokens || (ids.len() == tokens + 1 && ids[tokens] == ids[tokens - 1]), "{ids:?}");
                ids.truncate(tokens);
                eprintln!(
                    "request {n} T {t} seed {seed}: wall {:.1} ms, ttft {:.1} ms, free {} MiB",
                    tr.elapsed().as_secs_f64() * 1e3,
                    ttft.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3),
                    be.info().free_mem >> 20
                );
                let line = format!("{t} {seed} {}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
                match f.as_mut() {
                    Some(f) => writeln!(f, "{line}").unwrap(),
                    None => println!("{line}"),
                }
                n += 1;
            }
        }
    }
    let free1 = be.info().free_mem;
    eprintln!("free_mem after {} MiB (change {:+} MiB)", free1 >> 20, (free1 as i64 - free0 as i64) >> 20);
    let s = e.telemetry();
    eprintln!(
        "{n} requests in {:.1} s; spec verify passes {} proposed {} accepted {}",
        t0.elapsed().as_secs_f64(),
        s.spec_steps,
        s.spec_proposed,
        s.spec_accepted
    );
}
