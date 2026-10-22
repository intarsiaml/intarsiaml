//! The real Gemma-4-26B-A4B-NVFP4 checkpoint: CUDA against the CPU reference (the lane-g4 stand-in for a golden, because no HF or
//! vLLM reference of this checkpoint exists on the fleet: docs/results/2026-10-22-g4-gemma4-onboard.md).
//!
//! The CPU model is loaded first, its logits kept, and it is dropped before the CUDA model loads (host RAM and GPU share the
//! unified memory of the GB10 boxes). SKIPPED unless `INTARSIA_GEMMA4_DIR` is set and a CUDA device exists (`CI=1` /
//! `INTARSIA_REQUIRE_GPU=1` turn the skip into a failure).

use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::Telemetry;
use intarsia_models::SafeTensors;
use std::path::PathBuf;
use std::sync::Arc;

fn skip(why: &str) {
    println!("SKIPPED: gemma4_real: {why}");
    eprintln!("SKIPPED: gemma4_real: {why}");
    let set = |k: &str| std::env::var(k).is_ok_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"));
    if set("CI") || set("INTARSIA_REQUIRE_GPU") {
        panic!("gemma4_real was SKIPPED ({why}) but this run requires a GPU");
    }
}

fn load(dir: &PathBuf, backend: Arc<dyn intarsia_core::Backend>, chunk: usize) -> Model {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/gemma4.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(dir).unwrap();
    Model::load(
        backend,
        &recipe,
        &dims,
        &src,
        &LoadOptions { max_seq: 256, prefill_chunk: chunk, graphs: false, mem_budget: Some(26 << 30), ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

fn last(m: &Model, st: &mut intarsia_core::state::SequenceState, ids: &[u32]) -> Vec<f32> {
    match m.forward(st, ids, HeadMode::LastLogits, None).unwrap() {
        StepOut::Logits(l) => l,
        _ => panic!("no logits"),
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
fn real_gemma4_cuda_matches_cpu() {
    let Some(dir) = std::env::var_os("INTARSIA_GEMMA4_DIR").map(PathBuf::from) else {
        skip("INTARSIA_GEMMA4_DIR not set");
        return;
    };
    let gpu = match intarsia_backend_cuda::create(0) {
        Ok(b) => b,
        Err(e) => {
            skip(&format!("no usable CUDA device ({e})"));
            return;
        }
    };
    // 1 + 11 ids: the BOS token then ordinary vocabulary ids; a second prompt longer than nothing special (the sliding window is
    // 1024, so the window itself is covered by the tiny-model test and the kernel tests)
    let ids: Vec<u32> = std::iter::once(2u32).chain((0..11u32).map(|i| (i * 7919 + 13) % 200_000 + 1000)).collect();
    let (cpu, cpu_next) = {
        let m = load(&dir, intarsia_backend_cpu::CpuBackend::arc(), 64);
        let mut st = m.new_state().unwrap();
        let a = last(&m, &mut st, &ids);
        let b = last(&m, &mut st, &[4321]);
        (a, b)
    };
    assert!(cpu.iter().all(|v| v.is_finite()));
    let m = load(&dir, gpu, 64);
    let mut st = m.new_state().unwrap();
    let got = last(&m, &mut st, &ids);
    let r = rel_l2(&got, &cpu);
    println!("gemma4 26B prefill cuda vs cpu: rel l2 {r:.3e}, top-1 cuda {} cpu {}", argmax(&got), argmax(&cpu));
    assert!(r < 5e-2, "prefill rel l2 {r}");
    assert_eq!(argmax(&got), argmax(&cpu), "prefill top-1");
    let next = last(&m, &mut st, &[4321]);
    let r2 = rel_l2(&next, &cpu_next);
    println!("gemma4 26B decode cuda vs cpu: rel l2 {r2:.3e}");
    assert!(r2 < 5e-2, "decode rel l2 {r2}");
}
