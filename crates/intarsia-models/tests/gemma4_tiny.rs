//! Gemma 4 end to end on a tiny random-weight checkpoint against HF transformers 5.6.2 (`tools/gemma4_tiny.py`):
//! the CPU backend's logits for three prompts (the sliding window of 8 bites on the longer two), teacher-forced decode against
//! the full prefill, and chunked against whole-prompt prefill.
//!
//! The checkpoint and the reference live in `INTARSIA_GEMMA4_TINY_DIR` (default `golden/gemma4-tiny` under the repository);
//! the test is skipped (loudly) when they are missing, and fails under `CI=1`.

use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::Telemetry;
use intarsia_models::SafeTensors;
use std::path::PathBuf;
use std::sync::Arc;

fn tiny_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("INTARSIA_GEMMA4_TINY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/gemma4-tiny"));
    dir.join("config.json").exists().then_some(dir)
}

fn skip(why: &str) {
    println!("SKIPPED: gemma4_tiny: {why}");
    eprintln!("SKIPPED: gemma4_tiny: {why}");
    if std::env::var("CI").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("gemma4_tiny was SKIPPED ({why}) but CI is set");
    }
}

fn load(dir: &PathBuf, chunk: usize) -> Model {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/gemma4.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert!(dims.gemma.is_some());
    let src = SafeTensors::open_dir(dir).unwrap();
    Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe,
        &dims,
        &src,
        &LoadOptions { max_seq: 128, prefill_chunk: chunk, graphs: false, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

/// f32 tensor `name` of a safetensors file as a flat vector (minimal reader: one dtype, no index).
fn read_f32(path: &std::path::Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap();
    let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hl]).unwrap();
    let t = &h[name];
    assert_eq!(t["dtype"], "F32", "{name}");
    let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let (s, e) = (t["data_offsets"][0].as_u64().unwrap() as usize, t["data_offsets"][1].as_u64().unwrap() as usize);
    let data = b[8 + hl + s..8 + hl + e].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (shape, data)
}

fn all_logits(m: &Model, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut st = m.new_state().unwrap();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    m.forward(&mut st, ids, HeadMode::AllLogits, Some(&mut |_, row: &[f32]| {
        rows.push(row.to_vec());
        Ok(())
    }))
    .unwrap();
    rows
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
fn tiny_gemma4_cpu_matches_hf() {
    let Some(dir) = tiny_dir() else {
        skip("no tiny checkpoint (INTARSIA_GEMMA4_TINY_DIR or golden/gemma4-tiny)");
        return;
    };
    let refj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("ref_greedy.json")).unwrap()).unwrap();
    let prompts: Vec<Vec<u32>> = refj["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect())
        .collect();
    let m = load(&dir, 128);
    let vocab = m.dims.vocab;
    for (i, ids) in prompts.iter().enumerate() {
        let (shape, want) = read_f32(&dir.join("ref_logits.safetensors"), &format!("p{i}"));
        assert_eq!(shape, [ids.len(), vocab], "prompt {i}");
        let got = all_logits(&m, ids);
        assert_eq!(got.len(), ids.len());
        let mut agree = 0;
        let mut worst = 0f64;
        for (r, row) in got.iter().enumerate() {
            let w = &want[r * vocab..(r + 1) * vocab];
            assert!(row.iter().all(|x| x.is_finite()), "prompt {i} row {r}: non-finite logits");
            worst = worst.max(rel_l2(row, w));
            agree += usize::from(argmax(row) == argmax(w));
        }
        println!("prompt {i} ({} tokens): worst row rel-l2 {worst:.3e}, top-1 agreement {agree}/{}", ids.len(), ids.len());
        assert!(worst < 5e-2, "prompt {i}: worst row rel-l2 {worst}");
        assert!(agree * 100 >= ids.len() * 90, "prompt {i}: top-1 agreement {agree}/{}", ids.len());
    }

    // chunked prefill == whole-prompt prefill, and decode steps == the full prefill's rows (window and cache bookkeeping)
    let long = prompts.iter().max_by_key(|p| p.len()).unwrap();
    let whole = all_logits(&m, long);
    let small = load(&dir, 7);
    let chunked = all_logits(&small, long);
    for (r, (a, b)) in whole.iter().zip(&chunked).enumerate() {
        assert!(rel_l2(a, b) < 2e-2, "chunk 7 row {r}: rel-l2 {}", rel_l2(a, b));
    }
    let mut st = m.new_state().unwrap();
    let split = long.len() - 6;
    m.forward(&mut st, &long[..split], HeadMode::None, None).unwrap();
    for (k, &id) in long[split..].iter().enumerate() {
        let StepOut::Logits(l) = m.forward(&mut st, &[id], HeadMode::LastLogits, None).unwrap() else { panic!("no logits") };
        let want = &whole[split + k];
        assert!(rel_l2(&l, want) < 2e-2, "decode step {k}: rel-l2 {}", rel_l2(&l, want));
    }
}
