//! End-to-end check on a real checkpoint. Opt-in: skipped unless INTARSIA_TEST_REAL_MODEL=1 and a
//! model directory is available: set INTARSIA_TEST_MODEL, or have the Qwen3-Reranker-0.6B snapshot in the HF cache.

use intarsia_core::{Engine, GenEvent, GenRequest};
use std::path::PathBuf;

/// Real-model tests are opt-in (INTARSIA_TEST_REAL_MODEL=1).
fn real_model_enabled() -> bool {
    if std::env::var("INTARSIA_TEST_REAL_MODEL").is_ok_and(|v| v == "1") {
        return true;
    }
    eprintln!("SKIP: real-model test (set INTARSIA_TEST_REAL_MODEL=1; on endor run inside an exclusive fleetq slot)");
    false
}

fn model_dir() -> Option<PathBuf> {
    if !real_model_enabled() {
        return None;
    }
    if let Ok(p) = std::env::var("INTARSIA_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots");
    std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path()).find(|p| p.join("config.json").exists())
}

#[test]
fn counts_up_on_real_model() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no test model (set INTARSIA_TEST_MODEL)");
        return;
    };
    let loaded = intarsia_models::load_engine(
        &dir,
        intarsia_backend_cpu::CpuBackend::arc(),
        &intarsia_models::EngineOptions::default(),
    )
    .unwrap();
    let e = loaded.engine;
    let ids = e.tokenize("1, 2, 3, 4, 5,", false).unwrap();
    let mut text = String::new();
    e.generate(GenRequest { prompt: ids, max_tokens: 8, ..Default::default() }, &mut |ev| {
        if let GenEvent::Token { text: t, .. } = ev {
            text.push_str(&t);
        }
    })
    .unwrap();
    assert!(text.contains("6, 7, 8"), "got {text:?}");

    // ~32 token forward pass finishes in well under a minute (debug-profile deps are optimised)
    let long: Vec<u32> = e.tokenize(&"The quick brown fox jumps over the lazy dog. ".repeat(4), false).unwrap();
    let t0 = std::time::Instant::now();
    let rows = e.forward_logits(&long[..32.min(long.len())]).unwrap();
    assert!(!rows.is_empty() && rows[0].len() == 151669);
    assert!(t0.elapsed().as_secs() < 60, "forward took {:?}", t0.elapsed());
}
