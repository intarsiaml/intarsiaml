//! `load_engine` wiring of the hybrid batching opt-in (`INTARSIA_RECURRENT_BATCH=1`, cpu backend): the model directory
//! path the CLI takes (`--max-batch N`), with a real `tokenizer.json`, recipe selection from `config.json` and the
//! batched engine behind `Loaded.engine`. Its own test binary with a single test, because it sets a process environment
//! variable (tests of one binary run in parallel threads and would see each other's setting).
use intarsia_core::model::LoadOptions;
use intarsia_core::{DType, GenEvent, GenRequest};
use intarsia_models::{load_engine, EngineOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const VOCAB: usize = 96;

/// A model directory in cargo's test scratch space: the tiny hybrid fixture plus a word-level `tokenizer.json`.
fn model_dir() -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rb_hybrid_model");
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["config.json", "model.safetensors"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    let vocab: serde_json::Map<String, serde_json::Value> = (0..VOCAB).map(|i| (format!("t{i}"), i.into())).collect();
    let tok = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}
    });
    std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
    dir
}

fn opts(max_batch: Option<usize>) -> EngineOptions {
    EngineOptions {
        load: LoadOptions {
            max_seq: 200,
            act_dtype: Some(DType::F32),
            kv_dtype: Some(DType::F32),
            ..Default::default()
        },
        max_batch,
        ..Default::default()
    }
}

fn ids(e: &Arc<dyn intarsia_core::Engine>, req: GenRequest) -> Vec<u32> {
    let rx = e.submit(req).unwrap();
    let mut out = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
            GenEvent::Token { id, .. } => out.push(id),
            GenEvent::Done { .. } => return out,
            GenEvent::Error { message } => panic!("{message}"),
        }
    }
}

#[test]
fn load_engine_batches_a_hybrid_model_only_with_the_env_opt_in() {
    let dir = model_dir();
    let backend = || intarsia_backend_cpu::CpuBackend::arc();
    let req = |i: u32| GenRequest {
        prompt: (0..6 + i).map(|k| 1 + (k * 7 + i * 3) % 90).collect(),
        max_tokens: 40,
        ignore_eos: true,
        ..Default::default()
    };

    // opt-in off: `--max-batch 3` is refused exactly as before
    std::env::remove_var("INTARSIA_RECURRENT_BATCH");
    let err = load_engine(&dir, backend(), &opts(Some(3))).err().expect("must be refused").to_string();
    assert!(err.contains("row-exact"), "{err}");
    assert!(load_engine(&dir, backend(), &opts(Some(1))).is_ok());

    // the single-sequence engine is the reference
    let solo = load_engine(&dir, backend(), &opts(None)).unwrap();
    assert_eq!(solo.max_batch, 1);
    let want: Vec<Vec<u32>> = (0..3).map(|i| ids(&solo.engine, req(i))).collect();

    // opt-in on: the batched engine over the same directory, three requests in flight
    std::env::set_var("INTARSIA_RECURRENT_BATCH", "1");
    let loaded = load_engine(&dir, backend(), &opts(Some(3))).unwrap();
    assert_eq!(loaded.max_batch, 3);
    let rxs: Vec<_> = (0..3).map(|i| loaded.engine.submit(req(i)).unwrap()).collect();
    for (i, rx) in rxs.iter().enumerate() {
        let mut got = Vec::new();
        loop {
            match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
                GenEvent::Token { id, .. } => got.push(id),
                GenEvent::Done { .. } => break,
                GenEvent::Error { message } => panic!("{message}"),
            }
        }
        assert_eq!(got, want[i], "request {i}");
        assert_eq!(got.len(), 40);
    }
    // a value other than 1 does not opt in
    std::env::set_var("INTARSIA_RECURRENT_BATCH", "0");
    assert!(load_engine(&dir, backend(), &opts(Some(3))).is_err());
    std::env::remove_var("INTARSIA_RECURRENT_BATCH");
}
