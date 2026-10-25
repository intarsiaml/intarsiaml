//! `--spec mtp` under tensor parallelism is refused at load, with the vocab-parallel lm_head on or off, before any weight is read
//! (`load_engine_group`, `crates/intarsia-models/src/load.rs`): the MTP drafter head and its recurrent verify state are not sharded, so
//! a group would draft from one rank's slice of the head. cpu backend, host communicator, the tiny hybrid fixture directory (as
//! `batch_recurrent_load.rs`). The control is the same group loaded without MTP, so the refusal is not an unrelated load failure.
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::comm::HostCommGroup;
use intarsia_core::model::LoadOptions;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{Backend, Comm, DType};
use intarsia_models::{load_engine_tp, EngineOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const VOCAB: usize = 96;

fn model_dir() -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("tp_mtp_refused_model");
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

fn group() -> (Vec<Arc<dyn Backend>>, Vec<Arc<dyn Comm>>) {
    let backends: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc() as Arc<dyn Backend>).collect();
    let comms = HostCommGroup::new(backends.clone());
    (backends, comms)
}

fn opts(spec: SpecMode, load_mtp: bool, tp_vocab: bool) -> EngineOptions {
    EngineOptions {
        load: LoadOptions {
            max_seq: 200,
            act_dtype: Some(DType::F32),
            kv_dtype: Some(DType::F32),
            mtp: load_mtp,
            tp_vocab,
            ..Default::default()
        },
        spec: SpecConfig { mode: spec, ..Default::default() },
        ..Default::default()
    }
}

#[test]
fn spec_mtp_with_tp_is_refused_at_load_with_and_without_the_vocab_parallel_head() {
    let dir = model_dir();
    for tp_vocab in [true, false] {
        // `--spec mtp`, and the lower-level `LoadOptions::mtp` the engine also honours
        for (spec, load_mtp) in [(SpecMode::Mtp, false), (SpecMode::Off, true)] {
            let (b, c) = group();
            let err = load_engine_tp(&dir, b, c, &opts(spec, load_mtp, tp_vocab)).err().unwrap_or_else(|| {
                panic!("--spec mtp ran under tp=2 (tp_vocab {tp_vocab}, spec {spec:?}, load.mtp {load_mtp}): it must be refused")
            });
            let msg = err.to_string();
            assert!(msg.contains("--spec mtp") && msg.contains("tensor parallelism") && msg.contains("--tp 2"), "unclear refusal: {msg}");
        }
    }
    // control: the same group without MTP loads (so the refusal above is the MTP check, not a fixture that cannot shard)
    let (b, c) = group();
    let ok = load_engine_tp(&dir, b, c, &opts(SpecMode::Off, false, true));
    assert!(ok.is_ok(), "control load of the tp=2 group failed: {:?}", ok.err().map(|e| e.to_string()));
}
