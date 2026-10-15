//! The bitwise gate of batched decode on the real thing: Qwen3-8B BF16 on the Metal backend. For B in {2, 3, 4, 6, 8}
//! sequences at mixed context lengths (5 to 4000 tokens), greedy and seeded-sampled rows, several consecutive steps,
//! `Model::forward_batch` must return for every row the bits of that sequence's solo decode step (logits compared bit
//! for bit, the on-device argmax token for token) and leave identical state. See `common/batch_identity.rs`.
//!
//! Model: `INTARSIA_METAL_TEST_MODEL`, else `~/models/Qwen3-8B`; the test skips when it is not there. Runs on the GPU:
//! take an exclusive fleetq slot on endor first.
#![cfg(target_os = "macos")]
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::plan::Phase;
use intarsia_core::{Op, Telemetry};
use intarsia_models::load::SchemedSource;
use intarsia_models::{read_model_spec, EngineOptions};
use std::path::PathBuf;
use std::sync::Arc;

#[path = "common/batch_identity.rs"]
mod batch_identity;

fn model_dir() -> PathBuf {
    std::env::var_os("INTARSIA_METAL_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models/Qwen3-8B"))
}

#[test]
fn batch_bitwise_equals_solo_steps_qwen3_8b_metal() {
    let dir = model_dir();
    if !dir.join("config.json").exists() {
        eprintln!("SKIP: Qwen3-8B not found at {}", dir.display());
        return;
    }
    let backend = intarsia_backend_metal::MetalBackend::arc().unwrap();
    let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(&dir).unwrap();
    let chunk = recipe.resolve_prefill_chunk(None, "metal").value;
    let opts = LoadOptions { max_seq: 4352, prefill_chunk: chunk, profile_every: 0, ..Default::default() };
    let mut m = Model::load(backend, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
    assert!(m.batch_capable(), "the metal backend registers the row-exact kernels");
    m.enable_batch(8).unwrap();
    // plan resolution dry-runs every batch size at both ends of the context: a batched op without a row-exact kernel
    // would fail here
    m.configure_plan(&Default::default(), None).unwrap();
    assert!(m.plan.kv_cap_tokens >= 4352, "{} tokens per sequence", m.plan.kv_cap_tokens);

    // the batched pass runs the row-exact multi-row kernels for the projections, and the serving one-row kernels for
    // the per-row attention
    let rec = m.dry_run(Phase::Batch, 8, 1000).unwrap();
    let dec = m.dry_run(Phase::Decode, 1, 1000).unwrap();
    let pick = |rec: &intarsia_core::plan::Recorder, op: Op| -> Vec<String> {
        let mut v: Vec<String> = rec.calls.iter().filter(|c| c.call.op == op).map(|c| c.kernel.clone()).collect();
        v.sort();
        v.dedup();
        v
    };
    for op in [Op::Matmul, Op::MatmulPair, Op::Logits, Op::QkNormRopeKv, Op::AttnDecode] {
        eprintln!("kernels {op}: decode {:?} | batch {:?}", pick(&dec, op), pick(&rec, op));
    }
    // per-row attention and the fold are the decode step's own kernels
    for op in [Op::QkNormRopeKv, Op::AttnDecode] {
        assert_eq!(pick(&dec, op), pick(&rec, op), "{op}");
    }
    let pick = |op: Op| pick(&rec, op);
    assert!(pick(Op::Matmul).iter().all(|k| k.contains("rows")), "{:?}", pick(Op::Matmul));
    assert!(pick(Op::Logits).iter().all(|k| k.contains("rows")));
    assert!(pick(Op::AttnDecode).iter().all(|k| !k.contains("rows")));

    let cfg = batch_identity::Cfg {
        lens: vec![5, 37, 130, 500, 1100, 2000, 3000, 4000],
        batches: vec![2, 3, 4, 6, 8],
        steps: 6,
        vocab_range: (1000, 30000),
    };
    let t0 = std::time::Instant::now();
    let n = batch_identity::run(&m, &cfg);
    eprintln!("{n} (row, step) pairs bitwise identical in {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!(n, (2 + 3 + 4 + 6 + 8) * 6);

    // batched decode steps beside a prefill: 4 decoding sequences, a 1300-token prompt prefilled in 256-token chunks
    let t0 = std::time::Instant::now();
    let n = batch_identity::run_with_prefill(&m, &[37, 130, 500, 1100], 1300, 256, (1000, 30000));
    eprintln!("{n} decode rows bitwise identical beside 256-token prefill chunks, prefill bitwise equal, {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!(n, 4 * 2 * 1300usize.div_ceil(256));
}
