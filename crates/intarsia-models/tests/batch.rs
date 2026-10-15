//! Batched decode (`Model::enable_batch`, `Model::forward_batch`) on the tiny Qwen3-shaped fixture model, cpu backend:
//! the bitwise gate against solo decode steps (the cpu kernels compute every row of a multi-row call with the one-row
//! loop, so this checks the batch plumbing: row placement, per-row positions and caches, state commits, heads), the
//! enable-time checks, KV budgeting and the `--max-batch` resolution. The Qwen3-8B Metal gate with the real row-exact
//! kernels is `batch_metal.rs`.
mod common;
use common::*;
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::plan::Phase;
use intarsia_core::Op;

#[path = "common/batch_identity.rs"]
mod batch_identity;

fn model(opts: &LoadOptions) -> Model {
    load(&tiny_weights(intarsia_core::DType::Bf16, 7), opts)
}

fn batched(rows: usize) -> Model {
    let mut m = model(&LoadOptions { max_seq: 160, ..Default::default() });
    m.enable_batch(rows).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

#[test]
fn batch_bitwise_equals_solo_steps() {
    let m = batched(8);
    let cfg = batch_identity::Cfg {
        lens: vec![3, 7, 12, 20, 33, 47, 61, 90],
        batches: vec![2, 3, 4, 6, 8, 5, 8],
        steps: 6,
        vocab_range: (1, 250),
    };
    let n = batch_identity::run(&m, &cfg);
    assert_eq!(n, (2 + 3 + 4 + 6 + 8 + 5 + 8) * 6);
}

#[test]
fn batch_plan_records_the_batched_pass() {
    let m = batched(4);
    for pos0 in [0, 10, 159] {
        let rec = m.dry_run(Phase::Batch, 4, pos0).unwrap();
        assert!(rec.missing.is_empty());
        let count = |op: Op| rec.calls.iter().filter(|c| c.call.op == op).count();
        // one attention call per row and layer, one embedding for all rows
        assert_eq!(count(Op::AttnDecode), 4 * LAYERS, "pos0 {pos0}");
        assert_eq!(count(Op::Embedding), 1);
        assert!(rec.calls.iter().all(|c| c.phase == Phase::Batch));
        for c in rec.calls.iter().filter(|c| c.call.op == Op::AttnDecode) {
            assert_eq!(c.call.inputs[0].shape[0], 1, "attention is a one-row call");
        }
    }
    // batching off: no batch probes in the plan
    let off = model(&LoadOptions { max_seq: 160, ..Default::default() });
    assert_eq!(off.batch_rows(), 0);
}

#[test]
fn enable_and_forward_checks() {
    let mut m = model(&LoadOptions { max_seq: 160, ..Default::default() });
    let mut st = m.new_state().unwrap();
    // before enable_batch
    assert!(m.forward_batch(&mut [&mut st], &[1], &[HeadMode::LastArgmax]).is_err());
    for bad in [0usize, 1, 9, 17] {
        assert!(m.enable_batch(bad).is_err(), "rows {bad}");
    }
    assert!(!m.batch_capable(), "the cpu backend registers no row-exact kernels");
    m.enable_batch(3).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    assert_eq!(m.batch_rows(), 3);
    let (mut a, mut b, mut c, mut d) =
        (m.new_state().unwrap(), m.new_state().unwrap(), m.new_state().unwrap(), m.new_state().unwrap());
    let am = HeadMode::LastArgmax;
    // more rows than enabled, mismatched lengths, a mode that returns nothing, a token outside the vocabulary
    assert!(m.forward_batch(&mut [&mut a, &mut b, &mut c, &mut d], &[1, 2, 3, 4], &[am; 4]).is_err());
    assert!(m.forward_batch(&mut [&mut a, &mut b], &[1], &[am; 2]).is_err());
    assert!(m.forward_batch(&mut [&mut a, &mut b], &[1, 2], &[am]).is_err());
    assert!(m.forward_batch(&mut [&mut a, &mut b], &[1, 2], &[am, HeadMode::None]).is_err());
    assert!(m.forward_batch(&mut [&mut a, &mut b], &[1, V as u32], &[am; 2]).is_err());
    // nothing was committed by the rejected calls
    assert_eq!((a.len, b.len), (0, 0));
    // a full state is rejected before any row runs
    let cap = b.cap;
    m.forward(&mut b, &vec![1u32; cap], HeadMode::None, None).unwrap();
    assert!(m.forward_batch(&mut [&mut a, &mut b], &[1, 2], &[am; 2]).is_err());
    assert_eq!(a.len, 0);
    // one row is the solo step
    let one = m.forward_batch(&mut [&mut a], &[5], &[am]).unwrap();
    assert_eq!(a.len, 1);
    let mut a2 = m.new_state().unwrap();
    let solo = m.forward(&mut a2, &[5], am, None).unwrap();
    assert_eq!(format!("{:?}", one[0]), format!("{solo:?}"));
}

#[test]
fn kv_budget_is_shared_by_the_sequences() {
    let full = model(&LoadOptions { max_seq: 160, ..Default::default() });
    let bpt = full.kv_bytes_per_token();
    assert!(bpt > 0);
    // a memory budget that holds about 20 tokens of 8 sequences
    let mb = full.plan.weights_bytes + full.plan.workspace_bytes + 8 * 20 * bpt;
    let mut m = model(&LoadOptions { max_seq: 160, mem_budget: Some(mb), ..Default::default() });
    let before = m.plan.kv_cap_tokens;
    m.enable_batch(8).unwrap();
    let cap = m.plan.kv_cap_tokens;
    assert!(cap < before.min(160) || before < 160, "{cap} vs {before}");
    assert_eq!(cap, (m.plan.kv_budget_bytes / (8 * bpt)).min(before as u64) as usize);
    assert_eq!(m.plan.kv_total_bytes, 8 * cap as u64 * bpt);
    assert!(m.plan.notes.iter().any(|n| n.contains("share the KV budget")) || cap == before);
    assert_eq!(m.new_state().unwrap().cap, cap);
    // a budget too small for 16 tokens each is refused
    let mut tight = model(&LoadOptions { max_seq: 160, mem_budget: Some(full.plan.weights_bytes + full.plan.workspace_bytes + 40 * bpt), ..Default::default() });
    assert!(tight.enable_batch(8).is_err());
}

#[test]
fn max_batch_resolution() {
    use intarsia_models::select_recipe;
    let q3 = select_recipe(&serde_json::json!({"architectures": ["Qwen3ForCausalLM"]}), &[]).unwrap().1;
    // recipe default 1
    let d = q3.resolve_max_batch(None, 8).unwrap();
    assert_eq!(d.value, 1);
    assert!(q3.resolve_max_batch(Some(1), 8).is_ok());
    assert!(q3.resolve_max_batch(Some(0), 8).is_err());
    // above the recipe's max_batch
    assert!(q3.resolve_max_batch(Some(4), 8).unwrap_err().to_string().contains("exceeds"));
    // above what the engine implements
    assert!(q3.resolve_max_batch(Some(2), 1).unwrap_err().to_string().contains("not implemented yet"));
    // a recipe that declares more than the engine does (qwen3_5: 32): the default is clamped, not an error
    let q35 = select_recipe(&serde_json::json!({"architectures": ["Qwen3_5ForCausalLM"]}), &[]).unwrap().1;
    let d = q35.resolve_max_batch(None, 1).unwrap();
    assert_eq!(d.value, 1);
    assert!(d.source.contains("clamped to 1"), "{}", d.source);
    assert_eq!(q35.resolve_max_batch(None, 64).unwrap().value, 32);
    assert!(q35.resolve_max_batch(Some(2), 1).is_err());
    assert_eq!(q35.resolve_max_batch(Some(8), 8).unwrap().value, 8);
    assert!(q35.resolve_max_batch(Some(33), 64).is_err());
    // what serving implements today
    assert_eq!(intarsia_core::engine::SERVE_MAX_BATCH, 1);
}
