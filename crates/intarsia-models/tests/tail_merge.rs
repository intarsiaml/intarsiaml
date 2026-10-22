//! `INTARSIA_PREFILL_TAIL_MERGE` end to end on the CPU backend: `Model::load` sizes the workspace for `chunk + rows`, a
//! short remainder past a full chunk runs as one merged pass, and the logits equal the unmerged chunking. The merged
//! chunk is longer than `prefill_chunk()`, so a workspace sized only for `chunk` rows would fail the CPU backend's
//! bounds checks. This is the only test in this binary: it sets a process-wide environment variable.

mod common;
use common::*;
use intarsia_core::model::{HeadMode, LoadOptions};
use intarsia_core::scheduler::TAIL_MERGE_ENV;
use intarsia_core::DType;

fn ids(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % V) as u32).collect()
}

fn all_logits(m: &intarsia_core::model::Model, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut st = m.new_state().unwrap();
    let mut rows = vec![];
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    assert_eq!(st.len, ids.len());
    rows
}

fn rel_l2(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
        num += ((*x - *y) as f64).powi(2);
        den += (*x as f64).powi(2);
    }
    (num / den).sqrt()
}

#[test]
fn tail_merge_runs_the_merged_chunk_in_the_workspace() {
    let t = tiny_weights(DType::F32, 21);
    let opts = LoadOptions { kv_dtype: Some(DType::F32), prefill_chunk: 4, ..Default::default() };
    // 13 tokens at chunk 4: plain 4,4,4,1; merged (rows >= 1) 4,4,5; rows 5: 13 = 4 + 4 + 5 (remainder 5 = chunk + 1)
    // 14 tokens: plain 4,4,4,2; merged with rows 2: 4,4,6 (6 = chunk + 2 rows, the whole workspace)
    std::env::remove_var(TAIL_MERGE_ENV);
    let off = load(&t, &opts);
    assert_eq!(off.tail_merge(), 0);
    assert_eq!(off.prefill_chunk(), 4);
    let want13 = all_logits(&off, &ids(13));
    let want14 = all_logits(&off, &ids(14));

    std::env::set_var(TAIL_MERGE_ENV, "2");
    let on = load(&t, &opts);
    assert_eq!(on.tail_merge(), 2);
    assert_eq!(on.prefill_chunk(), 4, "the prefill chunk (what callers split on) is unchanged");
    let got13 = all_logits(&on, &ids(13));
    let got14 = all_logits(&on, &ids(14));
    assert!(rel_l2(&got13, &want13) < 1e-5, "13 tokens, merged 4,4,5: {}", rel_l2(&got13, &want13));
    assert!(rel_l2(&got14, &want14) < 1e-5, "14 tokens, merged 4,4,6: {}", rel_l2(&got14, &want14));

    // an absurd value is clamped to the chunk (workspace at most 2 * chunk), not an allocation of that many rows
    std::env::set_var(TAIL_MERGE_ENV, "1000000000");
    let big = load(&t, &opts);
    assert_eq!(big.tail_merge(), 4);
    assert!(rel_l2(&all_logits(&big, &ids(14)), &want14) < 1e-5);

    // a malformed value is refused at load
    std::env::set_var(TAIL_MERGE_ENV, "lots");
    let err = intarsia_core::model::Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &opts,
        std::sync::Arc::new(intarsia_core::telemetry::Telemetry::new()),
    );
    assert!(err.is_err());
    std::env::remove_var(TAIL_MERGE_ENV);
}
