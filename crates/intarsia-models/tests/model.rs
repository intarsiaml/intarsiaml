mod common;
use common::*;
use intarsia_core::model::{HeadMode, LoadOptions, StepOut};
use intarsia_core::state::{SequenceState, SlotSpec};
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
    rows
}

#[test]
fn decoder_matches_naive_reference_f32() {
    let t = tiny_weights(DType::F32, 5);
    // f32 KV cache too, otherwise the bf16 cache dominates the error
    let m = load(&t, &LoadOptions { kv_dtype: Some(DType::F32), ..Default::default() });
    let input = ids(9);
    let got = all_logits(&m, &input);
    let want = naive_logits(&t, &input);
    assert_eq!(got.len(), want.len());
    for (p, (g, w)) in got.iter().zip(&want).enumerate() {
        let num: f64 = g.iter().zip(w).map(|(a, b)| (*a as f64 - b).powi(2)).sum();
        let den: f64 = w.iter().map(|b| b * b).sum();
        assert!((num / den).sqrt() < 1e-4, "pos {p}: rel_l2 {}", (num / den).sqrt());
    }
}

#[test]
fn decoder_close_to_naive_reference_bf16() {
    let t = tiny_weights(DType::Bf16, 6);
    let m = load(&t, &LoadOptions::default());
    let input = ids(7);
    let got = all_logits(&m, &input);
    let want = naive_logits(&t, &input);
    for (p, (g, w)) in got.iter().zip(&want).enumerate() {
        let num: f64 = g.iter().zip(w).map(|(a, b)| (*a as f64 - b).powi(2)).sum();
        let den: f64 = w.iter().map(|b| b * b).sum();
        assert!((num / den).sqrt() < 3e-2, "pos {p}: rel_l2 {}", (num / den).sqrt());
    }
}

#[test]
fn prefill_equals_prefill_plus_decode_and_chunking() {
    // KV / position bookkeeping: full prefill vs prefill(N-1) + one decode step vs tiny chunks.
    let t = tiny_weights(DType::F32, 7);
    let input = ids(13);
    let f32kv = LoadOptions { kv_dtype: Some(DType::F32), ..Default::default() };
    let m = load(&t, &f32kv);
    let full = all_logits(&m, &input).pop().unwrap();

    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &input[..12], HeadMode::None, None).unwrap();
    let StepOut::Logits(dec) = m.forward(&mut st, &input[12..], HeadMode::LastLogits, None).unwrap() else { panic!() };
    assert_eq!(st.len, 13);

    let m4 = load(&t, &LoadOptions { prefill_chunk: 4, ..f32kv.clone() });
    let chunked = all_logits(&m4, &input).pop().unwrap();

    for (name, other) in [("decode", &dec), ("chunked", &chunked)] {
        let num: f64 = full.iter().zip(other.iter()).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum();
        let den: f64 = full.iter().map(|a| (*a as f64).powi(2)).sum();
        assert!((num / den).sqrt() < 1e-5, "{name}: rel_l2 {}", (num / den).sqrt());
    }

    // greedy argmax path agrees with logits path
    let mut st2 = m.new_state().unwrap();
    let StepOut::Token(tok) = m.forward(&mut st2, &input, HeadMode::LastArgmax, None).unwrap() else { panic!() };
    let best = full.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(tok as usize, best);
}

#[test]
fn context_overflow_and_truncate() {
    let t = tiny_weights(DType::F32, 8);
    let m = load(&t, &LoadOptions { max_seq: 32, ..Default::default() });
    let mut st = m.new_state().unwrap();
    assert_eq!(st.cap, 32);
    assert!(m.forward(&mut st, &ids(40), HeadMode::None, None).is_err());
    m.forward(&mut st, &ids(10), HeadMode::None, None).unwrap();
    st.truncate(4, &*m.backend).unwrap();
    assert_eq!((st.len, st.tokens.len()), (4, 4));
}

#[test]
fn recurrent_slot_is_allocated_but_inert() {
    let b = intarsia_backend_cpu::CpuBackend::arc();
    let specs =
        [SlotSpec::AttnKv { n_kv_heads: 2, head_dim: 8, dtype: DType::Bf16 }, SlotSpec::Recurrent { bytes: 4096 }];
    let mut st = SequenceState::new(&*b, &specs, 16).unwrap();
    assert!(st.has_recurrent());
    assert_eq!(st.total_bytes(), 2 * 2 * 16 * 8 * 2 + 4096);
    st.commit(&[1, 2, 3]);
    // recurrent state cannot roll back to a non-zero prefix
    assert!(st.truncate(1, &*b).is_err());
    st.reset(&*b).unwrap();
    assert_eq!(st.len, 0);
}
