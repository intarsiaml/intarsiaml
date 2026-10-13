//! The f64 reference math of the GatedDeltaNet ops against HF `transformers` (`torch_chunk_gated_delta_rule`,
//! `torch_recurrent_gated_delta_rule`, `causal_conv1d_fn`), fixture from `tools/gen_hybrid_fixture.py`.

use intarsia_core::gdn::{conv_silu, delta_rule, DeltaDims};
use serde_json::Value;

fn arr(v: &Value) -> Vec<f64> {
    v["data"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn uz(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn cases() -> Vec<Value> {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gdn_hf.json")).unwrap();
    serde_json::from_str::<Value>(&text).unwrap()["cases"].as_array().unwrap().clone()
}

#[test]
fn conv_matches_hf_with_and_without_carried_state() {
    for c in cases() {
        let (t, ks) = (uz(&c["t"]), uz(&c["ks"]));
        let cdim = 2 * uz(&c["hk"]) * uz(&c["dk"]) + uz(&c["hv"]) * uz(&c["dv"]);
        let (x, w) = (arr(&c["conv_x"]), arr(&c["conv_w"]));
        // zero state == HF's left zero padding
        let mut st = vec![0.0; cdim * (ks - 1)];
        let y = conv_silu(&x, &w, &mut st, t, cdim, ks);
        assert!(max_err(&y, &arr(&c["conv_y_zero_state"])) < 1e-5, "zero state, t={t}");
        // carried state: HF `conv_prev` is [C, K-1], oldest first, same layout as the engine
        let mut st = arr(&c["conv_prev"]);
        let y = conv_silu(&x, &w, &mut st, t, cdim, ks);
        assert!(max_err(&y, &arr(&c["conv_y_carry"])) < 1e-5, "carried state, t={t}");
        assert!(max_err(&st, &arr(&c["conv_new_state"])) < 1e-6, "new state, t={t}");
    }
}

#[test]
fn delta_rule_matches_hf_recurrent_and_chunked() {
    for c in cases() {
        let d = DeltaDims { t: uz(&c["t"]), hk: uz(&c["hk"]), hv: uz(&c["hv"]), dk: uz(&c["dk"]), dv: uz(&c["dv"]) };
        let (qkv, a, b) = (arr(&c["qkv"]), arr(&c["a"]), arr(&c["b"]));
        let (a_log, dtb) = (arr(&c["a_log"]), arr(&c["dt_bias"]));
        let mut state = arr(&c["state0"]);
        let out = delta_rule(&d, 1e-6, &qkv, &a, &b, &a_log, &dtb, &mut state);
        // HF runs fp32; the reference is f64
        assert!(max_err(&out, &arr(&c["out"])) < 2e-5, "output, {d:?}: {}", max_err(&out, &arr(&c["out"])));
        assert!(max_err(&state, &arr(&c["state1"])) < 2e-5, "final state, {d:?}");
    }
}
