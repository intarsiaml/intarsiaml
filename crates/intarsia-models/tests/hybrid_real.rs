//! The real Qwen3.6-35B-A3B-NVFP4 attention-side weights (FP8 projections, GatedDeltaNet and gated attention at
//! the 35B shapes: 16 key / 32 value heads of 128, conv width 8192, 16 q heads of 256 with partial rotary 64)
//! through the generic decoder, cuda vs the cpu backend. The MoE FFN is left out (layers 0..4 only, pattern
//! [gdn] / [attn]), so the hidden state is not a trained model's, but both backends see identical inputs.
//!
//! Prints SKIPPED (and fails under `CI=1` / `INTARSIA_REQUIRE_GPU=1`) unless `INTARSIA_QWEN36_DIR` points at the checkpoint directory and a CUDA device exists.

use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::Telemetry;
use intarsia_models::SafeTensors;
use std::path::PathBuf;
use std::sync::Arc;

const RECIPE: &str = r#"
[model]
match.architectures = ["Qwen3_5MoeForConditionalGeneration"]
family = "qwen3_5_moe_attention_only"
norm_offset = 1.0
[layer]
pattern = ["attn"]
pattern_by_layer = "layer_types"
[layer.by_type]
full_attention = ["attn"]
linear_attention = ["gdn"]
[layer.attn]
kind = "gqa"
gated = true
q = "model.language_model.layers.{i}.self_attn.q_proj"
k = "model.language_model.layers.{i}.self_attn.k_proj"
v = "model.language_model.layers.{i}.self_attn.v_proj"
o = "model.language_model.layers.{i}.self_attn.o_proj"
q_norm = "model.language_model.layers.{i}.self_attn.q_norm"
k_norm = "model.language_model.layers.{i}.self_attn.k_norm"
[layer.gdn]
kind = "gated_delta_net"
in_qkv = "model.language_model.layers.{i}.linear_attn.in_proj_qkv"
in_z = "model.language_model.layers.{i}.linear_attn.in_proj_z"
in_b = "model.language_model.layers.{i}.linear_attn.in_proj_b"
in_a = "model.language_model.layers.{i}.linear_attn.in_proj_a"
conv = "model.language_model.layers.{i}.linear_attn.conv1d"
a_log = "model.language_model.layers.{i}.linear_attn.A_log"
dt_bias = "model.language_model.layers.{i}.linear_attn.dt_bias"
norm = "model.language_model.layers.{i}.linear_attn.norm"
out = "model.language_model.layers.{i}.linear_attn.out_proj"
[layer.norms]
pre_attn = "model.language_model.layers.{i}.input_layernorm"
pre_gdn = "model.language_model.layers.{i}.input_layernorm"
[top]
embed = "model.language_model.embed_tokens"
final_norm = "model.language_model.norm"
lm_head = { tensor = "no_such_lm_head", tied_to = "model.language_model.embed_tokens" }
"#;

fn load(dir: &PathBuf, backend: Arc<dyn intarsia_core::Backend>, chunk: usize) -> Model {
    let recipe = intarsia_models::parse_recipe(RECIPE).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let mut dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert_eq!(dims.linear.unwrap().conv_dim(), 8192);
    dims.n_layers = 4; // linear, linear, linear, full
    dims.tie_embeddings = true; // the real head is its own tensor; this test stubs it with the embedding table
    dims.layer_types.truncate(4);
    assert_eq!(dims.layer_types, ["linear_attention", "linear_attention", "linear_attention", "full_attention"]);
    let src = SafeTensors::open_dir(dir).unwrap();
    Model::load(
        backend,
        &recipe,
        &dims,
        &src,
        &LoadOptions { max_seq: 512, prefill_chunk: chunk, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

fn last_logits(m: &Model, st: &mut intarsia_core::state::SequenceState, ids: &[u32]) -> Vec<f32> {
    match m.forward(st, ids, HeadMode::LastLogits, None).unwrap() {
        StepOut::Logits(l) => l,
        _ => panic!("no logits"),
    }
}

/// An honest skip: `SKIPPED:` on stdout and stderr, a failure when `CI=1` / `INTARSIA_REQUIRE_GPU=1`.
fn skip(why: &str) {
    println!("SKIPPED: real_35b_attention_layers_cuda_matches_cpu: {why}");
    eprintln!("SKIPPED: real_35b_attention_layers_cuda_matches_cpu: {why}");
    let set = |k: &str| std::env::var(k).is_ok_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"));
    if set("CI") || set("INTARSIA_REQUIRE_GPU") {
        panic!("test was SKIPPED ({why}) but this run requires a GPU (CI / INTARSIA_REQUIRE_GPU is set)");
    }
}

#[test]
fn real_35b_attention_layers_cuda_matches_cpu() {
    let Some(dir) = std::env::var_os("INTARSIA_QWEN36_DIR").map(PathBuf::from) else {
        skip("INTARSIA_QWEN36_DIR not set");
        return;
    };
    let gpu = match intarsia_backend_cuda::create(0) {
        Ok(b) => b,
        Err(e) => {
            skip(&format!("no usable CUDA device ({e})"));
            return;
        }
    };
    let ids: Vec<u32> = (0..37u32).map(|i| (i * 7919 + 13) % 200_000).collect();
    let cpu_m = load(&dir, intarsia_backend_cpu::CpuBackend::arc(), 512);
    let cpu = last_logits(&cpu_m, &mut cpu_m.new_state().unwrap(), &ids);
    assert!(cpu.iter().all(|v| v.is_finite()));
    for chunk in [512, 8] {
        let m = load(&dir, gpu.clone(), chunk);
        let mut st = m.new_state().unwrap();
        let got = last_logits(&m, &mut st, &ids);
        let r = rel_l2(&got, &cpu);
        println!("35B attention layers, cuda prefill (chunk {chunk}) vs cpu: rel l2 {r:.3e}");
        assert!(r < 5e-2, "chunk {chunk}: rel l2 {r}");
        // decode continues from the prefix state
        let more = last_logits(&m, &mut st, &[1234]);
        let mut st2 = cpu_m.new_state().unwrap();
        let mut all = ids.clone();
        all.push(1234);
        let want = last_logits(&cpu_m, &mut st2, &all);
        let r2 = rel_l2(&more, &want);
        println!("35B attention layers, cuda decode vs cpu full prefill: rel l2 {r2:.3e}");
        assert!(r2 < 5e-2, "decode rel l2 {r2}");
    }
}
