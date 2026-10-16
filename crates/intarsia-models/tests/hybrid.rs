//! The hybrid decoder (GatedDeltaNet + gated full attention, zero-centred norms, partial rotary) against HF.
//!
//! `tests/fixtures/tiny_qwen3_5/` is a random tiny `Qwen3_5ForCausalLM` (6 layers: linear, linear, linear,
//! full, linear, full; fp32) with the logits and per-layer hidden states HF produced for a fixed 26-token
//! prompt (`tools/gen_hybrid_fixture.py`). The cpu backend runs it in f32 through the recipe-driven decoder:
//! one-shot prefill, chunked prefill (the conv and delta-rule state must carry across chunk boundaries),
//! and prefill followed by token-by-token decode, which all have to reproduce HF.

use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::{DType, Telemetry};
use intarsia_models::SafeTensors;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

/// name -> (shape, values) of a safetensors file with f32 / i32 tensors.
fn read_safetensors(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
    let data = std::fs::read(path).unwrap();
    let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&data[8..8 + hlen]).unwrap();
    let mut out = BTreeMap::new();
    for (name, v) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let off = v["data_offsets"].as_array().unwrap();
        let (b, e) = (8 + hlen + off[0].as_u64().unwrap() as usize, 8 + hlen + off[1].as_u64().unwrap() as usize);
        let vals: Vec<f32> = match v["dtype"].as_str().unwrap() {
            "F32" => data[b..e].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "I32" => data[b..e].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            other => panic!("dtype {other}"),
        };
        out.insert(name.clone(), (shape, vals));
    }
    out
}

fn load(chunk: usize) -> Model {
    let dir = fixture_dir();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&dir).unwrap();
    Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe,
        &dims,
        &src,
        &LoadOptions {
            max_seq: 64,
            prefill_chunk: chunk,
            act_dtype: Some(DType::F32),
            kv_dtype: Some(DType::F32),
            ..Default::default()
        },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

fn golden() -> (Vec<u32>, Vec<Vec<f32>>) {
    let g = read_safetensors(&fixture_dir().join("golden.safetensors"));
    let ids: Vec<u32> = g["ids"].1.iter().map(|&x| x as u32).collect();
    let (shape, flat) = &g["logits"];
    let rows = flat.chunks(shape[1]).map(|r| r.to_vec()).collect();
    (ids, rows)
}

/// max |got - want| over a row, and the relative L2 error.
fn compare(got: &[f32], want: &[f32]) -> (f32, f64) {
    let max = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    (max, (num / den).sqrt())
}

fn check_rows(label: &str, first_pos: usize, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len() - first_pos, "{label}: row count");
    let (mut worst_abs, mut worst_rel) = (0f32, 0f64);
    for (i, g) in got.iter().enumerate() {
        let (max, rel) = compare(g, &want[first_pos + i]);
        assert!(max < 1e-3 && rel < 1e-4, "{label}: position {}: max abs {max}, rel l2 {rel}", first_pos + i);
        worst_abs = worst_abs.max(max);
        worst_rel = worst_rel.max(rel);
    }
    println!("{label}: {} rows, worst max abs {worst_abs:.2e}, worst rel l2 {worst_rel:.2e}", got.len());
}

fn all_rows(m: &Model, st: &mut intarsia_core::state::SequenceState, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(st, ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    rows
}

#[test]
fn layer_kinds_follow_the_config() {
    let m = load(64);
    let kinds: Vec<_> = m.layer_kinds().into_iter().map(|l| l[0]).collect();
    assert_eq!(
        kinds,
        ["gated_delta_net", "gated_delta_net", "gated_delta_net", "attention", "gated_delta_net", "attention"]
    );
    // 4 linear layers hold recurrent state, the 2 full-attention layers a KV cache
    let st = m.new_state().unwrap();
    assert_eq!(st.slots.len(), 6);
    assert!(st.has_recurrent());
}

#[test]
fn one_shot_prefill_matches_hf() {
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    let rows = all_rows(&m, &mut st, &ids);
    check_rows("prefill", 0, &rows, &want);
    assert_eq!(st.len, ids.len());
}

#[test]
fn chunked_prefill_matches_hf() {
    let (ids, want) = golden();
    // 26 tokens in chunks of 3, then 5: the recurrent and conv state cross every boundary
    for chunk in [3, 5, 7] {
        let m = load(chunk);
        let mut st = m.new_state().unwrap();
        let rows = all_rows(&m, &mut st, &ids);
        check_rows(&format!("chunk {chunk}"), 0, &rows, &want);
    }
}

#[test]
fn prefill_then_token_by_token_decode_matches_hf() {
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    let k = 17;
    let pre = all_rows(&m, &mut st, &ids[..k]);
    check_rows("prefill part", 0, &pre, &want[..k]);
    let mut dec = Vec::new();
    for i in k..ids.len() {
        match m.forward(&mut st, &ids[i..i + 1], HeadMode::LastLogits, None).unwrap() {
            StepOut::Logits(l) => dec.push(l),
            _ => panic!("no logits"),
        }
    }
    check_rows("decode", k, &dec, &want);
}

#[test]
fn argmax_decode_path_agrees() {
    // the on-device argmax head (what generation uses) picks the same token as the logits path
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &ids[..20], HeadMode::None, None).unwrap();
    match m.forward(&mut st, &ids[20..21], HeadMode::LastArgmax, None).unwrap() {
        StepOut::Token(t) => {
            let best = want[20].iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            assert_eq!(t as usize, best);
        }
        _ => panic!("no token"),
    }
}

#[test]
fn recurrent_snapshots_restore_exactly() {
    // Prefix reuse for hybrid models: snapshot at 12 tokens, run on to 26, roll back to 12, replay -> identical.
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &ids[..12], HeadMode::None, None).unwrap();
    st.snapshot(&*m.backend, 2).unwrap();
    let rows = all_rows(&m, &mut st, &ids[12..]);
    check_rows("first pass", 12, &rows, &want);
    // only lengths that were snapshotted (or 0, or the current length) can be returned to
    assert_eq!(st.restorable_len(20), 12);
    assert_eq!(st.restorable_len(11), 0);
    assert_eq!(st.restorable_len(26), 26);
    assert!(st.truncate(15, &*m.backend).is_err());
    st.truncate(12, &*m.backend).unwrap();
    assert_eq!((st.len, st.tokens.len()), (12, 12));
    let again = all_rows(&m, &mut st, &ids[12..]);
    check_rows("after restore", 12, &again, &want);
    assert_eq!(rows, again, "replaying from a restored snapshot must be bit-identical");
    // truncating to 0 zeroes the state: the full prompt reproduces HF again
    st.truncate(0, &*m.backend).unwrap();
    let full = all_rows(&m, &mut st, &ids);
    check_rows("after reset", 0, &full, &want);
}

#[test]
fn failed_forward_resets_recurrent_state() {
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &ids[..10], HeadMode::None, None).unwrap();
    // token id out of vocabulary: the pass fails, and the sequence must not keep half-updated state
    assert!(m.forward(&mut st, &[9999], HeadMode::None, None).is_err());
    assert_eq!(st.len, 0);
    let rows = all_rows(&m, &mut st, &ids);
    check_rows("after failed pass", 0, &rows, &want);
}

// ---------------------------------------------------------------------------
// engine level: prefix reuse with recurrent state
// ---------------------------------------------------------------------------

use intarsia_core::engine::{ChatOpts, TextCodec};
use intarsia_core::{Engine, EngineMeta, GenEvent, GenRequest, LocalEngine};

/// ids are rendered as `<id>`; the tiny model's vocabulary is 96.
struct IdCodec;

impl TextCodec for IdCodec {
    fn encode(&self, text: &str, _s: bool) -> intarsia_core::Result<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32 % 96).collect())
    }
    fn decode(&self, ids: &[u32], _s: bool) -> intarsia_core::Result<String> {
        Ok(ids.iter().map(|i| format!("<{i}>")).collect())
    }
    fn apply_chat_template(&self, _m: &[serde_json::Value], _o: &ChatOpts) -> intarsia_core::Result<String> {
        Ok(String::new())
    }
    fn has_chat_template(&self) -> bool {
        false
    }
}

fn engine(chunk: usize) -> LocalEngine {
    let m = load(chunk);
    let tel = Arc::new(Telemetry::new());
    LocalEngine::new(m, Arc::new(IdCodec), EngineMeta { id: "tiny-hybrid".into(), ..Default::default() }, tel).unwrap()
}

fn run(e: &LocalEngine, prompt: &[u32], n: usize) -> (Vec<u32>, usize) {
    let req = GenRequest { prompt: prompt.to_vec(), max_tokens: n, ignore_eos: true, ..Default::default() };
    let (mut ids, mut cached) = (vec![], 0);
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Done { cached_tokens, .. } => cached = cached_tokens,
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    (ids, cached)
}

#[test]
fn engine_reuses_prefix_through_snapshots_and_matches_a_fresh_engine() {
    let (ids, _) = golden();
    let prompt: Vec<u32> = ids[..22].to_vec();
    let e = engine(8); // 22-token prompt: chunk snapshots at 8 and 16, then the one before the end (21)
    let (a, cached) = run(&e, &prompt, 6);
    assert_eq!(cached, 0);
    // same prompt again: everything but the last token comes from the snapshot at 21
    let (b, cached) = run(&e, &prompt, 6);
    assert_eq!(a, b);
    assert_eq!(cached, 21);
    // diverges after 18 tokens: the nearest snapshot at or below 18 is 16 (or 8 once evicted); never a wrong state
    let mut other = prompt[..18].to_vec();
    other.extend_from_slice(&[5, 6, 7, 8, 9, 10]);
    let (c, cached) = run(&e, &other, 6);
    assert_eq!(cached, 16);
    let (c_fresh, cached_fresh) = run(&engine(8), &other, 6);
    assert_eq!(c, c_fresh, "reuse must not change the result");
    assert_eq!(cached_fresh, 0);
    // a prompt that shares only 3 tokens has no snapshot to return to: full recompute, same result
    let mut short = prompt[..3].to_vec();
    short.extend_from_slice(&[40, 41, 42, 43, 44]);
    let (d, cached) = run(&e, &short, 5);
    assert_eq!(cached, 0);
    assert_eq!(d, run(&engine(8), &short, 5).0);
}

#[test]
fn engine_continues_a_conversation_without_recomputing_the_history() {
    let (ids, _) = golden();
    let e = engine(8);
    let first: Vec<u32> = ids[..14].to_vec();
    let (gen1, _) = run(&e, &first, 5);
    // next turn = previous prompt + the reply + new user tokens
    let mut second = first.clone();
    second.extend_from_slice(&gen1);
    second.extend_from_slice(&[30, 31, 32]);
    let (g2, cached) = run(&e, &second, 5);
    // the state holds prompt + reply except its last token (never forwarded): all of it is reused
    assert_eq!(cached, first.len() + gen1.len() - 1);
    assert_eq!(g2, run(&engine(8), &second, 5).0);
}

// ---------------------------------------------------------------------------
// CUDA (skipped without a device): the same tiny hybrid model in bf16
// ---------------------------------------------------------------------------

fn load_bf16(backend: Arc<dyn intarsia_core::Backend>, chunk: usize) -> Model {
    let dir = fixture_dir();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let mut w = intarsia_core::weights::MemWeights::new();
    for (name, (shape, vals)) in read_safetensors(&dir.join("model.safetensors")) {
        w.insert(&name, intarsia_core::HostTensor::from_f32_as(DType::Bf16, &shape, &vals));
    }
    Model::load(
        backend,
        &recipe,
        &dims,
        &w,
        &LoadOptions { max_seq: 64, prefill_chunk: chunk, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

fn cuda() -> Option<Arc<dyn intarsia_core::Backend>> {
    match intarsia_backend_cuda::create(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable CUDA device ({e})");
            None
        }
    }
}

fn rel_l2_rows(got: &[Vec<f32>], want: &[Vec<f32>]) -> f64 {
    got.iter().zip(want).map(|(g, w)| compare(g, w).1).fold(0.0, f64::max)
}

#[test]
fn cuda_bf16_matches_cpu_bf16_and_hf() {
    let Some(gpu) = cuda() else { return };
    let (ids, want) = golden();
    let cpu_m = load_bf16(intarsia_backend_cpu::CpuBackend::arc(), 64);
    let mut st = cpu_m.new_state().unwrap();
    let cpu_rows = all_rows(&cpu_m, &mut st, &ids);
    // bf16 weights alone move a random tiny model far from the fp32 HF logits: the cpu bf16 run is the yardstick
    let cpu_vs_hf = rel_l2_rows(&cpu_rows, &want);
    let hf_budget = (2.0 * cpu_vs_hf).max(1e-2);
    println!("cpu bf16 vs HF fp32: worst rel l2 {cpu_vs_hf:.2e}");
    for chunk in [64, 5] {
        let m = load_bf16(gpu.clone(), chunk);
        let mut st = m.new_state().unwrap();
        let rows = all_rows(&m, &mut st, &ids);
        let vs_cpu = rel_l2_rows(&rows, &cpu_rows);
        let vs_hf = rel_l2_rows(&rows, &want);
        println!("cuda prefill chunk {chunk}: worst rel l2 vs cpu bf16 {vs_cpu:.2e}, vs HF fp32 {vs_hf:.2e}");
        assert!(vs_cpu < 3e-2, "chunk {chunk}: cuda vs cpu {vs_cpu}");
        assert!(vs_hf < hf_budget, "chunk {chunk}: cuda vs HF {vs_hf} (budget {hf_budget})");
        let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let agree = rows.iter().zip(&cpu_rows).filter(|(g, w)| am(g) == am(w)).count();
        println!("cuda top-1 vs cpu bf16: {agree}/{}", rows.len());
        assert!(agree + 1 >= rows.len(), "top-1 agreement with cpu bf16 {agree}/{}", rows.len());
        // prefill 17 then single-token decode through the T=1 kernels
        let mut st = m.new_state().unwrap();
        let pre = all_rows(&m, &mut st, &ids[..17]);
        assert!(rel_l2_rows(&pre, &want[..17]) < hf_budget);
        let mut dec = Vec::new();
        for i in 17..ids.len() {
            match m.forward(&mut st, &ids[i..i + 1], HeadMode::LastLogits, None).unwrap() {
                StepOut::Logits(l) => dec.push(l),
                _ => panic!("no logits"),
            }
        }
        let d = rel_l2_rows(&dec, &want[17..]);
        println!("cuda decode: worst rel l2 vs HF {d:.2e}");
        assert!(d < hf_budget, "decode {d} (budget {hf_budget})");
        // snapshot / restore on the device state
        let mut st = m.new_state().unwrap();
        m.forward(&mut st, &ids[..12], HeadMode::None, None).unwrap();
        st.snapshot(&*m.backend, 2).unwrap();
        let a = all_rows(&m, &mut st, &ids[12..]);
        st.truncate(12, &*m.backend).unwrap();
        let b = all_rows(&m, &mut st, &ids[12..]);
        assert_eq!(a, b, "cuda replay from a restored snapshot must be bit-identical");
    }
}
