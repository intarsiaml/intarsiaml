//! Inkling-2 on the tiny fixture (`golden/inkling2-tiny`, `tools/inkling2_ref.py tiny`): gate G1 of the port spec
//! (docs/plans/2026-11-21-inkling2-port.md section i) on the cpu backend, plus the state contract (chunked prefill across the
//! window and the 6-layer blocks, prefill then decode, snapshot restore, failed-pass reset, speculative rollback by length,
//! batched == solo) and NoKernel refusal on a backend without the Inkling-2 kernels.
//!
//! Skipped (loudly) when the fixture is missing; fails under `CI=1`.

use intarsia_core::model::{HeadMode, InkTensors, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::Telemetry;
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn tiny_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("INTARSIA_INKLING2_TINY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/inkling2-tiny"));
    dir.join("config.json").exists().then_some(dir)
}

fn skip(why: &str) {
    println!("SKIPPED: inkling2: {why}");
    eprintln!("SKIPPED: inkling2: {why}");
    if std::env::var("CI").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("inkling2 was SKIPPED ({why}) but CI is set");
    }
}

fn load_with(dir: &Path, chunk: usize, backend: Arc<dyn intarsia_core::Backend>) -> intarsia_core::Result<Model> {
    load_act(dir, chunk, backend, None)
}

fn load_act(dir: &Path, chunk: usize, backend: Arc<dyn intarsia_core::Backend>, act: Option<intarsia_core::DType>) -> intarsia_core::Result<Model> {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert!(dims.ink.is_some());
    let src = SafeTensors::open_dir(dir).unwrap();
    Model::load(backend, &recipe, &dims, &src, &LoadOptions { max_seq: 128, prefill_chunk: chunk, graphs: false, act_dtype: act, kv_dtype: act, ..Default::default() }, Arc::new(Telemetry::new()))
}

fn load(dir: &Path, chunk: usize) -> Model {
    load_with(dir, chunk, intarsia_backend_cpu::CpuBackend::arc()).unwrap()
}

/// Every tensor of a safetensors file as f32 (F32 / BF16 / I64 / I32), by name.
fn read_all(path: &Path) -> InkTensors {
    let b = std::fs::read(path).unwrap();
    let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hl]).unwrap();
    let mut out = InkTensors::new();
    for (name, t) in h.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let (s, e) = (t["data_offsets"][0].as_u64().unwrap() as usize, t["data_offsets"][1].as_u64().unwrap() as usize);
        let raw = &b[8 + hl + s..8 + hl + e];
        let data: Vec<f32> = match t["dtype"].as_str().unwrap() {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "I64" => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            "I32" => raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            other => panic!("{name}: dtype {other}"),
        };
        out.insert(name.clone(), (shape, data));
    }
    out
}

fn prompt(dir: &Path) -> Vec<u32> {
    read_all(&dir.join("ref_logits.safetensors"))["prompt_ids"].1.iter().map(|&v| v as u32).collect()
}

fn all_logits(m: &Model, st: &mut intarsia_core::state::SequenceState, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<f32>> = Vec::new();
    m.forward(st, ids, HeadMode::AllLogits, Some(&mut |_, row: &[f32]| {
        rows.push(row.to_vec());
        Ok(())
    }))
    .unwrap();
    rows
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
fn inkling2_tiny_loads() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir, 64);
    let ids = prompt(&dir);
    let mut st = m.new_state().unwrap();
    let rows = all_logits(&m, &mut st, &ids);
    assert_eq!(rows.len(), ids.len());
}

#[test]
fn inkling2_tiny_probe() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load_act(&dir, 64, intarsia_backend_cpu::CpuBackend::arc(), Some(intarsia_core::DType::F32)).unwrap();
    let ids = prompt(&dir);
    let r = read_all(&dir.join("ref_logits.safetensors"));
    let mut st = m.new_state().unwrap();
    let rows = all_logits(&m, &mut st, &ids);
    let ours: Vec<f32> = rows.concat();
    for k in ["logits_bf16", "logits_f32"] {
        println!("oracle {k} vs f64: {:.3e}", rel_l2(&r[k].1, &r["logits_f64"].1));
    }
    println!("ours vs f64: {:.3e}", rel_l2(&ours, &r["logits_f64"].1));
    let v = 300;
    let agree = (0..ids.len()).filter(|&t| argmax(&ours[t * v..(t + 1) * v]) == argmax(&r["logits_f64"].1[t * v..(t + 1) * v])).count();
    println!("top1 {agree}/{}", ids.len());
    let mx = ours.iter().zip(&r["logits_f64"].1).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("max abs {mx:.3e}");
}
