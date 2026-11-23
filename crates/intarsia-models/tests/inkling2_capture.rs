//! Diagnostics (lane ink-cuda-mixmoe): a module-boundary capture of the real Inkling-2-Mini on this backend, in the layout of
//! ink/spec's `tools/inkling2_ref.py replay` (prefill of the first P tokens in one pass, then one-token decode steps up to
//! position L, each visit's rows concatenated along T), so the oracle can replay every attention and MoE module on the inputs
//! this backend computed. Env-gated, no-op without `INTARSIA_INK_CAPTURE` (output dir). Also needs
//! `INTARSIA_INK_CAPTURE_MODEL` (checkpoint dir), `INTARSIA_INK_CAPTURE_GOLDEN` (a top-k golden; its `ids_<prompt>`),
//! `INTARSIA_INK_CAPTURE_PROMPT` (index), `INTARSIA_INK_CAPTURE_PREFILL` (P) and `INTARSIA_INK_CAPTURE_LAST` (L).
//! Backend: `INTARSIA_INKLING2_BACKEND` (cpu / cuda). Writes raw f32 files `v<NN>.<key>.f32` (rows x hidden) and `meta.json`;
//! the lane's `.scratch` converter turns them into the replay's safetensors.

use intarsia_core::model::{InkTensors, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::Telemetry;
use intarsia_models::SafeTensors;
use std::path::PathBuf;
use std::sync::Arc;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn backend() -> Arc<dyn intarsia_core::Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        Ok("cuda") => intarsia_backend_cuda::create(0).unwrap(),
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu or cuda"),
    }
}

/// `ids_<i>` of a top-k golden (I32 or I64).
fn golden_ids(path: &str, i: usize) -> Vec<u32> {
    let b = std::fs::read(path).unwrap();
    let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hl]).unwrap();
    let t = &h[format!("ids_{i}")];
    let (s, e) = (t["data_offsets"][0].as_u64().unwrap() as usize, t["data_offsets"][1].as_u64().unwrap() as usize);
    let raw = &b[8 + hl + s..8 + hl + e];
    match t["dtype"].as_str().unwrap() {
        "I32" => raw.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
        "I64" => raw.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
        d => panic!("ids dtype {d}"),
    }
}

const KEYS: [&str; 6] = ["attn_in_q", "attn_in_k", "attn_in_v", "attn_proj", "mlp_in", "moe_out"];

#[test]
fn inkling2_real_module_capture() {
    let Some(out) = env("INTARSIA_INK_CAPTURE").map(PathBuf::from) else { return };
    let model = PathBuf::from(env("INTARSIA_INK_CAPTURE_MODEL").expect("INTARSIA_INK_CAPTURE_MODEL"));
    let golden = env("INTARSIA_INK_CAPTURE_GOLDEN").expect("INTARSIA_INK_CAPTURE_GOLDEN");
    let pi: usize = env("INTARSIA_INK_CAPTURE_PROMPT").expect("INTARSIA_INK_CAPTURE_PROMPT").parse().unwrap();
    let pre: usize = env("INTARSIA_INK_CAPTURE_PREFILL").expect("INTARSIA_INK_CAPTURE_PREFILL").parse().unwrap();
    let last: usize = env("INTARSIA_INK_CAPTURE_LAST").expect("INTARSIA_INK_CAPTURE_LAST").parse().unwrap();
    let ids = golden_ids(&golden, pi);
    assert!(pre >= 1 && pre <= last && last < ids.len(), "prefill {pre}, last {last}, prompt of {} tokens", ids.len());
    std::fs::create_dir_all(&out).unwrap();

    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(model.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&model).unwrap();
    let opts = LoadOptions { max_seq: (last + 64).next_power_of_two(), prefill_chunk: 256, graphs: false, ..Default::default() };
    let m = Model::load(backend(), &recipe, &dims, &src, &opts, Arc::new(Telemetry::new())).unwrap();
    let mut st = m.new_state().unwrap();

    // the gate's schedule: one prefill pass of `pre` rows, then one-token steps
    let mut steps: Vec<InkTensors> = vec![m.ink_dump(&mut st, &ids[..pre], None).unwrap()];
    for p in pre..=last {
        steps.push(m.ink_dump(&mut st, &ids[p..p + 1], None).unwrap());
    }
    let d = dims.hidden;
    let mut visits = Vec::new();
    for vi in 0.. {
        let tag = format!("v{vi:02}");
        if !steps[0].contains_key(&format!("{tag}.mlp_in")) {
            break;
        }
        for key in KEYS {
            let name = format!("{tag}.{key}");
            let mut all: Vec<f32> = Vec::new();
            for s in &steps {
                let (shape, v) = s.get(&name).unwrap_or_else(|| panic!("no {name} in the dump"));
                assert_eq!(shape.last(), Some(&d), "{name} shape {shape:?}");
                all.extend_from_slice(v);
            }
            assert_eq!(all.len(), (last + 1) * d, "{name}");
            let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(out.join(format!("{name}.f32")), bytes).unwrap();
        }
        visits.push(vi);
    }
    let meta = serde_json::json!({
        "ids": &ids[..=last], "prefill": pre, "last": last, "hidden": d, "visits": visits,
        "backend": m.backend.name(), "golden": golden, "prompt": pi,
    });
    std::fs::write(out.join("meta_raw.json"), serde_json::to_string(&meta).unwrap()).unwrap();
    println!("capture: {} visits, {} rows ({} prefill + {} decode) -> {}", visits.len(), last + 1, pre, last + 1 - pre, out.display());
}
