//! The verify head's logits rows are the decode logits after the final softcap (lane ink-mtp; found by ink-cuda-attn, gate5,
//! fleetq #11254). Inkling-2 caps its logits (`final_logit_softcapping`, 30 in the tiny fixture as in the real checkpoint);
//! decode applies the cap on readback (`Model::softcap_logits`), and a sampled verify (`Model::forward_verify_logits`, rejection
//! sampling) must sample from the same capped distribution. Greedy is unaffected (the cap is monotone). Backend: cpu, or Metal
//! with `INTARSIA_INKLING2_BACKEND=metal`, CUDA with `=cuda`, or HIP with `=hip`.

mod common;
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::Telemetry;
use intarsia_models::safetensors::SafeTensors;
use std::path::PathBuf;
use std::sync::Arc;

fn backend() -> Arc<dyn intarsia_core::Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        #[cfg(target_os = "macos")]
        Ok("metal") => intarsia_backend_metal::MetalBackend::arc().unwrap(),
        Ok("cuda") => intarsia_backend_cuda::create(0).unwrap(),
        // lane ink-hip-spec: one R9700 (HIP_VISIBLE_DEVICES picks the card)
        Ok("hip") => intarsia_backend_hip::create(0).unwrap(),
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu, metal, cuda or hip"),
    }
}

/// Row `t` of a verify pass equals the logits of the one-token decode step that feeds the same token at the same position
/// after the same prefill, after the softcap: bitwise (verify rows are row-exact to decode steps; the cap is the same host
/// function). Unfixed, every logit of every row differs (the verify rows were pre-softcap).
#[test]
fn verify_logits_rows_are_the_softcapped_decode_logits() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/inkling2-tiny");
    if !dir.join("config.json").exists() {
        eprintln!("skipped: no tiny fixture");
        return;
    }
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let cap = cfg["text_config"]["final_logit_softcapping"].as_f64().or(cfg["final_logit_softcapping"].as_f64()).unwrap_or(0.0) as f32;
    assert!(cap > 0.0, "the fixture must cap its logits (vacuous test)");
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&dir).unwrap();
    let opts = LoadOptions { max_seq: 128, prefill_chunk: 16, graphs: false, ..Default::default() };
    let mut m = Model::load(backend(), &recipe, &dims, &src, &opts, Arc::new(Telemetry::new())).unwrap();
    m.enable_verify(4).unwrap();
    let prompt: &[u32] = &[3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12];
    let ids: &[u32] = &[44, 5, 200, 17];

    // both: the prompt minus its last token; then decode one token at a time over vids, or one verify pass over vids
    let mut vids = vec![*prompt.last().unwrap()];
    vids.extend_from_slice(&ids[..ids.len() - 1]);
    let mut sd = m.new_state().unwrap();
    m.forward(&mut sd, &prompt[..prompt.len() - 1], HeadMode::LastArgmax, None).unwrap();
    let mut want = Vec::new();
    for &id in &vids {
        match m.forward(&mut sd, &[id], HeadMode::LastLogits, None).unwrap() {
            StepOut::Logits(l) => want.push(l),
            o => panic!("decode returned {o:?}"),
        }
    }
    let mut sv = m.new_state().unwrap();
    m.forward(&mut sv, &prompt[..prompt.len() - 1], HeadMode::LastArgmax, None).unwrap();
    let (_, rows) = m.forward_verify_logits(&mut sv, &vids).unwrap();
    assert_eq!(rows.len(), want.len());
    for (t, (got, w)) in rows.iter().zip(&want).enumerate() {
        assert!(w.iter().all(|x| x.abs() <= cap), "row {t}: decode logits above the cap");
        let diff = got.iter().zip(w).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let maxd = got.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert_eq!(diff, 0, "row {t}: {diff} of {} verify logits differ from the capped decode logits (max |d| {maxd:e})", w.len());
    }
}
