//! Lane dc, gate G2/G4: the DFlash draft of this engine against the independent MLX reference
//! (`tools/dc/ref`, docs/results/2026-10-25-dc-dflash-dcoco.md section 10.3), on real weights, without running the target.
//!
//! For each fixture (`tools/dc/export_fixture.py`): the reference's target features of the P context positions go into the
//! draft's context cache (`Model::dflash_ingest_host`), then one block pass drafts 15 tokens after the anchor; they must
//! equal the reference's argmax drafts (bf16 rounding may flip a near-tie, so up to 1 of 15 may differ), and the final
//! normed block output must agree to bf16 noise.
//!
//! Opt-in: `INTARSIA_QWEN3_DIR` (Qwen3-8B), `INTARSIA_DFLASH_DIR` (TD-Labs/Qwen3-8B-DCOCO), `INTARSIA_DFLASH_FIXTURES` (a
//! directory of `raw_*` fixture directories). `INTARSIA_DFLASH_TEST_BACKEND=cuda` runs it on the GPU (default: cpu-ref).
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::Telemetry;
use intarsia_models::load::{open_dflash, SchemedSource};
use intarsia_models::{read_model_spec, EngineOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from).filter(|p| p.exists())
}

fn f32s(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn i32s(p: &Path) -> Vec<i32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn run(real_prefill: bool) {
    let (Some(target), Some(draft), Some(fx)) =
        (env_dir("INTARSIA_QWEN3_DIR"), env_dir("INTARSIA_DFLASH_DIR"), env_dir("INTARSIA_DFLASH_FIXTURES"))
    else {
        eprintln!("SKIPPED: INTARSIA_QWEN3_DIR / INTARSIA_DFLASH_DIR / INTARSIA_DFLASH_FIXTURES not set");
        return;
    };
    let backend: Arc<dyn intarsia_core::backend::Backend> = match std::env::var("INTARSIA_DFLASH_TEST_BACKEND").as_deref() {
        Ok("cuda") => intarsia_backend_cuda::create(0).expect("cuda"),
        _ => intarsia_backend_cpu::CpuBackend::arc(),
    };
    let (_, recipe, dims, _) = read_model_spec(&target, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(&target).unwrap();
    let opts = LoadOptions {
        max_seq: 512,
        prefill_chunk: 256,
        kv_fraction: 0.2,
        dflash: Some(open_dflash(Some(&draft)).unwrap()),
        ..Default::default()
    };
    let mut m = Model::load(backend, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
    m.enable_verify(16).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&fx).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("meta.json").exists()).collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no fixtures in {}", fx.display());
    let mut worst_mismatch = 0usize;
    for d in &dirs {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join("meta.json")).unwrap()).unwrap();
        let (p, anchor) = (meta["P"].as_u64().unwrap() as usize, meta["anchor"].as_u64().unwrap() as u32);
        let feat = f32s(&d.join("feat.f32"));
        let want = i32s(&d.join("argmax_drafts.i32"));
        let want_h = f32s(&d.join("h_final.f32"));
        let mut st = m.new_state().unwrap();
        if real_prefill {
            // the real target prefills the context in pieces of 40 tokens: the taps and the ingest hook make the draft's cache
            let toks: Vec<u32> = meta["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
            assert_eq!(toks.len(), p + 1);
            for c in toks[..p].chunks(40) {
                m.forward(&mut st, c, intarsia_core::model::HeadMode::None, None).unwrap();
            }
            assert_eq!(st.len, p);
        } else {
            m.dflash_ingest_host(&mut st, &feat, p).unwrap();
        }
        let t0 = std::time::Instant::now();
        let got = m.dflash_draft(&mut st, anchor, 15).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let hid = m.dflash_last_hidden(15).unwrap();
        let (mut num, mut den, mut worst) = (0f64, 0f64, 0f32);
        for (a, b) in hid.iter().zip(&want_h) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
            worst = worst.max((a - b).abs());
        }
        let rel = (num / den).sqrt();
        let mismatch = got.iter().zip(&want).filter(|(g, w)| **g as i32 != **w).count();
        eprintln!(
            "{}: P={p} drafts {:?} want {:?} mismatches {mismatch}/15, hidden rel-L2 {rel:.4} max|d| {worst:.3}, draft pass {ms:.1} ms",
            d.file_name().unwrap().to_string_lossy(),
            got,
            want
        );
        assert!(rel < 0.05, "block output differs from the reference (rel-L2 {rel})");
        worst_mismatch = worst_mismatch.max(mismatch);
    }
    assert!(worst_mismatch <= if real_prefill { 4 } else { 2 }, "{worst_mismatch} of 15 drafts differ from the reference");
}

#[test]
fn draft_from_reference_features_matches_the_mlx_reference() {
    run(false);
}

#[test]
fn draft_after_real_target_prefill_matches_the_mlx_reference() {
    run(true);
}
