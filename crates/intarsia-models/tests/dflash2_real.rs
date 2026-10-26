//! Lane dfh probe (docs/results/2026-10-26-dfh-hip-dflash.md): the DFlash2 draft on a real target (the Qwen3.8-27B MXFP4 on HIP).
//!
//! * Writes the feature taps of a long prefill's last chunk (`[rows, H]` f32 per tap) to `INTARSIA_DFLASH2_TAPS_OUT`, so two
//!   runs (fusion on, `INTARSIA_NO_FUSION=1`) can be compared: a fused kernel that never writes the residual at a layer
//!   boundary would show as a large difference (`scripts/dfh/cmp_taps.py`).
//! * Times the draft pass (`Model::dflash_draft`, 7 drafts) against one decode step and an 8-row verify pass.
//!
//! Opt-in, loads the whole checkpoint: `INTARSIA_DFLASH2_TARGET` (target dir), `INTARSIA_DFLASH2_DRAFT` (draft dir),
//! `INTARSIA_BATCH_TEST_BACKEND=hip` (else CUDA, which refuses DFlash2). `INTARSIA_DFLASH2_LEN` context length (default 3000).
//! Take a fleetq slot first. Unset: SKIPPED, passes.
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::Telemetry;
use intarsia_models::load::{open_dflash, SchemedSource};
use intarsia_models::{read_model_spec, EngineOptions};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, lo: u32, hi: u32) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lo + ((self.0 >> 33) as u32) % (hi - lo)
    }
}

#[test]
fn dflash2_real_taps_and_cost() {
    let env = |k: &str| std::env::var(k).ok();
    eprintln!(
        "PROBE_CONFIG target={:?} draft={:?} backend={:?} len={:?} out={:?} INTARSIA_NO_FUSION={:?}",
        env("INTARSIA_DFLASH2_TARGET"),
        env("INTARSIA_DFLASH2_DRAFT"),
        env("INTARSIA_BATCH_TEST_BACKEND"),
        env("INTARSIA_DFLASH2_LEN"),
        env("INTARSIA_DFLASH2_TAPS_OUT"),
        env("INTARSIA_NO_FUSION")
    );
    if env("INTARSIA_VERIFY_DRYRUN").as_deref() == Some("1") {
        eprintln!("PROBE_DRYRUN: not loading");
        return;
    }
    let (Some(tdir), Some(ddir)) = (env("INTARSIA_DFLASH2_TARGET").map(PathBuf::from), env("INTARSIA_DFLASH2_DRAFT").map(PathBuf::from)) else {
        eprintln!("SKIPPED: INTARSIA_DFLASH2_TARGET / INTARSIA_DFLASH2_DRAFT unset");
        return;
    };
    let hip = env("INTARSIA_BATCH_TEST_BACKEND").as_deref() == Some("hip");
    let be = match if hip { intarsia_backend_hip::create(0) } else { intarsia_backend_cuda::create(0) } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIPPED: no usable GPU ({e})");
            return;
        }
    };
    let len: usize = env("INTARSIA_DFLASH2_LEN").and_then(|v| v.parse().ok()).unwrap_or(3000);
    let (_, recipe, dims, _) = read_model_spec(&tdir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(&tdir).unwrap();
    let chunk = recipe.resolve_prefill_chunk(None, be.name()).value;
    let dl = open_dflash(Some(&ddir), false).unwrap();
    let opts = LoadOptions {
        max_seq: (len + 1100).next_multiple_of(1024),
        prefill_chunk: chunk,
        kv_fraction: 0.5,
        profile_every: 0,
        dflash: Some(dl),
        ..Default::default()
    };
    let mut m = Model::load(be, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
    m.enable_verify(8).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    let hi = (dims.vocab as u32).min(150_000);
    let mut rng = Lcg(7);
    let prompt: Vec<u32> = (0..len).map(|_| rng.next(1000, hi)).collect();
    let mut st = m.new_state().unwrap();
    let mut last = 0;
    for c in prompt.chunks(chunk) {
        m.forward(&mut st, c, HeadMode::None, None).unwrap();
        last = c.len();
    }
    // taps of the last chunk
    let taps = m.dflash_taps(last).unwrap();
    if let Some(out) = env("INTARSIA_DFLASH2_TAPS_OUT") {
        let mut b = Vec::with_capacity(taps.len() * last * dims.hidden * 4 + 16);
        b.extend_from_slice(&(taps.len() as u32).to_le_bytes());
        b.extend_from_slice(&(last as u32).to_le_bytes());
        b.extend_from_slice(&(dims.hidden as u32).to_le_bytes());
        for t in &taps {
            for v in t {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(&out, b).unwrap();
        eprintln!("PROBE_TAPS wrote {} taps x {last} rows x {} to {out}", taps.len(), dims.hidden);
    }
    for (i, t) in taps.iter().enumerate() {
        let rms = (t.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / t.len() as f64).sqrt();
        let finite = t.iter().all(|v| v.is_finite());
        eprintln!("PROBE_TAP {i}: rms {rms:.4} finite {finite}");
        assert!(finite && rms > 0.0, "tap {i} is empty or not finite");
    }
    // cost: one decode step (the last prompt token's successor), the draft, an 8-row verify pass
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let next = match m.forward(&mut st, &[prompt[len - 1]], HeadMode::LastArgmax, None).unwrap() {
        intarsia_core::model::StepOut::Token(t) => t,
        _ => unreachable!(),
    };
    let base = st.len;
    st.snapshot(&*m.backend, intarsia_core::state::RECURRENT_SNAPSHOTS).unwrap();
    let (mut dec, mut drf, mut ver) = (Vec::new(), Vec::new(), Vec::new());
    let mut drafts = Vec::new();
    for _ in 0..12 {
        let t = Instant::now();
        drafts = m.dflash_draft(&mut st, next, 7).unwrap();
        drf.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        m.forward(&mut st, &[next], HeadMode::LastArgmax, None).unwrap();
        dec.push(t.elapsed().as_secs_f64() * 1e3);
        st.truncate(base, &*m.backend).unwrap();
    }
    eprintln!("PROBE_DRAFTS {drafts:?}");
    let mut ids = vec![next];
    ids.extend(&drafts);
    ids.truncate(8);
    for _ in 0..8 {
        let t = Instant::now();
        let v = m.forward_verify(&mut st, &ids).unwrap();
        ver.push(t.elapsed().as_secs_f64() * 1e3);
        let acc = v.iter().zip(&ids[1..]).take_while(|(a, b)| a == b).count();
        eprintln!("PROBE_VERIFY target {v:?}, drafts accepted {acc} of {}", ids.len() - 1);
        st.truncate(base, &*m.backend).unwrap();
    }
    eprintln!(
        "PROBE_TIME decode1 {:.2} ms (n {}), draft7 {:.2} ms, verify8 {:.2} ms",
        med(dec.clone()),
        dec.len(),
        med(drf),
        med(ver)
    );
}
