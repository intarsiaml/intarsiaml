//! Lane dc probe (docs/results/2026-10-25-dc-dflash-dcoco.md, gate G1): is a 16-row verify pass on the real Qwen3-8B
//! bitwise the 16 single-row decode steps it replaces, on CUDA (and HIP with `INTARSIA_BATCH_TEST_BACKEND=hip`)?
//!
//! DFlash verifies `[anchor, 15 drafts]` = 16 rows (`Model::VERIFY_MAX_ROWS`). The MTP and ngram gates covered at most
//! 7 rows. `Model::verify_check` runs the ids through q single-row steps and through one `forward_verify`, from the same
//! state, and returns the max |logit difference| and both argmaxes per row. Greedy exactness of `--spec dflash` needs
//! every difference to be 0.0.
//!
//! Opt-in, loads 16 GB: `INTARSIA_QWEN3_DIR` (the 8B checkpoint). Take a fleetq slot first. Unset: SKIPPED, passes.
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::state::SequenceState;
use intarsia_core::Telemetry;
use intarsia_models::load::SchemedSource;
use intarsia_models::{read_model_spec, EngineOptions};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var_os("INTARSIA_QWEN3_DIR")?);
    d.join("config.json").exists().then_some(d)
}

fn backend() -> Option<Arc<dyn intarsia_core::backend::Backend>> {
    let hip = std::env::var("INTARSIA_BATCH_TEST_BACKEND").is_ok_and(|v| v == "hip");
    let made = if hip { intarsia_backend_hip::create(0) } else { intarsia_backend_cuda::create(0) };
    match made {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("SKIPPED: no usable GPU ({e})");
            None
        }
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, lo: u32, hi: u32) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lo + ((self.0 >> 33) as u32) % (hi - lo)
    }
}

fn prefill(m: &Model, st: &mut SequenceState, ids: &[u32]) {
    let chunk = m.prefill_chunk();
    for c in ids.chunks(chunk) {
        m.forward(st, c, HeadMode::None, None).unwrap();
    }
}

#[test]
fn verify16_is_bitwise_the_single_row_steps() {
    let (Some(dir), Some(be)) = (dir(), backend()) else {
        eprintln!("SKIPPED: INTARSIA_QWEN3_DIR / GPU unavailable");
        return;
    };
    let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(&dir).unwrap();
    let chunk = recipe.resolve_prefill_chunk(None, be.name()).value;
    let opts = LoadOptions { max_seq: 4096, prefill_chunk: chunk, kv_fraction: 0.5, ..Default::default() };
    let mut m = Model::load(be, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
    m.enable_verify(Model::VERIFY_MAX_ROWS).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    eprintln!("plan notes: {:?}", m.plan.notes);

    let mut rng = Lcg(42);
    let mut bad = 0usize;
    let mut rows = 0usize;
    let mut worst = 0f32;

    // 1. every row count 2..=16 at several context lengths, random ids (worst case for argmax ties), logits compared
    for &len in &[17usize, 130, 513, 1024, 1900, 3000] {
        let prompt: Vec<u32> = (0..len).map(|_| rng.next(1000, 30000)).collect();
        let mut st = m.new_state().unwrap();
        prefill(&m, &mut st, &prompt);
        for q in 2..=Model::VERIFY_MAX_ROWS {
            let ids: Vec<u32> = (0..q).map(|_| rng.next(1000, 30000)).collect();
            let r = m.verify_check(&mut st, &ids).unwrap();
            let md = r.iter().map(|x| x.0).fold(0f32, f32::max);
            let am = r.iter().filter(|x| x.1 != x.2).count();
            worst = worst.max(md);
            rows += r.len();
            if md != 0.0 || am != 0 {
                bad += 1;
                eprintln!("DIFF len={len} q={q}: max|dlogit|={md:e} argmax mismatches={am}");
            }
        }
        eprintln!("len={len}: q=2..=16 checked");
    }

    // 2. on-distribution: the model's own greedy continuation as the 16-row drafts (all accepted), 24 windows
    let prompt: Vec<u32> = (0..37).map(|_| rng.next(1000, 30000)).collect();
    let mut st = m.new_state().unwrap();
    prefill(&m, &mut st, &prompt);
    let mut seq = prompt.clone();
    let mut tok = match m.forward(&mut st, &[*seq.last().unwrap()], HeadMode::LastArgmax, None).unwrap() {
        StepOut::Token(t) => t,
        _ => unreachable!(),
    };
    for _ in 0..(24 * 16 + 16) {
        seq.push(tok);
        tok = match m.forward(&mut st, &[tok], HeadMode::LastArgmax, None).unwrap() {
            StepOut::Token(t) => t,
            _ => unreachable!(),
        };
    }
    let mut st2 = m.new_state().unwrap();
    prefill(&m, &mut st2, &seq[..prompt.len()]);
    let mut at = prompt.len();
    for w in 0..24 {
        // fill the state up to `at`, then probe ids = seq[at .. at+16]
        if st2.len < at {
            let from = st2.len;
            prefill(&m, &mut st2, &seq[from..at]);
        }
        let r = m.verify_check(&mut st2, &seq[at..at + 16]).unwrap();
        let md = r.iter().map(|x| x.0).fold(0f32, f32::max);
        let am = r.iter().filter(|x| x.1 != x.2).count();
        worst = worst.max(md);
        rows += r.len();
        if md != 0.0 || am != 0 {
            bad += 1;
            eprintln!("DIFF greedy window {w} at {at}: max|dlogit|={md:e} argmax mismatches={am}");
        }
        at += 16;
    }

    // 3. rough cost (probe only, not a benchmark: localeval measures the real thing): median of 12 passes at length 1024
    let prompt: Vec<u32> = (0..1024).map(|_| rng.next(1000, 30000)).collect();
    let mut st = m.new_state().unwrap();
    prefill(&m, &mut st, &prompt);
    let ids16: Vec<u32> = (0..16).map(|_| rng.next(1000, 30000)).collect();
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let mut t1 = vec![];
    let mut t16 = vec![];
    for _ in 0..12 {
        let t = Instant::now();
        m.forward(&mut st, &[ids16[0]], HeadMode::LastArgmax, None).unwrap();
        t1.push(t.elapsed().as_secs_f64() * 1e3);
        st.truncate(1024, &*m.backend).unwrap();
        let t = Instant::now();
        m.forward_verify(&mut st, &ids16).unwrap();
        t16.push(t.elapsed().as_secs_f64() * 1e3);
        st.truncate(1024, &*m.backend).unwrap();
    }
    let (a, b) = (med(t1), med(t16));
    eprintln!("PROBE_TIME decode1 {a:.1} ms, verify16 {b:.1} ms, ratio {:.2}", b / a);
    eprintln!("PROBE_RESULT rows={rows} windows_with_diff={bad} worst_abs_logit_diff={worst:e}");
    assert_eq!(bad, 0, "16-row verify is not bitwise the single-row steps");
}
