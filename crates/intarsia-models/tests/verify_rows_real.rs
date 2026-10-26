//! Lane dfh probe (docs/results/2026-10-26-dfh-hip-dflash.md, window 1 stop condition): is a q-row verify pass on a real
//! checkpoint (the hybrid Qwen3.8-27B MXFP4 on HIP is the target) bitwise the q single-row decode steps it replaces, and what
//! does it cost against one decode step? Same method as `verify16_cuda_real.rs` (lane dc), for recurrent models too:
//! `Model::verify_check` restores the GatedDeltaNet state from a snapshot, and the timing loop restores from one.
//!
//! Opt-in, loads the whole checkpoint: `INTARSIA_VERIFY_DIR` (checkpoint dir), `INTARSIA_VERIFY_ROWS` (most rows, default 8),
//! `INTARSIA_VERIFY_LENS` (comma list of context lengths, default 17,130,1024,4000,16000), `INTARSIA_BATCH_TEST_BACKEND=hip`
//! (else CUDA). Take a fleetq slot first. Unset: SKIPPED, passes.
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::state::SequenceState;
use intarsia_core::Telemetry;
use intarsia_models::load::SchemedSource;
use intarsia_models::{read_model_spec, EngineOptions};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var_os("INTARSIA_VERIFY_DIR")?);
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

fn greedy(m: &Model, st: &mut SequenceState, id: u32) -> u32 {
    match m.forward(st, &[id], HeadMode::LastArgmax, None).unwrap() {
        StepOut::Token(t) => t,
        _ => unreachable!(),
    }
}

#[test]
fn verify_rows_are_bitwise_the_single_row_steps() {
    let (Some(dir), Some(be)) = (dir(), backend()) else {
        eprintln!("SKIPPED: INTARSIA_VERIFY_DIR / GPU unavailable");
        return;
    };
    let most: usize = std::env::var("INTARSIA_VERIFY_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let lens: Vec<usize> = std::env::var("INTARSIA_VERIFY_LENS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![17, 130, 1024, 4000, 16000]);
    let max_len = lens.iter().copied().max().unwrap_or(1024).max(1100) + 600;
    let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(&dir).unwrap();
    let chunk = recipe.resolve_prefill_chunk(None, be.name()).value;
    // profile_every 0 as the engine runs (`--profile-every 0`): the default (1) profiles every pass, eagerly and synchronised per op
    let opts = LoadOptions { max_seq: max_len.next_multiple_of(1024), prefill_chunk: chunk, kv_fraction: 0.5, profile_every: 0, ..Default::default() };
    let mut m = Model::load(be, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
    m.enable_verify(most).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    eprintln!("plan notes: {:?}", m.plan.notes);
    let vocab = dims.vocab as u32;
    let hi = vocab.min(150_000);

    let mut rng = Lcg(42);
    let (mut bad, mut rows, mut flips) = (0usize, 0usize, 0usize);
    let mut worst = 0f32;
    let mut check = |m: &Model, st: &mut SequenceState, ids: &[u32], what: &str| {
        let r = m.verify_check(st, ids).unwrap();
        let md = r.iter().map(|x| x.0).fold(0f32, f32::max);
        let am = r.iter().filter(|x| x.1 != x.2).count();
        worst = worst.max(md);
        rows += r.len();
        flips += am;
        if md != 0.0 || am != 0 {
            bad += 1;
            eprintln!("DIFF {what} q={}: max|dlogit|={md:e} argmax mismatches={am}", ids.len());
        }
    };

    // 1. every row count 2..=most at several context lengths, random ids (worst case for argmax ties), logits compared
    for &len in &lens {
        let prompt: Vec<u32> = (0..len).map(|_| rng.next(1000, hi)).collect();
        let mut st = m.new_state().unwrap();
        prefill(&m, &mut st, &prompt);
        for q in 2..=most {
            let ids: Vec<u32> = (0..q).map(|_| rng.next(1000, hi)).collect();
            check(&m, &mut st, &ids, &format!("len={len}"));
        }
        eprintln!("len={len}: q=2..={most} checked");
    }

    // 2. on-distribution: the model's own greedy continuation as the drafts (all accepted), 16 windows of `most` rows
    let prompt: Vec<u32> = (0..37).map(|_| rng.next(1000, hi)).collect();
    let mut st = m.new_state().unwrap();
    prefill(&m, &mut st, &prompt);
    let mut seq = prompt.clone();
    let mut tok = greedy(&m, &mut st, *seq.last().unwrap());
    for _ in 0..(16 * most + most) {
        seq.push(tok);
        tok = greedy(&m, &mut st, tok);
    }
    let mut st2 = m.new_state().unwrap();
    prefill(&m, &mut st2, &seq[..prompt.len()]);
    let mut at = prompt.len();
    for w in 0..16 {
        if st2.len < at {
            let from = st2.len;
            prefill(&m, &mut st2, &seq[from..at]);
        }
        check(&m, &mut st2, &seq[at..at + most], &format!("greedy window {w} at {at}"));
        at += most;
    }

    // 3. cost (probe, not a benchmark): median of 12 passes at length 1024, each verify row count as the engine runs it
    //    (a captured replay where `Model::forward_verify` can, see `INTARSIA_VERIFY_GRAPH`)
    let len = 1024;
    let prompt: Vec<u32> = (0..len).map(|_| rng.next(1000, hi)).collect();
    let mut st = m.new_state().unwrap();
    prefill(&m, &mut st, &prompt);
    st.snapshot(&*m.backend, intarsia_core::state::RECURRENT_SNAPSHOTS).unwrap();
    let ids: Vec<u32> = (0..most).map(|_| rng.next(1000, hi)).collect();
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); most + 1];
    for _ in 0..12 {
        let t = Instant::now();
        m.forward(&mut st, &[ids[0]], HeadMode::LastArgmax, None).unwrap();
        times[1].push(t.elapsed().as_secs_f64() * 1e3);
        st.truncate(len, &*m.backend).unwrap();
        for q in 2..=most {
            let t = Instant::now();
            m.forward_verify(&mut st, &ids[..q]).unwrap();
            times[q].push(t.elapsed().as_secs_f64() * 1e3);
            st.truncate(len, &*m.backend).unwrap();
        }
    }
    let d1 = med(times[1].clone());
    let mut line = format!("PROBE_TIME decode1 {d1:.2} ms");
    for (q, t) in times.iter().enumerate().skip(2) {
        if !t.is_empty() {
            let v = med(t.clone());
            line += &format!(", verify{q} {v:.2} ms ({:.2}x)", v / d1);
        }
    }
    eprintln!("{line}");
    eprintln!("PROBE_GRAPHS {:?}", m.graph_stats());
    eprintln!("PROBE_RESULT rows={rows} windows_with_diff={bad} argmax_flips={flips} worst_abs_logit_diff={worst:e}");
    assert_eq!(bad, 0, "the verify pass is not bitwise the single-row steps");
}
