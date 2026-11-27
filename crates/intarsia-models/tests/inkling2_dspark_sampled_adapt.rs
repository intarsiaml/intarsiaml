//! Lane ink-sampled-adapt (docs/results/2026-11-26-inkling2-sampled-adapt.md): the per-request draft length of sampled DSpark speculation
//! (`INTARSIA_SPEC_SAMPLED_ADAPT`, `Model::set_spec_sampled_adapt` here, `crate::spec::AdaptK`) on the tiny fixtures of
//! `tests/inkling2_dspark.rs`. Backend: cpu, or Metal / CUDA / HIP with `INTARSIA_INKLING2_BACKEND=metal|cuda|hip`.
//!
//! * `plain_steps_keep_the_draft_in_sync`: a plain step (the policy's k = 0) leaves the DSpark draft's context in sync: after prefill
//!   and `n` single-token decode steps (n = 2, 3, 17) the state is synced with every row accounted for, and the next draft equals the draft of a state
//!   that took the same tokens through verify passes (whose rows are bitwise the decode steps' and go into the draft's cache directly).
//! * `adapt_seeded_ids_equal_spec_off`: with the policy on (the backend's table, an aggressive table that stops and resumes drafting
//!   often, and the `drafts` diagnostic) a seeded request emits the ids of `--spec off`; the aggressive table both drafts and takes
//!   plain steps without a draft call.

mod common;
mod ink_sampled;
use ink_sampled::*;

use intarsia_core::model::{DraftScript, HeadMode};
use intarsia_core::spec::{AdaptMode, AdaptParams, SpecConfig, SpecMode};
use intarsia_core::{EngineMeta, LocalEngine, Telemetry};
use std::sync::{Arc, Mutex};

/// Every test of this binary: the draft proposes `k` tokens every call (no gate, no cut: `gate_off`), and the back-off gate is off
/// (`INTARSIA_SPEC_GATE` fails = 0), so plain steps between draft calls come from the policy only.
fn setup() {
    gate_off();
    static ONCE: Mutex<()> = Mutex::new(());
    let _g = ONCE.lock().unwrap();
    std::env::set_var("INTARSIA_SPEC_GATE", "0,0,0,2");
}

fn engine_a(k: usize, mode: Option<AdaptMode>, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>) {
    let (m, tel) = model(Some(k), Some(true), script);
    m.set_spec_sampled_adapt(mode);
    let spec = SpecConfig { mode: SpecMode::Dflash, k, k_min: None };
    let e = LocalEngine::new(
        m,
        Arc::new(common::ByteCodec),
        EngineMeta { id: "tiny-dspark".into(), eos_ids: vec![], spec, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    (e, tel)
}

#[test]
fn plain_steps_keep_the_draft_in_sync() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    setup();
    let k = 3;
    let (m, _) = model(Some(k), Some(true), None);
    let prompt = PROMPTS[0];
    let toks: Vec<u32> = (0..17u32).map(|i| (i * 31 + 7) % 250 + 1).collect();
    let fresh = || {
        let mut st = m.new_state().unwrap();
        st.dflash.as_mut().expect("a DSpark state").active = true;
        m.forward(&mut st, prompt, HeadMode::None, None).unwrap();
        st
    };
    for n in [2usize, 3, 17] {
        let next = toks[n];
        // plain decode steps: each stashes its tap row
        let mut a = fresh();
        for &t in &toks[..n] {
            m.forward(&mut a, &[t], HeadMode::LastLogits, None).unwrap();
        }
        let d = a.dflash.as_ref().unwrap();
        assert!(d.active && d.synced, "n {n}: the draft fell out of sync after plain steps");
        assert_eq!((d.kv.len + d.pending, d.pending), (a.len, n), "n {n}: the stash must hold every plain step's row");
        let da = m.dflash_draft(&mut a, next, k).unwrap();
        let d = a.dflash.as_ref().unwrap();
        assert_eq!((d.kv.len, d.pending), (a.len, 0), "n {n}: the draft call flushes the stash");
        // the same tokens through verify passes of 2..=k + 1 rows, kept whole: their rows go into the draft's cache directly
        let mut b = fresh();
        let mut rest = &toks[..n];
        while !rest.is_empty() {
            let c = if rest.len() == k + 2 { k } else { rest.len().min(k + 1) };
            m.forward_verify(&mut b, &rest[..c]).unwrap();
            rest = &rest[c..];
        }
        assert_eq!(b.len, a.len);
        let db = m.dflash_draft(&mut b, next, k).unwrap();
        assert_eq!(da.len(), k, "n {n}: the draft proposes k tokens (no gate, no cut)");
        assert_eq!(da, db, "n {n}: the draft after plain steps differs from the draft over the same rows ingested directly");
    }
}

/// An aggressive table: the draft costs almost a plain step, so the tiny draft's acceptance rarely pays; holds are short, so the request
/// stops, probes and resumes drafting many times per request.
fn aggressive() -> AdaptMode {
    AdaptMode::On(AdaptParams { draft: 0.6, margin: 0.0, hold0: 2, hold_max: 4, explore: 3, ..AdaptParams::default_for("cpu") })
}

/// Tokens of a request not made by a verify pass (plain steps; the first token comes from prefill).
fn plain_tokens(n_total: u64, t: &intarsia_core::TelemetrySnapshot) -> u64 {
    n_total - t.spec_steps - t.spec_accepted
}

#[test]
fn adapt_seeded_ids_equal_spec_off() {
    if dirs().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    setup();
    let (off, _) = engine(None, None, None);
    let be = backend().name().to_string();
    let modes = [("table", AdaptMode::On(AdaptParams::default_for(&be))), ("aggressive", aggressive()), ("drafts", AdaptMode::Drafts)];
    for (name, mode) in &modes {
        let (on, tel) = engine_a(3, Some(mode.clone()), None);
        for cfg in CFGS {
            for p in PROMPTS {
                for n in [5usize, 40] {
                    for seed in [1u64, 7] {
                        assert_eq!(run(&on, sreq(p, n, seed, cfg)), run(&off, sreq(p, n, seed, cfg)), "{name}: random draft, cfg {cfg:?} n {n} seed {seed}");
                    }
                }
            }
        }
        let t = tel.snapshot();
        eprintln!("{name}: random draft: passes {} proposed {} accepted {}", t.spec_steps, t.spec_proposed, t.spec_accepted);
        if *name == "drafts" {
            assert_eq!(t.spec_steps, 0, "drafts: the diagnostic never verifies");
        }
    }
    // scripted proposals that are partly right: the aggressive table must both draft and hold plain steps; fixed k never holds
    let mut plain = vec![];
    for (name, mode) in [("fixed", AdaptMode::Off), ("aggressive", aggressive())] {
        let full = Arc::new(Mutex::new(Vec::new()));
        let (sc, tel) = engine_a(3, Some(mode), Some(partial_script(full.clone())));
        let mut total = 0u64;
        for cfg in CFGS {
            for p in PROMPTS {
                let reference = run(&off, sreq(p, 48, 11, cfg));
                *full.lock().unwrap() = p.iter().chain(&reference).copied().collect();
                assert_eq!(run(&sc, sreq(p, 48, 11, cfg)), reference, "{name}: scripted cfg {cfg:?} prompt {p:?}");
                assert_eq!(run(&sc, sreq(p, 30, 12, cfg)), run(&off, sreq(p, 30, 12, cfg)), "{name}: scripted cfg {cfg:?} seed 12");
                total += 47 + 29;
            }
        }
        let t = tel.snapshot();
        let pl = plain_tokens(total, &t);
        eprintln!("{name}: scripted: passes {} proposed {} accepted {} plain-step tokens {pl}", t.spec_steps, t.spec_proposed, t.spec_accepted);
        assert!(t.spec_steps > 0 && t.spec_accepted > 0, "{name}: scripted requests never drafted (vacuous)");
        plain.push(pl);
    }
    // fixed k takes plain steps only where max_tokens leaves no room for a draft (one per request at most here)
    assert!(plain[0] <= 2 * 9 * 2, "fixed: {} plain-step tokens", plain[0]);
    assert!(plain[1] > plain[0] + 50, "aggressive: {} plain-step tokens against fixed k's {}: the policy never held drafting", plain[1], plain[0]);
}
