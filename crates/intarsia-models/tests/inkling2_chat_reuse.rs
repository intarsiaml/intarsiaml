//! Inkling-2 chat reuse (lane ink-spec-land): a prompt's prefill leaves a ring snapshot at the prompt's end
//! (`SequenceState::ring_snapshot`, `INTARSIA_INK_RING_SNAPSHOT`), so the next turn of a chat resumes there after a reply longer
//! than the rings' 16-row margin, with the logits of a cold prefill of the whole new prompt, bit for bit. A cut more than the
//! margin below the snapshot starts over; with the snapshots off the sequence starts over as before. On the tiny fixture of
//! tests/inkling2.rs (window 8, margin 16: rings of 24 rows).
//!
//! `INTARSIA_INKLING2_BACKEND=metal` runs them on Metal (prefill chunk 16: every pass takes the row-exact kernels).
//! Skipped (loudly) when the fixture is missing; fails under `CI=1`.

use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::state::SequenceState;
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
    println!("SKIPPED: inkling2_chat_reuse: {why}");
    eprintln!("SKIPPED: inkling2_chat_reuse: {why}");
    if std::env::var("CI").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("inkling2_chat_reuse was SKIPPED ({why}) but CI is set");
    }
}

fn backend() -> Arc<dyn intarsia_core::Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        #[cfg(target_os = "macos")]
        Ok("metal") => intarsia_backend_metal::MetalBackend::arc().unwrap(),
        #[cfg(not(target_os = "macos"))]
        Ok("metal") => panic!("INTARSIA_INKLING2_BACKEND=metal: the Metal backend is macOS only"),
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu or metal"),
    }
}

/// One pass per prompt piece on cpu (rows-exact at any row count); 16 on Metal, where passes of 2..=16 rows take the row-exact
/// kernels.
fn chunk() -> usize {
    if std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() == Ok("metal") {
        16
    } else {
        64
    }
}

fn load(dir: &Path) -> Model {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(dir).unwrap();
    let opts = LoadOptions { max_seq: 128, prefill_chunk: chunk(), graphs: false, ..Default::default() };
    Model::load(backend(), &recipe, &dims, &src, &opts, Arc::new(Telemetry::new())).unwrap()
}

/// A 40-token prompt over the fixture's real vocab (ids 5..285; the tiny vocab is 320, 300 unpadded).
fn ids() -> Vec<u32> {
    (0..40u32).map(|i| (i * 37 + 11) % 280 + 5).collect()
}

/// A reply of `n` tokens that the next prompt does not contain.
fn reply(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 53 + salt * 17 + 3) % 270 + 7).collect()
}

fn all_logits(m: &Model, st: &mut SequenceState, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<f32>> = Vec::new();
    m.forward(st, ids, HeadMode::AllLogits, Some(&mut |_, row: &[f32]| {
        rows.push(row.to_vec());
        Ok(())
    }))
    .unwrap();
    rows
}

/// A prompt's prefill as the engines run it (`Model::forward_prefill`, last-row head): this is what takes the ring snapshot.
/// Returns the last row's logits.
fn prefill(m: &Model, st: &mut SequenceState, ids: &[u32]) -> Vec<f32> {
    match m.forward_prefill(st, ids, HeadMode::LastLogits, None).unwrap() {
        StepOut::Logits(l) => l,
        _ => panic!("no logits"),
    }
}

/// A reply's decode steps (one row each).
fn decode(m: &Model, st: &mut SequenceState, ids: &[u32]) {
    for &id in ids {
        m.forward(st, &[id], HeadMode::LastLogits, None).unwrap();
    }
}

/// The engine's prefix reuse for a new prompt (`Engine`: `reuse_len` on the shared prefix, then `truncate`), then the rest
/// of the prompt; returns the resume point and the rows of the rest.
fn resume(m: &Model, st: &mut SequenceState, prompt: &[u32]) -> (usize, Vec<Vec<f32>>) {
    let want = st.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
    let keep = m.reuse_len(st, prompt.len(), want);
    st.truncate(keep, &*m.backend).unwrap();
    (keep, all_logits(m, st, &prompt[keep..]))
}

fn cold(m: &Model, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut st = m.new_state().unwrap();
    all_logits(m, &mut st, ids)
}

/// Turn 2 resumes at turn 1's prompt end after an 18-token reply (past the 16-row margin), with a cold prefill's logits.
#[test]
fn inkling2_chat_turn2_resumes_at_prompt_end_bitwise() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir);
    let ids = ids();
    let whole = cold(&m, &ids);
    let mut st = m.new_state().unwrap();
    prefill(&m, &mut st, &ids[..20]);
    assert_eq!(st.ring_snap_at.map(|(s, _)| s), Some(20), "a snapshot at the prompt's end");
    decode(&m, &mut st, &reply(18, 1));
    assert_eq!(st.len, 38);
    let (keep, rows) = resume(&m, &mut st, &ids);
    assert_eq!(keep, 20, "turn 2 resumes at turn 1's prompt end");
    assert!(rows == whole[20..], "turn 2 rows differ from a cold prefill of the whole prompt");
}

/// Three turns, each reply past the margin: turns 2 and 3 resume at the previous prompt's end.
#[test]
fn inkling2_chat_three_turns() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir);
    let ids = ids();
    let whole = cold(&m, &ids);
    let mut st = m.new_state().unwrap();
    prefill(&m, &mut st, &ids[..12]);
    decode(&m, &mut st, &reply(20, 2));
    // turn 2: the prompt is ids[..24]; its prefill (last-row head) takes the next snapshot
    let want = 12;
    let keep = m.reuse_len(&st, 24, want);
    assert_eq!(keep, 12, "turn 2 resumes at turn 1's prompt end");
    st.truncate(keep, &*m.backend).unwrap();
    assert!(prefill(&m, &mut st, &ids[12..24]) == whole[23], "turn 2's last row");
    assert_eq!(st.ring_snap_at.map(|(s, _)| s), Some(24));
    decode(&m, &mut st, &reply(21, 3));
    let (keep, rows) = resume(&m, &mut st, &ids);
    assert_eq!(keep, 24, "turn 3 resumes at turn 2's prompt end");
    assert!(rows == whole[24..], "turn 3 rows differ from a cold prefill");
}

/// A shared prefix more than the margin below the snapshot starts over (and is right); one within the margin below it resumes
/// there; either cut drops the snapshot.
#[test]
fn inkling2_chat_cut_below_snapshot() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir);
    let ids = ids();
    let whole = cold(&m, &ids);
    let turn1 = |st: &mut SequenceState| {
        st.truncate(0, &*m.backend).unwrap();
        prefill(&m, st, &ids[..20]);
        decode(&m, st, &reply(18, 4));
    };
    let mut st = m.new_state().unwrap();
    turn1(&mut st);
    // 3 rows shared: 17 below the snapshot at 20
    assert_eq!(st.restorable_len(3), 0);
    assert_eq!(m.reuse_len(&st, 40, 3), 0);
    st.truncate(0, &*m.backend).unwrap();
    assert!(all_logits(&m, &mut st, &ids) == whole, "the fallback (a full prefill)");
    // 4 rows shared: exactly the margin below
    turn1(&mut st);
    assert_eq!(m.reuse_len(&st, 40, 4), 4);
    st.truncate(4, &*m.backend).unwrap();
    assert_eq!(st.ring_snap_at, None, "a cut below the snapshot drops it");
    assert!(all_logits(&m, &mut st, &ids[4..]) == whole[4..], "resumed 16 below the snapshot");
    // a later cut past the live margin, with no snapshot, starts over again
    assert_eq!(m.reuse_len(&st, 40, 4), 0);
}

/// `set_ring_snapshots(false)` (as `INTARSIA_INK_RING_SNAPSHOT=0`): no snapshot, and turn 2 starts over, as before.
#[test]
fn inkling2_chat_snapshot_off_starts_over() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir);
    m.set_ring_snapshots(false);
    let ids = ids();
    let whole = cold(&m, &ids);
    let mut st = m.new_state().unwrap();
    prefill(&m, &mut st, &ids[..20]);
    assert_eq!(st.ring_snap_at, None);
    decode(&m, &mut st, &reply(18, 5));
    let (keep, rows) = resume(&m, &mut st, &ids);
    assert_eq!(keep, 0);
    assert!(rows == whole);
}

/// Two sequences interleaved keep their own snapshots: each resumes with its own cold logits, whichever ran last.
#[test]
fn inkling2_chat_snapshots_per_sequence() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir);
    let a_ids = ids();
    let b_ids: Vec<u32> = a_ids.iter().rev().copied().collect();
    let (wa, wb) = (cold(&m, &a_ids), cold(&m, &b_ids));
    let (mut a, mut b) = (m.new_state().unwrap(), m.new_state().unwrap());
    prefill(&m, &mut a, &a_ids[..20]);
    prefill(&m, &mut b, &b_ids[..20]);
    decode(&m, &mut b, &reply(18, 6));
    decode(&m, &mut a, &reply(18, 7));
    let (kb, rb) = resume(&m, &mut b, &b_ids);
    let (ka, ra) = resume(&m, &mut a, &a_ids);
    assert_eq!((ka, kb), (20, 20));
    assert!(ra == wa[20..], "sequence a");
    assert!(rb == wb[20..], "sequence b");
}

/// The remainder after the resume point runs as passes the cold prefill would run too: on Metal, Inkling-2 passes of 2..=16 rows
/// take the row-exact kernels (not bitwise the fast prefill kernels' rows), so a short remainder moves the resume point back
/// until the remainder has 17 rows (still within the ring margin below the snapshot). Prefill chunk 64 here (every pass of this
/// test above 16 rows takes the fast kernels on Metal). On cpu (any cut) the resume point stays at the prompt end.
#[test]
fn inkling2_chat_short_remainder_matches_cold() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&dir).unwrap();
    let opts = LoadOptions { max_seq: 128, prefill_chunk: 64, graphs: false, ..Default::default() };
    let m = Model::load(backend(), &recipe, &dims, &src, &opts, Arc::new(Telemetry::new())).unwrap();
    let metal = std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() == Ok("metal");
    let ids = ids();
    for (turn2, want_keep) in [(30usize, if metal { 13 } else { 20 }), (40, 20)] {
        let mut st = m.new_state().unwrap();
        prefill(&m, &mut st, &ids[..20]);
        decode(&m, &mut st, &reply(18, 8));
        let cold_rows = cold(&m, &ids[..turn2]);
        let (keep, rows) = resume(&m, &mut st, &ids[..turn2]);
        assert_eq!(keep, want_keep, "turn 2 of {turn2} rows resumes at {want_keep}");
        assert!(rows == cold_rows[keep..], "turn 2 of {turn2} rows: rows differ from a cold prefill");
    }
}
