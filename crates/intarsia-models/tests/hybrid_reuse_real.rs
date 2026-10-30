//! Lane hrr: served greedy prefix reuse on the hybrid Qwen3.6-35B-A3B (GatedDeltaNet + attention) differs from cold on Metal
//! (lane mmtp's repro: request A = the 4096-token long-context prompt, then B = A + " Summarize the above in one paragraph."
//! resumes at 4095 and its text differs from a fresh server's). With no prefix store `Seq::new` resumes a recurrent state at any
//! snapshot `Model::reuse_len` finds (`SequenceState::restorable_len`): A's prefill snapshots at its `plen - 1` = 4095, so B
//! runs [0, 4095) (A's pass) + [4095, 4104) + [4104] alone, while cold B runs [0, 4096) + [4096, 4104) + [4104] alone.
//!
//! Opt-in: `INTARSIA_PROBE_DIR` = the hybrid checkpoint (`~/models/Qwen3.6-35B-A3B-NVFP4`); unset prints SKIPPED and passes.
//! `INTARSIA_PROBE_PROMPT` = the text of A (default: docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt).
//! * `hybrid_reuse_shapes`: diagnostic. The two pass shapes above from fresh states (logits of the 9 / 8-row pass, the last token
//!   and teacher-forced decode steps, the recurrent slots at 4105), the real path (A prefilled as the engine does, snapshot,
//!   decode, truncate to 4095, B's remainder), and small-pass splits from a common [0, 4096) prefix. With
//!   `INTARSIA_PROBE_ASSERT=1` it fails unless the shapes agree bit for bit.
//! * `engine_hybrid_reuse_equals_cold`: the gate. A single-sequence engine runs A then B; B's tokens must be a fresh engine's.
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::state::{SequenceState, StateSlot};
use intarsia_core::{GenEvent, GenRequest, Telemetry};
use intarsia_models::load::SchemedSource;
use intarsia_models::{load_engine, read_model_spec, EngineOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

type Be = Arc<dyn intarsia_core::backend::Backend>;

const SUFFIX: &str = " Summarize the above in one paragraph.";
const MAX_SEQ: usize = 8192;
const STEPS: usize = 4;

fn skip(why: &str) {
    eprintln!("SKIPPED: {why}");
    if std::env::var("INTARSIA_REQUIRE_GPU").is_ok_and(|v| v == "1") {
        panic!("SKIPPED ({why}) but INTARSIA_REQUIRE_GPU=1");
    }
}

fn model_dir() -> Option<PathBuf> {
    match std::env::var_os("INTARSIA_PROBE_DIR").map(PathBuf::from) {
        Some(d) if d.join("config.json").exists() => Some(d),
        Some(d) => {
            skip(&format!("INTARSIA_PROBE_DIR={} has no config.json", d.display()));
            None
        }
        None => {
            eprintln!("SKIPPED: INTARSIA_PROBE_DIR not set");
            None
        }
    }
}

fn backend() -> Option<Be> {
    let want = std::env::var("INTARSIA_PROBE_BACKEND").unwrap_or_else(|_| if cfg!(target_os = "macos") { "metal".into() } else { "cpu".into() });
    let made: Result<Be, String> = match want.as_str() {
        "cpu" => Ok(intarsia_backend_cpu::CpuBackend::arc()),
        #[cfg(target_os = "macos")]
        "metal" => intarsia_backend_metal::MetalBackend::arc().map_err(|e| e.to_string()),
        "hip" => intarsia_backend_hip::create(0).map_err(|e| e.to_string()),
        other => Err(format!("INTARSIA_PROBE_BACKEND={other} is not available in this build")),
    };
    match made {
        Ok(b) => Some(b),
        Err(e) => {
            skip(&format!("no usable {want} backend ({e})"));
            None
        }
    }
}

/// One test at a time: each loads the 35B.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn prompt_text() -> String {
    let p = std::env::var_os("INTARSIA_PROBE_PROMPT").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt")
    });
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Token ids of A and B as `serve` encodes a completion prompt.
fn prompts(dir: &PathBuf) -> (Vec<u32>, Vec<u32>) {
    use intarsia_core::engine::TextCodec;
    let codec = intarsia_models::HfCodec::load(dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    let text = prompt_text();
    let a = codec.encode(&text, false).unwrap();
    let b = codec.encode(&format!("{text}{SUFFIX}"), false).unwrap();
    (a, b)
}

fn logits_of(o: StepOut) -> Vec<f32> {
    match o {
        StepOut::Logits(l) => l,
        o => panic!("expected logits, got {o:?}"),
    }
}

fn argmax_of(l: &[f32]) -> u32 {
    l.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

/// What a run of B leaves: logits rows of the second-to-last pass (by absolute position), the last token's logits and
/// `STEPS` teacher-forced decode steps, the recurrent slots' bytes at the end of the prompt.
struct Run {
    rows: Vec<(usize, Vec<f32>)>,
    tail: Vec<Vec<f32>>,
    rec: Vec<Vec<u8>>,
}

fn recurrent_bytes(m: &Model, st: &SequenceState) -> Vec<Vec<u8>> {
    st.slots
        .iter()
        .filter_map(|s| match s {
            StateSlot::Recurrent(r) => {
                let mut v = vec![0u8; r.bytes];
                m.backend.download(&r.buf, 0, &mut v, 0).unwrap();
                Some(v)
            }
            _ => None,
        })
        .collect()
}

/// B from `st` (holding `b[..st.len]`): passes ending at each of `cuts`, then the last token alone. The pass ending at
/// `b.len() - 1` returns every row's logits.
fn finish(m: &Model, st: &mut SequenceState, b: &[u32], cuts: &[usize], forced: &[u32]) -> Run {
    let mut rows = Vec::new();
    for &c in cuts {
        let ids = &b[st.len..c];
        if c == b.len() - 1 {
            let mut cb = |pos: usize, row: &[f32]| -> intarsia_core::Result<()> {
                rows.push((pos, row.to_vec()));
                Ok(())
            };
            m.forward_prefill(st, ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
        } else {
            m.forward_prefill(st, ids, HeadMode::None, None).unwrap();
        }
    }
    let mut tail = vec![logits_of(m.forward_prefill(st, &b[b.len() - 1..], HeadMode::LastLogits, None).unwrap())];
    let rec = recurrent_bytes(m, st);
    for &t in forced {
        tail.push(logits_of(m.forward(st, &[t], HeadMode::LastLogits, None).unwrap()));
    }
    Run { rows, tail, rec }
}

fn fresh(m: &Model, b: &[u32], cuts: &[usize], forced: &[u32]) -> Run {
    let mut st = m.new_state().unwrap();
    finish(m, &mut st, b, cuts, forced)
}

/// Print how `got` compares with `want`; true when bitwise equal.
fn compare(what: &str, want: &Run, got: &Run) -> bool {
    let mut equal = true;
    let mut line = format!("{what}:");
    for (pos, r) in &got.rows {
        if let Some((_, w)) = want.rows.iter().find(|(p, _)| p == pos) {
            if r != w {
                equal = false;
                line.push_str(&format!(" row{pos}={:.3e}", maxabs(r, w)));
            }
        }
    }
    for (i, (r, w)) in got.tail.iter().zip(&want.tail).enumerate() {
        if r != w {
            equal = false;
            line.push_str(&format!(" tail{i}={:.3e}{}", maxabs(r, w), if argmax_of(r) != argmax_of(w) { "(FLIP)" } else { "" }));
        }
    }
    let rec_diff: Vec<usize> = got.rec.iter().zip(&want.rec).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect();
    if !rec_diff.is_empty() {
        equal = false;
        line.push_str(&format!(" recurrent slots differ {:?} of {}", rec_diff, got.rec.len()));
    }
    eprintln!("{line} {}", if equal { "EQUAL" } else { "DIFFER" });
    equal
}

fn load(dir: &PathBuf, backend: Be) -> Model {
    let (_, recipe, dims, _) = read_model_spec(dir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(dir).unwrap();
    let chunk = recipe.resolve_prefill_chunk(None, backend.name()).value;
    let opts = LoadOptions { max_seq: MAX_SEQ, prefill_chunk: chunk, kv_fraction: 0.5, ..Default::default() };
    Model::load(backend, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap()
}

#[test]
fn hybrid_reuse_shapes() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir(), backend()) else { return };
    let m = load(&dir, be);
    let (a, b) = prompts(&dir);
    let chunk = m.prefill_chunk();
    let (na, nb) = (a.len(), b.len());
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    eprintln!("backend {} chunk {chunk} A {na} tokens, B {nb} tokens, common {common}", m.backend.name());
    assert!(common == na && na <= chunk && nb > chunk, "the repro needs B = A + suffix, A within one chunk, B over it");

    // cold B (the engine's recurrent schedule): [0, chunk) + [chunk, nb - 1) + last
    let forced: Vec<u32> = {
        let mut st = m.new_state().unwrap();
        let mut f = Vec::new();
        let r = finish(&m, &mut st, &b, &[chunk, nb - 1], &[]);
        let mut l = r.tail[0].clone();
        for _ in 0..STEPS {
            let t = argmax_of(&l);
            f.push(t);
            l = logits_of(m.forward(&mut st, &[t], HeadMode::LastLogits, None).unwrap());
        }
        f
    };
    let cold = fresh(&m, &b, &[chunk, nb - 1], &forced);
    let mut ok = compare("cold repeat (determinism)", &cold, &fresh(&m, &b, &[chunk, nb - 1], &forced));
    // warm B's shape from a fresh state: [0, na - 1) + [na - 1, nb - 1) + last
    let warm = fresh(&m, &b, &[na - 1, nb - 1], &forced);
    ok &= compare("warm shape [0,4095)+[4095,4104)+1 vs cold", &cold, &warm);

    // the real path: A as the engine prefills it ([0, na - 1), snapshot, last), 32 decode steps, truncate to na - 1, B's rest
    let mut st = m.new_state().unwrap();
    m.forward_prefill(&mut st, &a[..na - 1], HeadMode::None, None).unwrap();
    st.snapshot(&*m.backend, intarsia_core::state::RECURRENT_SNAPSHOTS).unwrap();
    let mut l = logits_of(m.forward_prefill(&mut st, &a[na - 1..], HeadMode::LastLogits, None).unwrap());
    for _ in 0..32 {
        let t = argmax_of(&l);
        l = logits_of(m.forward(&mut st, &[t], HeadMode::LastLogits, None).unwrap());
    }
    let want = common.min(nb - 1);
    let r = m.reuse_len(&st, nb, want);
    eprintln!("reuse_len(plen {nb}, want {want}) = {r} (snapshots at {:?})", st.snapshots.iter().map(|s| s.len).collect::<Vec<_>>());
    st.truncate(na - 1, &*m.backend).unwrap();
    let real = finish(&m, &mut st, &b, &[nb - 1], &forced);
    let restore_ok = compare("restored A snapshot + rest vs warm shape (restore exactness)", &warm, &real);
    compare("restored A snapshot + rest vs cold", &cold, &real);

    // small passes after a common [0, chunk) prefix: is a short pass's output a function of its row count?
    for split in [vec![chunk + 4], vec![chunk + 3], vec![chunk + 1], vec![chunk + 2, chunk + 5]] {
        let mut cuts = vec![chunk];
        cuts.extend(&split);
        cuts.push(nb - 1);
        compare(&format!("cold, tail split at {split:?} vs one 8-row pass"), &cold, &fresh(&m, &b, &cuts, &forced));
    }
    // big pass of chunk - 1 rows then 1 + 8: separates the 4095-row pass from the 9-row one
    compare("[0,4095)+[4095,4096)+[4096,4104)+1 vs cold", &cold, &fresh(&m, &b, &[na - 1, chunk, nb - 1], &forced));
    compare("[0,4094)+[4094,4096)+[4096,4104)+1 vs cold", &cold, &fresh(&m, &b, &[na - 2, chunk, nb - 1], &forced));
    compare("[0,2048)+[2048,4096)+[4096,4104)+1 vs cold", &cold, &fresh(&m, &b, &[2048, chunk, nb - 1], &forced));

    if std::env::var("INTARSIA_PROBE_ASSERT").is_ok_and(|v| v == "1") {
        assert!(restore_ok, "the restored snapshot is not the state a fresh prefill of the same passes writes");
        assert!(ok, "B resumed at A's plen - 1 snapshot differs from cold B");
    }
}

fn collect(rx: &std::sync::mpsc::Receiver<GenEvent>) -> (Vec<u32>, usize) {
    let mut ids = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(900)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { cached_tokens, .. } => return (ids, cached_tokens),
            GenEvent::Error { message, .. } => panic!("request failed: {message}"),
        }
    }
}

#[test]
fn engine_hybrid_reuse_equals_cold() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir(), backend()) else { return };
    let (a, b) = prompts(&dir);
    let opts = EngineOptions { load: LoadOptions { max_seq: MAX_SEQ, kv_fraction: 0.5, ..Default::default() }, ..Default::default() };
    let req = |prompt: &[u32], n: usize| GenRequest { prompt: prompt.to_vec(), max_tokens: n, ignore_eos: true, ..Default::default() };
    let gen = 64;
    let reference = {
        let l = load_engine(&dir, be.clone(), &opts).unwrap();
        collect(&l.engine.submit(req(&b, gen)).unwrap()).0
    };
    let l = load_engine(&dir, be, &opts).unwrap();
    let (cold_a, _) = collect(&l.engine.submit(req(&a, 256)).unwrap());
    let (got, cached) = collect(&l.engine.submit(req(&b, gen)).unwrap());
    let flip = got.iter().zip(&reference).position(|(x, y)| x != y);
    eprintln!("B after A: cached {cached}, {}", flip.map_or("= cold".to_string(), |i| format!("DIFFERS from cold at token {i}")));
    assert_eq!(got, reference, "B resumed from A's state (cached {cached}) differs from a fresh engine's B");
    // A after B: B's cold schedule has no state at 4095 (its passes end at 4096 and 4104), so nothing; A once more then
    // resumes at its own plen - 1 snapshot, a cut of its cold schedule
    for (what, want_cached) in [("A after B", 0), ("A repeated", a.len() - 1)] {
        let (ta, ca) = collect(&l.engine.submit(req(&a, gen)).unwrap());
        let same = ta[..] == cold_a[..gen];
        eprintln!("{what}: cached {ca}, {}", if same { "= cold" } else { "DIFFERS from cold" });
        assert!(same, "{what} differs from cold A");
        assert_eq!(ca, want_cached, "{what}: cached tokens");
    }
}
