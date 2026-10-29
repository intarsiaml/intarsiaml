//! Lane prb (docs/results/2026-10-29-prb-prefix-reuse-probe.md): does a request that resumes from a slot's KV prefix compute the
//! bits of its cold run, on a backend other than CUDA? Lane bi8 (docs/results/2026-10-29-bi8-serve-batch-invariance.md) found that
//! on CUDA it did not (the repeated prompt's last token prefilled alone is a 1-row pass) and fixed it in `Model::reuse_len`.
//! The twin of `batch_cuda_real.rs`'s `real_qwen3_8b_prefix_reuse_probe` / `real_qwen3_8b_batched_engine_equals_the_cold_reference`
//! for Metal (`INTARSIA_PROBE_BACKEND=metal`, the default on macOS), HIP (`hip`) and cpu (`cpu`, minutes on an 8B).
//!
//! Opt-in: `INTARSIA_PROBE_DIR` = a dense attention checkpoint (Qwen3-8B BF16: `~/Models/Qwen3-8B`); unset prints SKIPPED and passes.
//! * `prefix_reuse_probe` (diagnostic, prints; `INTARSIA_PROBE_VERBOSE=1` for every split): is a prefill split (k, plen - k) bitwise
//!   the one-pass prefill, per prompt; the repeated-prompt reuse the old rule did (truncate to `plen - 1`, last token alone); the
//!   cross-prompt reuse.
//! * `reuse_rule_resumes_with_cold_bits`: the gate. For every prompt, a slot that held the same prompt (and its decode tokens), or
//!   another prompt sharing a prefix, resumes at `Model::reuse_len(.., want = min(common, plen - 1))`, exactly as `pick_slot` /
//!   `Seq::new` do; the logits of the resumed prefill and of four teacher-forced decode steps must be the cold run's, bit for bit.
//! * `engine_repeats_equal_the_cold_reference`: the same through the engines. A single-sequence engine and a batched one run the
//!   prompts in an order full of repeats; every request's tokens must be the cold reference's.
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::{Engine, FinishReason, GenEvent, GenRequest, Telemetry};
use intarsia_models::load::SchemedSource;
use intarsia_models::{load_engine, read_model_spec, EngineOptions};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

type Be = Arc<dyn intarsia_core::backend::Backend>;

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

/// `INTARSIA_PROBE_BACKEND` = metal (default on macOS) | hip | cpu.
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

/// One test at a time: each loads the checkpoint.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `INTARSIA_PROBE_CHUNK=N`: a prefill chunk of N rows instead of the recipe's (2048 on Metal), so short prompts have cold chunk
/// boundaries and a reuse rule that cuts only there reuses something (the default chunk is longer than every prompt here).
fn probe_chunk() -> Option<usize> {
    std::env::var("INTARSIA_PROBE_CHUNK").ok().and_then(|v| v.parse().ok())
}

fn load(dir: &PathBuf, backend: Be, max_seq: usize) -> Model {
    let (_, recipe, dims, _) = read_model_spec(dir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(dir).unwrap();
    let chunk = probe_chunk().unwrap_or_else(|| recipe.resolve_prefill_chunk(None, backend.name()).value);
    let opts = LoadOptions { max_seq, prefill_chunk: chunk, kv_fraction: 0.5, ..Default::default() };
    Model::load(backend, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap()
}

/// The 16 prompts of `scripts/htk/equal.py` (the SB8 / BI8 served-identity set).
const EQ_PROMPTS: [&str; 16] = [
    "The history of the printing press begins",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which",
    "In 1969, the Apollo 11 mission",
    "Explain, step by step, why the sky is blue. First,",
    "The three laws of thermodynamics state that",
    "A recipe for sourdough bread needs",
    "The capital of Australia is",
    "Once upon a time, in a small village by the sea,",
    "To install the package, run",
    "The Pythagorean theorem says",
    "In machine learning, overfitting happens when",
    "The French Revolution started because",
    "A haiku about autumn:",
    "The difference between TCP and UDP is",
    "Quantum entanglement is",
];

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

/// Greedy continuation of `st` from the logits row `first`: (tokens, logits rows incl. `first`).
fn greedy(m: &Model, st: &mut intarsia_core::state::SequenceState, first: Vec<f32>, n: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut rows = vec![first];
    let mut toks = Vec::new();
    for _ in 0..n {
        let t = argmax_of(rows.last().unwrap());
        toks.push(t);
        rows.push(logits_of(m.forward(st, &[t], HeadMode::LastLogits, None).unwrap()));
    }
    (toks, rows)
}

/// The prompt's cold run: one `forward` over the whole prompt, then `gen` greedy steps.
struct Cold {
    toks: Vec<u32>,
    rows: Vec<Vec<f32>>,
}

fn ids_of(dir: &PathBuf) -> Vec<Vec<u32>> {
    use intarsia_core::engine::TextCodec;
    let codec = intarsia_models::HfCodec::load(dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    EQ_PROMPTS.iter().map(|p| codec.encode(p, false).unwrap()).collect()
}

fn cold_runs(m: &Model, ids: &[Vec<u32>], gen: usize) -> Vec<Cold> {
    ids.iter()
        .map(|p| {
            let mut st = m.new_state().unwrap();
            let l = logits_of(m.forward(&mut st, p, HeadMode::LastLogits, None).unwrap());
            let (toks, rows) = greedy(m, &mut st, l, gen);
            Cold { toks, rows }
        })
        .collect()
}

const FORCED: usize = 4;

/// Resume `st` (which holds `held` as its tokens) for prompt `p` the way the engines do: `want = min(common prefix, plen - 1)`,
/// then `Model::reuse_len`, truncate there, prefill the rest and teacher-force `FORCED` decode steps. Returns (resume point, rows).
fn resume(m: &Model, st: &mut intarsia_core::state::SequenceState, p: &[u32], cold: &Cold) -> (usize, Vec<Vec<f32>>) {
    let common = st.tokens.iter().zip(p).take_while(|(a, b)| a == b).count();
    let want = common.min(p.len() - 1);
    let r = m.reuse_len(st, p.len(), want);
    st.truncate(r, &*m.backend).unwrap();
    let mut rows = vec![logits_of(m.forward(st, &p[r..], HeadMode::LastLogits, None).unwrap())];
    for &t in &cold.toks[..FORCED] {
        rows.push(logits_of(m.forward(st, &[t], HeadMode::LastLogits, None).unwrap()));
    }
    (r, rows)
}

fn first_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[test]
fn prefix_reuse_probe() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir(), backend()) else { return };
    let verbose = std::env::var("INTARSIA_PROBE_VERBOSE").is_ok_and(|v| v == "1");
    let m = load(&dir, be, 2048);
    eprintln!("backend {} prefix_reuse_any_cut {} avoid_single_row_prefill {} prefill_chunk {}", m.backend.name(), m.backend.prefix_reuse_any_cut(), m.backend.avoid_single_row_prefill(), m.prefill_chunk());
    let ids = ids_of(&dir);
    let gen: usize = std::env::var("INTARSIA_PROBE_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(64).max(FORCED);
    let cold = cold_runs(&m, &ids, gen);
    // the old rule: the slot held the prompt + its decode tokens; truncate to plen - 1 and prefill the last token alone
    let mut self_diff = 0;
    for (i, p) in ids.iter().enumerate() {
        let mut st = m.new_state().unwrap();
        let l = logits_of(m.forward(&mut st, p, HeadMode::LastLogits, None).unwrap());
        let _ = greedy(&m, &mut st, l, gen);
        st.truncate(p.len() - 1, &*m.backend).unwrap();
        let l = logits_of(m.forward(&mut st, &p[p.len() - 1..], HeadMode::LastLogits, None).unwrap());
        let (rt, rr) = greedy(&m, &mut st, l, gen);
        let flip = rt.iter().zip(&cold[i].toks).position(|(a, b)| a != b);
        self_diff += (rr[0] != cold[i].rows[0]) as usize;
        eprintln!("OLDRULE {i} plen {}: logits row0 {} (maxabs {:.3e}), first logits diff row {:?}, first token flip {:?}", p.len(), if rr[0] == cold[i].rows[0] { "EQUAL" } else { "DIFFER" }, maxabs(&rr[0], &cold[i].rows[0]), first_diff(&rr, &cold[i].rows), flip);
    }
    eprintln!("OLDRULE summary: {self_diff} / {} prompts' row 0 differ from cold", ids.len());
    // every split k
    let (mut tail1_diff, mut tail1, mut other_diff, mut other) = (0, 0, 0, 0);
    for (i, p) in ids.iter().enumerate() {
        let mut line = String::new();
        for k in 1..p.len() {
            let mut st = m.new_state().unwrap();
            m.forward(&mut st, &p[..k], HeadMode::None, None).unwrap();
            let mut rows = vec![logits_of(m.forward(&mut st, &p[k..], HeadMode::LastLogits, None).unwrap())];
            for &t in &cold[i].toks[..FORCED] {
                rows.push(logits_of(m.forward(&mut st, &[t], HeadMode::LastLogits, None).unwrap()));
            }
            let fd = first_diff(&rows, &cold[i].rows[..=FORCED]);
            let tail = p.len() - k;
            if tail == 1 {
                tail1 += 1;
                tail1_diff += fd.is_some() as usize;
            } else {
                other += 1;
                other_diff += fd.is_some() as usize;
            }
            line.push_str(&format!(" k{k}(tail {tail}):{}", fd.map_or("=".to_string(), |r| format!("x{r}"))));
        }
        if verbose {
            eprintln!("SPLIT {i} plen {}:{line}", p.len());
        }
    }
    eprintln!("SPLIT summary: tail 1 (the (plen-1, 1) split): {tail1_diff} / {tail1} differ from one pass; every other split: {other_diff} / {other} differ");
    // cross-prompt reuse at the longest shared prefix, with the old rule's cut
    let (mut cross, mut cross_diff) = (0, 0);
    for (i, p) in ids.iter().enumerate() {
        for (j, q) in ids.iter().enumerate() {
            let c = p.iter().zip(q).take_while(|(a, b)| a == b).count().min(p.len() - 1);
            if i == j || c == 0 {
                continue;
            }
            let mut st = m.new_state().unwrap();
            let l = logits_of(m.forward(&mut st, q, HeadMode::LastLogits, None).unwrap());
            let _ = greedy(&m, &mut st, l, FORCED);
            st.truncate(c, &*m.backend).unwrap();
            let mut rows = vec![logits_of(m.forward(&mut st, &p[c..], HeadMode::LastLogits, None).unwrap())];
            for &t in &cold[i].toks[..FORCED] {
                rows.push(logits_of(m.forward(&mut st, &[t], HeadMode::LastLogits, None).unwrap()));
            }
            let fd = first_diff(&rows, &cold[i].rows[..=FORCED]);
            cross += 1;
            cross_diff += fd.is_some() as usize;
            eprintln!("CROSS {i}<-{j} shared {c} of {}: first logits diff row {fd:?}", p.len());
        }
    }
    eprintln!("CROSS summary: {cross_diff} / {cross} differ");
}

#[test]
fn reuse_rule_resumes_with_cold_bits() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir(), backend()) else { return };
    let m = load(&dir, be, 2048);
    let ids = ids_of(&dir);
    let cold = cold_runs(&m, &ids, FORCED);
    let mut bad = Vec::new();
    let mut cuts = Vec::new();
    // a slot that ran prompt i (and decoded 64 tokens), asked for prompt i again
    for (i, p) in ids.iter().enumerate() {
        let mut st = m.new_state().unwrap();
        let l = logits_of(m.forward(&mut st, p, HeadMode::LastLogits, None).unwrap());
        let _ = greedy(&m, &mut st, l, FORCED);
        let (r, rows) = resume(&m, &mut st, p, &cold[i]);
        cuts.push((p.len(), r));
        if let Some(row) = first_diff(&rows, &cold[i].rows[..=FORCED]) {
            bad.push(format!("repeat of prompt {i} (plen {}, resumed at {r}): first differing row {row}, maxabs {:.3e}", p.len(), maxabs(&rows[row], &cold[i].rows[row])));
        }
    }
    // a slot that ran prompt j, asked for prompt i sharing a prefix with it
    for (i, p) in ids.iter().enumerate() {
        for (j, q) in ids.iter().enumerate() {
            let shared = p.iter().zip(q).take_while(|(a, b)| a == b).count();
            if i == j || shared == 0 {
                continue;
            }
            let mut st = m.new_state().unwrap();
            let l = logits_of(m.forward(&mut st, q, HeadMode::LastLogits, None).unwrap());
            let _ = greedy(&m, &mut st, l, FORCED);
            let (r, rows) = resume(&m, &mut st, p, &cold[i]);
            if let Some(row) = first_diff(&rows, &cold[i].rows[..=FORCED]) {
                bad.push(format!("prompt {i} after {j} (shared {shared}, resumed at {r}): first differing row {row}, maxabs {:.3e}", maxabs(&rows[row], &cold[i].rows[row])));
            }
        }
    }
    eprintln!("reuse cuts (plen, resumed at) of the repeated prompts: {cuts:?}");
    assert!(bad.is_empty(), "{} resumed prefills differ from the cold run:\n{}", bad.len(), bad.join("\n"));
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    reason: Option<FinishReason>,
}

/// The request's output and the prompt tokens it found cached (the KV prefix it resumed from).
fn collect(rx: &Receiver<GenEvent>) -> (Out, usize) {
    let mut ids = vec![];
    loop {
        match rx.recv_timeout(Duration::from_secs(900)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { finish_reason, cached_tokens, .. } => return (Out { ids, reason: Some(finish_reason) }, cached_tokens),
            GenEvent::Error { message, .. } => panic!("request failed: {message}"),
            _ => {}
        }
    }
}

/// The served check: a repeated prompt on the same slot must give the cold result. Reference: each prompt alone in order on one
/// fresh single-sequence engine (no two consecutive prompts share a first token, so nothing is reused); then the single-sequence
/// engine in an order full of repeats, and a 4-slot batched engine, whose slot for a prompt it ran before is idle again. The repeats
/// include prompts 3, 4, 5, 9, 13 and 15, the ones whose Metal greedy tokens flip within 64 steps under the old rule (the probe's
/// OLDRULE lines: first flips at tokens 37, 49, 63, 9, 11, 28), hence `GEN` = 64. Prints the prompt tokens each request found cached.
#[test]
fn engine_repeats_equal_the_cold_reference() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir(), backend()) else { return };
    const GEN: usize = 64;
    let mk = |max_batch: Option<usize>| EngineOptions { load: LoadOptions { max_seq: 2048, kv_fraction: 0.5, ..Default::default() }, max_batch, prefill_chunk: probe_chunk(), ..Default::default() };
    let req = |e: &dyn Engine, i: usize| GenRequest { prompt: e.tokenize(EQ_PROMPTS[i], false).unwrap(), max_tokens: GEN, ignore_eos: true, ..Default::default() };
    let reference: Vec<Out> = {
        let l = load_engine(&dir, be.clone(), &mk(None)).unwrap();
        let reqs: Vec<GenRequest> = (0..EQ_PROMPTS.len()).map(|i| req(&*l.engine, i)).collect();
        for w in reqs.windows(2) {
            assert_ne!(w[0].prompt[0], w[1].prompt[0], "consecutive reference prompts must not share a prefix");
        }
        reqs.into_iter().map(|r| collect(&l.engine.submit(r).unwrap()).0).collect()
    };
    assert!(reference.iter().all(|o| o.ids.len() == GEN));
    let mut bad = Vec::new();
    {
        let l = load_engine(&dir, be.clone(), &mk(None)).unwrap();
        for (n, &i) in [4usize, 4, 5, 0, 5, 8, 8, 7, 12, 5, 14, 1, 1, 9, 9, 13, 13, 15, 15, 3, 3, 3].iter().enumerate() {
            let (got, cached) = collect(&l.engine.submit(req(&*l.engine, i)).unwrap());
            eprintln!("single-sequence engine, request {n} (prompt {i}): cached {cached}, {}", if got == reference[i] { "= cold" } else { "DIFFERS from cold" });
            if got != reference[i] {
                bad.push(format!("single-sequence engine, request {n} (prompt {i})"));
            }
        }
    }
    {
        let l = load_engine(&dir, be.clone(), &mk(Some(4))).unwrap();
        let rounds = [vec![3usize, 4, 5, 9], vec![3, 4, 5, 9], vec![9, 5, 4, 3], vec![13, 15, 13, 15], vec![13, 15, 3, 9], vec![4, 5, 8, 4, 5, 8]];
        for (round, idx) in rounds.iter().enumerate() {
            let rxs: Vec<_> = idx.iter().map(|&i| l.engine.submit(req(&*l.engine, i)).unwrap()).collect();
            for (&i, rx) in idx.iter().zip(&rxs) {
                let (got, cached) = collect(rx);
                eprintln!("4-slot batched engine, round {round}, prompt {i}: cached {cached}, {}", if got == reference[i] { "= cold" } else { "DIFFERS from cold" });
                if got != reference[i] {
                    bad.push(format!("4-slot batched engine, round {round}, prompt {i}"));
                }
            }
        }
    }
    assert!(bad.is_empty(), "requests whose tokens differ from the cold reference: {bad:?}");
}

/// Diagnostic and gate for the shape of a backend's pass-size dependence: a 700-token prompt prefilled in one pass against the
/// same prompt split in two (k, n - k) and in three, at k on and around the tile edges (16, 32, 64, 128, 256 rows). Prints every
/// split that differs; with `INTARSIA_PROBE_LONG_ASSERT=1` fails unless the only differing splits are the ones with a 1-row pass.
#[test]
fn prefix_reuse_long_splits() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir(), backend()) else { return };
    use intarsia_core::engine::TextCodec;
    let m = load(&dir, be, 2048);
    let codec = intarsia_models::HfCodec::load(&dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    let text = EQ_PROMPTS.iter().cycle().take(160).cloned().collect::<Vec<_>>().join(" ");
    let mut p = codec.encode(&text, false).unwrap();
    p.truncate(700);
    let n = p.len();
    assert!(n >= 600, "prompt of {n} tokens");
    let steps = [11u32, 12, 13];
    let run = |cuts: &[usize]| {
        let mut st = m.new_state().unwrap();
        let mut at = 0;
        for &c in cuts {
            m.forward(&mut st, &p[at..c], HeadMode::None, None).unwrap();
            at = c;
        }
        let mut rows = vec![logits_of(m.forward(&mut st, &p[at..], HeadMode::LastLogits, None).unwrap())];
        for &s in &steps {
            rows.push(logits_of(m.forward(&mut st, &[s], HeadMode::LastLogits, None).unwrap()));
        }
        rows
    };
    let cold = run(&[]);
    let mut ks: Vec<usize> = vec![1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300, 511, 512, 513];
    ks.extend([n - 33, n - 32, n - 17, n - 16, n - 9, n - 8, n - 5, n - 3, n - 2, n - 1]);
    ks.retain(|&k| k >= 1 && k < n);
    let mut splits: Vec<Vec<usize>> = ks.iter().map(|&k| vec![k]).collect();
    splits.extend([vec![2, 40], vec![33, 100], vec![64, 65 + 64], vec![5, 6, 300], vec![128, 129, 400], vec![2, 4, 8, 16, 32, 64, 128, 256], vec![100, 400, n - 2]]);
    let mut bad = Vec::new();
    for cuts in &splits {
        let mut bounds = vec![0];
        bounds.extend(cuts.iter().copied());
        bounds.push(n);
        let sizes: Vec<usize> = bounds.windows(2).map(|w| w[1] - w[0]).collect();
        let one_row = sizes.iter().any(|&s| s == 1);
        let rows = run(cuts);
        let fd = first_diff(&rows, &cold);
        eprintln!("LONG n {n} passes {sizes:?}: {}", fd.map_or("=".to_string(), |r| format!("DIFFER row {r} maxabs {:.3e}", maxabs(&rows[r], &cold[r]))));
        if fd.is_some() != one_row {
            bad.push(format!("{sizes:?}: {}", if fd.is_some() { "differs without a 1-row pass" } else { "equal despite a 1-row pass" }));
        }
    }
    eprintln!("LONG summary: {} of {} splits not as the 1-row-pass rule says: {bad:?}", bad.len(), splits.len());
    if std::env::var("INTARSIA_PROBE_LONG_ASSERT").is_ok_and(|v| v == "1") {
        assert!(bad.is_empty(), "{bad:?}");
    }
}
