//! CUDA (and, with `INTARSIA_BATCH_TEST_BACKEND=hip`, HIP) gate G-C and the engine-level identity on the real models (docs/results/2026-10-23-rb2-cuda-batching.md):
//!
//! * `real_35b_batched_decode_is_bitwise_the_solo_step`: Qwen3.6-35B-A3B NVFP4. `Model::forward_batch` of N = 2, 3, 4, 8, 16
//!   sequences at mixed context lengths, greedy and seeded-sampled rows, 130 consecutive steps each, must return for every
//!   row the bits of that sequence's solo decode step (logits bit for bit, the on-device argmax token for token) and leave
//!   identical state (`common/batch_identity.rs`); decode rows beside prefill chunks too.
//! * `real_35b_batched_engine_equals_the_local_engine`: the whole `BatchedEngine` (scheduler, slots, snapshots, prefill
//!   chunks, sampling) over greedy and seeded-sampled requests: at one stream it equals `LocalEngine` token for token, and
//!   with 4 and 16 requests in flight each request's tokens still equal its solo run. (Prefill of a busy engine runs in
//!   smaller chunks than a solo one: for the greedy prompts below the chunk schedule is the same, see the test.)
//! * `real_qwen3_8b_batched_decode_is_bitwise_the_solo_step`: the dense bf16 model on the same kernels (CUDA, and HIP
//!   since #154 dropped `hip_batch_needs_hybrid`).
//!
//! Tensor parallelism (lane tp2/batch): with `INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_BATCH_TEST_TP=2` the models load as an
//! in-process 2-way group on HIP devices 0 and 1 (peer-to-peer communicator, row-local fusion on and the vocab-parallel head
//! as `intarsia serve --tp 2 --backend hip` loads them; `INTARSIA_TP_VOCAB=0` keeps the replicated head) and the same gates
//! compare batched tp=2 steps against solo tp=2 steps.
//!
//! Opt-in (loads 20 GB onto the GPU: take a fleetq slot on the Spark first): `INTARSIA_QWEN36_DIR` (35B checkpoint) and
//! `INTARSIA_QWEN3_DIR` (8B); a test whose variable is unset prints SKIPPED and passes, also under `INTARSIA_REQUIRE_GPU=1`
//! (a checkpoint left out is not a missing GPU); one set to a directory without config.json, or without a usable GPU,
//! panics under `INTARSIA_REQUIRE_GPU=1`. The tests run one at a time ([`serial`]).
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::{Engine, FinishReason, GenEvent, GenRequest, Telemetry};
use intarsia_models::load::SchemedSource;
use intarsia_models::{load_engine, load_engine_tp, read_model_spec, EngineOptions, Loaded};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

#[path = "common/batch_identity.rs"]
mod batch_identity;

fn skip(why: &str) {
    eprintln!("SKIPPED: {why}");
    if std::env::var("INTARSIA_REQUIRE_GPU").is_ok_and(|v| v == "1") {
        panic!("SKIPPED ({why}) but INTARSIA_REQUIRE_GPU=1");
    }
}

fn model_dir(var: &str) -> Option<PathBuf> {
    match std::env::var_os(var).map(PathBuf::from) {
        Some(d) if d.join("config.json").exists() => Some(d),
        Some(d) => {
            skip(&format!("{var}={} has no config.json", d.display()));
            None
        }
        None => {
            eprintln!("SKIPPED: {var} not set");
            None
        }
    }
}

/// The GPU backend: CUDA, or HIP with `INTARSIA_BATCH_TEST_BACKEND=hip` (lane hb, docs/results/2026-10-23-hb-hip-hybrid-batching.md).
fn cuda() -> Option<Arc<dyn intarsia_core::backend::Backend>> {
    let hip = std::env::var("INTARSIA_BATCH_TEST_BACKEND").is_ok_and(|v| v == "hip");
    let made = if hip { intarsia_backend_hip::create(0) } else { intarsia_backend_cuda::create(0) };
    match made {
        Ok(b) => Some(b),
        Err(e) => {
            skip(&format!("no usable {} device ({e})", if hip { "HIP" } else { "CUDA" }));
            None
        }
    }
}

/// The tests of this file one at a time (every test takes it first): each loads a real checkpoint with most of the free
/// device memory for its KV cache (`kv_fraction` 0.95), so two at once under the default parallel test runner do not fit.
/// A failed test does not poison it.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The model of `dir` on the GPU with batching enabled for `rows` sequences of `max_seq` tokens.
/// `INTARSIA_BATCH_TEST_TP=2` (HIP only): the tensor-parallel degree the models load with (default 1).
fn tp() -> usize {
    let tp = std::env::var("INTARSIA_BATCH_TEST_TP").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    assert!(tp == 1 || (tp == 2 && std::env::var("INTARSIA_BATCH_TEST_BACKEND").is_ok_and(|v| v == "hip")), "INTARSIA_BATCH_TEST_TP=2 needs INTARSIA_BATCH_TEST_BACKEND=hip");
    tp
}

/// The load options `intarsia serve --tp 2 --backend hip` adds at tp=2: row-local fusion, the vocab-parallel head (unless
/// `INTARSIA_TP_VOCAB=0`). Unchanged at tp=1.
fn with_tp(mut o: LoadOptions) -> LoadOptions {
    if tp() > 1 {
        o.tp_fusion = true;
        o.tp_vocab = !std::env::var("INTARSIA_TP_VOCAB").is_ok_and(|v| v == "0");
    }
    o
}

/// The HIP group of devices 0 and 1 (backends and the peer-to-peer communicator).
fn hip_group() -> (Vec<Arc<dyn intarsia_core::backend::Backend>>, Vec<Arc<dyn intarsia_core::Comm>>) {
    intarsia_backend_hip::create_group(&[0, 1]).expect("two HIP devices with peer access")
}

/// `load_engine`, or at tp=2 `load_engine_tp` over the HIP group (`be` is then unused).
fn engine(dir: &PathBuf, be: &Arc<dyn intarsia_core::backend::Backend>, opts: &EngineOptions) -> Loaded {
    if tp() == 1 {
        return load_engine(dir, be.clone(), opts).unwrap();
    }
    let (bs, comms) = hip_group();
    let opts = EngineOptions { load: with_tp(opts.load.clone()), ..opts.clone() };
    load_engine_tp(dir, bs, comms, &opts).unwrap()
}

fn load(dir: &PathBuf, backend: Arc<dyn intarsia_core::backend::Backend>, rows: usize, max_seq: usize) -> Model {
    load_with(dir, backend, rows, max_seq, 0).0
}

fn load_with(
    dir: &PathBuf,
    backend: Arc<dyn intarsia_core::backend::Backend>,
    rows: usize,
    max_seq: usize,
    profile_every: u32,
) -> (Model, Arc<Telemetry>) {
    try_load_with(dir, backend, rows, max_seq, profile_every).unwrap()
}

/// [`load_with`], with the refusal of `Model::allow_recurrent_batch` returned instead of panicking.
fn try_load_with(
    dir: &PathBuf,
    backend: Arc<dyn intarsia_core::backend::Backend>,
    rows: usize,
    max_seq: usize,
    profile_every: u32,
) -> intarsia_core::Result<(Model, Arc<Telemetry>)> {
    let (_, recipe, dims, _) = read_model_spec(dir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(dir).unwrap();
    let chunk = recipe.resolve_prefill_chunk(None, backend.name()).value;
    let opts = with_tp(LoadOptions {
        max_seq,
        prefill_chunk: chunk,
        profile_every,
        // the Spark's `free` after the weights is the page-cache-reduced figure (about 2 GB on a shared box): take most of it
        kv_fraction: 0.95,
        // INTARSIA_TEST_MXFP4_ACT=fp8|none|dynamic: the MXFP4 activation mode (default: the recipe's / the checkpoint's)
        mxfp4_act: std::env::var("INTARSIA_TEST_MXFP4_ACT").ok().map(|v| intarsia_core::quant::Mxfp4Act::parse(&v).unwrap()),
        ..Default::default()
    });
    let tel = Arc::new(Telemetry::new());
    let mut m = if tp() == 1 {
        Model::load(backend, &recipe, &dims, &weights, &opts, tel.clone()).unwrap()
    } else {
        drop(backend);
        let (bs, comms) = hip_group();
        let m = Model::load_tp(bs, comms, &recipe, &dims, &weights, &opts, tel.clone()).unwrap();
        eprintln!("tensor parallel x{}: vocab-parallel head {}", m.tp_world(), m.vocab_parallel());
        m
    };
    assert!(m.batch_capable(), "the GPU backend registers the row-exact kernels");
    eprintln!("plan before batching: {:?}", m.plan);
    eprintln!("recurrent bytes per state {}", m.recurrent_bytes_per_state());
    m.allow_recurrent_batch()?;
    m.enable_batch(rows).unwrap();
    // plan resolution dry-runs every batch size at both ends of the context: a batched op that would pick a kernel that is
    // neither the solo step's nor a row-exact one fails here (`plan::check_batch_ops`)
    m.configure_plan(&Default::default(), None).unwrap();
    eprintln!(
        "batch plan: {} KV tokens per sequence, {} KV bytes in all, {} bytes per token; notes {:?}",
        m.plan.kv_cap_tokens,
        m.plan.kv_total_bytes,
        m.kv_bytes_per_token(),
        m.plan.notes
    );
    assert!(m.plan.kv_cap_tokens >= max_seq, "{} tokens per sequence", m.plan.kv_cap_tokens);
    Ok((m, tel))
}

fn gate(m: &Model, lens: Vec<usize>, batches: Vec<usize>, steps: usize, vocab_range: (u32, u32)) {
    let expect: usize = batches.iter().sum::<usize>() * steps;
    let t0 = std::time::Instant::now();
    let n = batch_identity::run(m, &batch_identity::Cfg { lens, batches, steps, vocab_range });
    eprintln!("{n} (row, step) pairs bitwise identical in {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!(n, expect);
}

/// Lane moe16b: wall time of the 35B's batched decode step (graphs on, device argmax, as in serve) at B = 8 and 16, three runs of 60
/// steps after 20 warm-up steps. `--ignored`, inside a fleetq slot quiet on the GPU; prints `STEP B=.. ms ..`. The side-stream knobs
/// (`INTARSIA_MOE_SHARED_STREAM`, `INTARSIA_SIDE_PROJ`) are read once per process: one variant per run.
#[test]
#[ignore = "micro-timing"]
fn timing_35b_batched_step() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    for b in [8usize, 16] {
        let mut sts: Vec<_> = (0..b)
            .map(|i| {
                let mut st = m.new_state().unwrap();
                let p: Vec<u32> = (0..64).map(|k| 1000 + ((i * 977 + k * 131) % 29000) as u32).collect();
                m.forward(&mut st, &p, HeadMode::None, None).unwrap();
                st
            })
            .collect();
        let mut ids: Vec<u32> = (0..b).map(|i| 2000 + i as u32).collect();
        let modes = vec![HeadMode::LastArgmax; b];
        let mut run = |n: usize| -> f64 {
            let t = std::time::Instant::now();
            for _ in 0..n {
                let mut refs: Vec<_> = sts.iter_mut().collect();
                let out = m.forward_batch(&mut refs, &ids, &modes).unwrap();
                for (r, o) in out.iter().enumerate() {
                    if let StepOut::Token(t) = o {
                        ids[r] = *t;
                    }
                }
            }
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        };
        run(20);
        let ms: Vec<f64> = (0..3).map(|_| run(60)).collect();
        println!("STEP B={b} ms {:.3} {:.3} {:.3}", ms[0], ms[1], ms[2]);
    }
}

#[test]
fn real_35b_batched_decode_is_bitwise_the_solo_step() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    eprintln!("batch plan notes: {:?}", m.plan.notes);
    // N = 2, 3, 4 (the gate), then 8 and 16; 130 consecutive batched steps per size on the same, growing states
    let lens = vec![5, 37, 130, 300, 17, 64, 9, 222, 400, 3, 77, 150, 28, 11, 90, 45];
    // on HIP (lane hb) every N from 2 to the cap, not only the sizes the CUDA gate samples
    let hip = std::env::var("INTARSIA_BATCH_TEST_BACKEND").is_ok_and(|v| v == "hip");
    let sizes: Vec<usize> = if hip { (2..=16).collect() } else { vec![2, 3, 4, 8, 16] };
    gate(&m, lens, sizes, 130, (1000, 30000));
    // decode rows beside prefill chunks (the engine's busy chunk is 256): the decode rows stay bitwise, the prefilled state
    // equals the prompt prefilled alone with the same chunks
    let n = batch_identity::run_with_prefill(&m, &[37, 130, 300, 17], 700, 256, (1000, 30000));
    assert_eq!(n, 4 * 2 * 700usize.div_ceil(256));
    // lane moe16c: with neither side-stream knob set (both default on), a single-rank CUDA run must take the side-stream forks
    let forks = intarsia_core::model::SIDE_FORKS_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let unset = ["INTARSIA_MOE_SHARED_STREAM", "INTARSIA_SIDE_PROJ"].iter().all(|k| std::env::var_os(k).is_none());
    eprintln!("side-stream forks: {forks} (knobs unset: {unset})");
    if unset && !hip && tp() == 1 {
        assert!(forks > 0, "the default-on side-stream knobs took no fork");
    }
    if tp() == 2 {
        // the batched steps ran through recordings on BOTH ranks, in lockstep (graphs are on by default, as in serve)
        let (g0, g1) = (m.graph_stats(), m.rank_model(1).unwrap().graph_stats());
        eprintln!("tp=2 batched graphs: rank 0 {g0:?}, rank 1 {g1:?}");
        assert!(g0.batch_replays > 0, "rank 0 never replayed a batched graph: {g0:?}");
        assert_eq!((g0.batch_captures, g0.batch_replays), (g1.batch_captures, g1.batch_replays), "the peer did not follow rank 0");
    }
}

/// Print both ranks' collective ledgers (`INTARSIA_TP_COLL_TRACE=1`, `Model::coll_trace_dump`): the first line where they differ, then the last
/// 40 lines of each. The first differing pass is the step the ranks decided differently.
/// Debugging (w26): where every host thread of this process sits. A signal handler on each thread prints its backtrace (a thread blocked in
/// an uninterruptible driver call prints nothing: its kernel wait channel is listed first). Used at a hang to tell a device-side spin from a host-side
/// deadlock (one rank's host thread inside hipMalloc, a sync or a module load while the other rank's ar2 kernel spins).
#[cfg(target_os = "linux")]
fn dump_host_stacks() {
    extern "C" fn on_signal(_: libc::c_int) {
        // SAFETY: gettid has no preconditions.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) };
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("STACK tid {tid}:\n{bt}");
    }
    // SAFETY: installing a handler for a signal this test process does not otherwise use.
    unsafe {
        libc::signal(libc::SIGUSR2, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
    }
    let pid = std::process::id() as libc::c_long;
    let me = unsafe { libc::syscall(libc::SYS_gettid) };
    let Ok(rd) = std::fs::read_dir("/proc/self/task") else { return };
    for e in rd.flatten() {
        let Some(tid) = e.file_name().to_str().and_then(|t| t.parse::<libc::c_long>().ok()) else { continue };
        let rd = |f: &str| std::fs::read_to_string(format!("/proc/self/task/{tid}/{f}")).unwrap_or_default().trim().replace('\n', " ");
        eprintln!("THREAD {tid}{}: comm {:?} wchan {:?} state {:?}", if tid == me { " (this one)" } else { "" }, rd("comm"), rd("wchan"), rd("stat").split(' ').nth(2).unwrap_or(""));
        if tid != me {
            // SAFETY: tgkill to a thread of this process.
            unsafe { libc::syscall(libc::SYS_tgkill, pid, tid, libc::SIGUSR2) };
            std::thread::sleep(Duration::from_millis(150));
        }
    }
}

/// `INTARSIA_REPRO_WARMUP=1` (lane tp2, w26): one eager solo step and one eager batched step of 2 rows before anything else, with the graphs off, so that
/// every lazy allocation and code-object load of the first step of a fresh model happens before the first collective of the measured steps.
fn maybe_warm(m: &Model) {
    if std::env::var("INTARSIA_REPRO_WARMUP").is_ok_and(|v| v == "1") {
        m.set_graphs(false);
        batch_identity::run(m, &batch_identity::Cfg { lens: vec![5, 7], batches: vec![2], steps: 1, vocab_range: (1000, 30000) });
        m.set_graphs(true);
    }
}

fn dump_ledgers(m: &Model, why: &str) {
    let a = m.coll_trace_dump();
    let b = m.rank_model(1).map(|p| p.coll_trace_dump()).unwrap_or_default();
    eprintln!("LEDGER {why}: rank 0 {} lines, rank 1 {} lines", a.len(), b.len());
    let strip = |l: Option<&String>| l.map(|x| x.split(" dev=").next().unwrap_or("").to_string());
    match (0..a.len().max(b.len())).find(|&i| strip(a.get(i)) != strip(b.get(i))) {
        Some(i) => eprintln!("LEDGER first difference at line {i}:\n  rank 0: {:?}\n  rank 1: {:?}", a.get(i), b.get(i)),
        None => eprintln!("LEDGER the ledgers are identical"),
    }
    for (r, l) in [(0, &a), (1, &b)] {
        for line in &l[l.len().saturating_sub(if std::env::var("INTARSIA_REPRO_CTRS").is_ok_and(|v| v == "1") { 400 } else { 40 })..] {
            eprintln!("LEDGER rank {r}: {line}");
        }
    }
    #[cfg(target_os = "linux")]
    if why == "watchdog" || std::env::var("INTARSIA_REPRO_STACKS").is_ok_and(|v| v == "1") {
        dump_host_stacks();
    }
    // Last: the read may block when the device is already in an exception state (w20: it took the process down before any ledger line printed).
    eprintln!("LEDGER device state: {}", m.comm_debug_state().unwrap_or_else(|| "none".into()));
}

/// Run `f`; if it takes longer than `INTARSIA_REPRO_WATCHDOG` seconds (25) the ranks are probably stuck in a collective: dump the ledgers and
/// leave (the bounded wait would trap after about a minute and take the evidence with it).
fn with_watchdog<R>(m: &Model, f: impl FnOnce() -> R) -> R {
    let wd: u64 = std::env::var("INTARSIA_REPRO_WATCHDOG").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
    if wd == 0 {
        return f(); // INTARSIA_REPRO_WATCHDOG=0: no watchdog thread, no catch_unwind (the plain run of w17/w18)
    }
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|sc| {
        sc.spawn(|| {
            let t0 = std::time::Instant::now();
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(250));
                if t0.elapsed().as_secs() > wd {
                    eprintln!("REPRO watchdog: a rep took more than {wd} s, the ranks are stuck");
                    dump_ledgers(m, "watchdog");
                    eprintln!("REPRO failures 1 (hang)");
                    std::process::exit(4);
                }
            }
        });
        // A mismatch panics inside `f`: dump the ledgers at once (the idle device is still in the state the mismatch left it), then go on unwinding.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        match r {
            Ok(r) => r,
            Err(e) => {
                dump_ledgers(m, "mismatch");
                std::panic::resume_unwind(e)
            }
        }
    })
}

/// `INTARSIA_BATCH_TEST_TP=2` only (lane tp2, w18): w18 showed the size-loop mismatch with the batched graphs off, with launch blocking and
/// with the peers' graphs prepared first, and the SOLO logit of the failing row differed between runs (2.473 vs 2.125 for the same
/// sequence and context): the solo step of a fresh state is wrong too. This takes the batched step out: `INTARSIA_REPRO_STATES` (60) fresh
/// states of one prompt each prefilled and decoded one token (the first decode step of a state records its decode graph on both ranks), all
/// logits must be bit-identical to the first. `INTARSIA_REPRO_BATCH_BETWEEN=1` runs one batched step (3 rows, other prompts) between states;
/// `INTARSIA_REPRO_NO_GRAPHS=1` runs everything eager.
#[test]
#[ignore]
fn real_35b_tp2_solo_fresh_states_repro() {
    if tp() != 2 {
        eprintln!("SKIPPED: INTARSIA_BATCH_TEST_TP=2 only");
        return;
    }
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let num = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (reps, states) = (num("INTARSIA_REPRO_REPS", 3), num("INTARSIA_REPRO_STATES", 60));
    let between = std::env::var("INTARSIA_REPRO_BATCH_BETWEEN").is_ok_and(|v| v == "1");
    let prompt: Vec<u32> = (0..40u32).map(|i| 1000 + (i * 7919) % 20000).collect();
    for rep in 0..reps {
        let m = load(&dir, be.clone(), 16, 2048);
        maybe_warm(&m);
        if std::env::var("INTARSIA_REPRO_NO_GRAPHS").is_ok_and(|v| v == "1") {
            m.set_graphs(false);
        }
        let mut first: Option<Vec<u32>> = None;
        for k in 0..states {
            let probe = |what: &str| {
                if std::env::var("INTARSIA_REPRO_CTRS").is_ok_and(|v| v == "1") {
                    eprintln!("CTRS rep {rep} state {k} {what}: {}", m.comm_debug_state().unwrap_or_default());
                }
            };
            probe("before");
            if between && k % 2 == 1 {
                let r = with_watchdog(&m, || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    batch_identity::run(&m, &batch_identity::Cfg { lens: vec![11, 29, 53], batches: vec![3], steps: 1, vocab_range: (1000, 30000) })
                })));
                if let Err(e) = r {
                    let msg = e.downcast_ref::<String>().cloned().unwrap_or_default();
                    eprintln!("REPRO rep {rep} state {k}: the batched step between FAILED: {msg}");
                    dump_ledgers(&m, "batched step between");
                    eprintln!("REPRO failures 1 (batched step, rep {rep}, state {k})");
                    std::process::exit(3);
                }
            }
            let mut st = m.new_state().unwrap();
            // w21: a rep failed at the first solo step of a fresh model (state 0), before any batched step: watch it too
            let solo = |st: &mut _| {
                m.forward(st, &prompt[..prompt.len() - 1], HeadMode::None, None).unwrap();
                m.forward(st, &prompt[prompt.len() - 1..], HeadMode::LastLogits, None).unwrap()
            };
            // INTARSIA_REPRO_WD_SOLO=1 only (w22: n0 with it passed 20/20 where the same config failed in w17..w21): the probes and the watchdog
            // are the things that differ from a plain run, so each is switched separately
            let out = if std::env::var("INTARSIA_REPRO_WD_SOLO").is_ok_and(|v| v == "1") { with_watchdog(&m, || solo(&mut st)) } else { solo(&mut st) };
            let StepOut::Logits(l) = out else { panic!("no logits") };
            probe("after the solo prefill + decode step");
            let bits: Vec<u32> = l.iter().map(|x| x.to_bits()).collect();
            match &first {
                None => first = Some(bits),
                Some(f) if *f != bits => {
                    let n = f.iter().zip(&bits).filter(|(a, b)| a != b).count();
                    eprintln!("REPRO rep {rep} state {k}: SOLO first decode step differs from state 0 at {n} of {} entries (first {} vs {})", f.len(), f32::from_bits(f[0]), f32::from_bits(bits[0]));
                    eprintln!("REPRO failures 1 (solo, rep {rep}, state {k})");
                    std::process::exit(3);
                }
                Some(_) => {}
            }
        }
        eprintln!("REPRO rep {rep}: {states} fresh states, first decode step bit-identical");
    }
    eprintln!("REPRO failures 0/{reps}");
}

/// `INTARSIA_BATCH_TEST_TP=2` only (lane tp2, w16/w17): w16 ran the full gate with the batched graphs off (`INTARSIA_BATCH_GRAPHS=0`): the
/// 17,550 pairs of the sizes passed, then "decode rows beside prefill chunks" (4 decode rows, one prefill of 700 tokens in chunks of 256) failed at
/// its first batched step: `batch 4 step 0 row 0 (sequence 0, context 36 -> 37): logits differ at 248320 of 248320 entries, solo 2.16 batch 5.72`.
/// This runs only that part on one model load, `INTARSIA_REPRO_REPS` (3) times, and counts the mismatches (`REPRO failures f/reps`).
#[test]
#[ignore]
fn real_35b_tp2_batch_with_prefill_repro() {
    if tp() != 2 {
        eprintln!("SKIPPED: INTARSIA_BATCH_TEST_TP=2 only");
        return;
    }
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let reps = std::env::var("INTARSIA_REPRO_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let m = load(&dir, be, 16, 2048);
    let mut failures = 0;
    for rep in 0..reps {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| batch_identity::run_with_prefill(&m, &[37, 130, 300, 17], 700, 256, (1000, 30000))));
        match r {
            Ok(n) => eprintln!("REPRO rep {rep}: ok ({n} rows)"),
            Err(e) => {
                failures += 1;
                let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                eprintln!("REPRO rep {rep}: FAIL {msg}");
            }
        }
    }
    eprintln!("REPRO failures {failures}/{reps}");
    assert_eq!(failures, 0, "{failures} of {reps} reps mismatched");
}

/// `INTARSIA_BATCH_TEST_TP=2` only (lane tp2, w15/w17): repro of an intermittent mismatch of the FIRST batched step of a batch size
/// at tp=2 (27B, N = 4, step 0, every logit about 8 % off; the same test passed on the next run). Each rep loads the model afresh (so every
/// batch size records its graphs again), runs `INTARSIA_REPRO_STEPS` (3) batched steps per size in `INTARSIA_REPRO_SIZES` ("2-16") against
/// the solo steps, and counts a mismatch instead of stopping at it. `INTARSIA_REPRO_REPS` (6) reps. Prints `REPRO failures f/reps`.
#[test]
#[ignore]
fn real_35b_tp2_batch_first_steps_repro() {
    if tp() != 2 {
        eprintln!("SKIPPED: INTARSIA_BATCH_TEST_TP=2 only");
        return;
    }
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let num = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (reps, steps) = (num("INTARSIA_REPRO_REPS", 6), num("INTARSIA_REPRO_STEPS", 3));
    let sizes: Vec<usize> = {
        let spec = std::env::var("INTARSIA_REPRO_SIZES").unwrap_or_else(|_| "2-16".into());
        let (lo, hi) = spec.split_once('-').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap_or_else(|| (spec.parse().unwrap(), spec.parse().unwrap()));
        (lo..=hi).collect()
    };
    let lens = vec![5, 37, 130, 300, 17, 64, 9, 222, 400, 3, 77, 150, 28, 11, 90, 45];
    let mut failures = 0;
    for rep in 0..reps {
        if failures > 0 {
            break;
        }
        let m = load(&dir, be.clone(), 16, 2048);
        maybe_warm(&m);
        if std::env::var("INTARSIA_REPRO_NO_GRAPHS").is_ok_and(|v| v == "1") {
            m.set_graphs(false); // every step eager, solo and batched, on both ranks
        }
        let r = with_watchdog(&m, || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let n = batch_identity::run(&m, &batch_identity::Cfg { lens: lens.clone(), batches: sizes.clone(), steps, vocab_range: (1000, 30000) });
            assert_eq!(n, sizes.iter().sum::<usize>() * steps);
        })));
        match r {
            Ok(()) => eprintln!("REPRO rep {rep}: ok"),
            Err(e) => {
                failures += 1;
                let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                eprintln!("REPRO rep {rep}: FAIL {msg}");
                eprintln!("REPRO graphs at the failure: rank 0 {:?}, rank 1 {:?}", m.graph_stats(), m.rank_model(1).unwrap().graph_stats());
                dump_ledgers(&m, "mismatch");
                // the ranks are out of step after a bad step (a rank waits in a collective its peer never launches), and freeing
                // device memory would wait for it until the bounded wait traps: report and leave
                eprintln!("REPRO failures {failures}/{} (stopped at the first, after {} reps)", reps, rep + 1);
                std::process::exit(3);
            }
        }
        drop(m);
    }
    eprintln!("REPRO failures {failures}/{reps}");
}

/// `INTARSIA_BATCH_TEST_TP=2` only (lane tp2/batch): the gate above must catch a peer that breaks the batched step protocol on
/// the real model (`Model::set_tp_batch_mutant`): its rows in another order (its row table), row 0 one position behind (its
/// `batch_pos`), and its multi-row calls without the row-exact mark (the ordinary kernels, invisible on the cpu reference).
/// Not run here: a peer with another row count or a dropped collective (the P2P communicator cannot detect a size mismatch
/// and waits on the peer; the cpu tests `tests/tp_batch.rs` cover both).
#[test]
fn real_35b_tp2_batch_mutants_are_caught() {
    use intarsia_core::model::TpBatchMutant;
    if tp() != 2 {
        eprintln!("SKIPPED: INTARSIA_BATCH_TEST_TP=2 only");
        return;
    }
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 1024);
    let lens: Vec<usize> = vec![5, 37, 130, 300, 17, 64, 9, 222, 400, 3, 77, 150, 28, 11, 90, 45];
    let cfg = |b: usize| batch_identity::Cfg { lens: lens[..b].to_vec(), batches: vec![b], steps: 4, vocab_range: (1000, 30000) };
    // control: the unmutated gate passes at 8 rows
    assert_eq!(batch_identity::run(&m, &cfg(8)), 8 * 4);
    // the kernels mutant first, at a row count no recording exists for yet (a recorded graph would keep the exact kernels);
    // the other two act through the row table and positions the host rewrites before every step, recorded or not
    for (mutant, b) in [(TpBatchMutant::PlainKernels, 12usize), (TpBatchMutant::ReverseRows, 8)] {
        m.set_tp_batch_mutant(1, Some(mutant)).unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| batch_identity::run(&m, &cfg(b))));
        m.set_tp_batch_mutant(1, None).unwrap();
        assert!(r.is_err(), "mutant {mutant:?} at {b} rows was not caught by the batched-vs-solo gate");
        // the group is still healthy (no aborted communicator), so the next mutant's catch is its own
        assert_eq!(batch_identity::run(&m, &cfg(8)), 8 * 4, "the control after mutant {mutant:?} failed");
        eprintln!("mutant {mutant:?} at {b} rows: caught (control after it passes)");
    }
}

/// The stale-position mutant is its own test: the peer's row 0 runs one position behind and stays damaged (its sequence is one
/// token short), so the group is not usable after it. On HIP the next step then faults the device (HSA exception 0x1016, the
/// process aborts), which is the group going down, not a failure of the gate: the gate has caught it when the batched-vs-solo
/// comparison panics first. No control after it.
#[test]
fn real_35b_tp2_batch_stale_position_mutant_is_caught() {
    use intarsia_core::model::TpBatchMutant;
    if tp() != 2 {
        eprintln!("SKIPPED: INTARSIA_BATCH_TEST_TP=2 only");
        return;
    }
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 1024);
    let lens: Vec<usize> = vec![5, 37, 130, 300, 17, 64, 9, 222];
    let cfg = batch_identity::Cfg { lens, batches: vec![8], steps: 4, vocab_range: (1000, 30000) };
    assert_eq!(batch_identity::run(&m, &cfg), 8 * 4);
    m.set_tp_batch_mutant(1, Some(TpBatchMutant::StalePosition)).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| batch_identity::run(&m, &cfg)));
    assert!(r.is_err(), "mutant StalePosition at 8 rows was not caught by the batched-vs-solo gate");
    eprintln!("mutant StalePosition at 8 rows: caught");
    std::process::exit(0); // the group is damaged: leave without tearing it down
}

#[test]
fn real_qwen3_8b_batched_decode_is_bitwise_the_solo_step() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN3_DIR"), cuda()) else { return };
    let m = match try_load_with(&dir, be, 16, 2048, 0) {
        Ok((m, _)) => m,
        Err(e) => panic!("{e:?}"),
    };
    let lens = vec![5, 37, 130, 300, 17, 64, 9, 222, 400, 3, 77, 150, 28, 11, 90, 45];
    // on HIP (lane hb) every N from 2 to the cap, not only the sizes the CUDA gate samples
    let hip = std::env::var("INTARSIA_BATCH_TEST_BACKEND").is_ok_and(|v| v == "hip");
    let sizes: Vec<usize> = if hip { (2..=16).collect() } else { vec![2, 3, 4, 8, 16] };
    gate(&m, lens.clone(), sizes, 130, (1000, 30000));
    if !hip {
        // every other N up to the cap, shorter (lane bi8: the served 8B differed from solo from 5 streams up; that was prefix
        // reuse, not the batch width, and this gate shows the batch width is bitwise at every N)
        gate(&m, lens, (5..=16).filter(|n| *n != 8 && *n != 16).collect(), 32, (1000, 30000));
    }
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

/// BI8 probe (diagnostic, prints a table): is a dense prefill that resumes from a KV prefix bitwise the cold prefill? For each
/// equal.py prompt: the cold run (one `forward` over the prompt, 64 greedy tokens), every split `k` (`forward(P[..k])`, then
/// `forward(P[k..])`), the repeated-prompt reuse a slot gives (the cold run's state truncated to `plen - 1`, then the last
/// token alone: what `pick_slot` / `Seq::new` do for a prompt the slot already ran) and a cross-prompt reuse (another prompt's
/// state truncated to the shared prefix). Opt-in: `INTARSIA_BI8_PROBE=1` and `INTARSIA_QWEN3_DIR`.
#[test]
fn real_qwen3_8b_prefix_reuse_probe() {
    let _serial = serial();
    use intarsia_core::engine::TextCodec;
    if !std::env::var("INTARSIA_BI8_PROBE").is_ok_and(|v| v == "1") {
        eprintln!("SKIPPED: INTARSIA_BI8_PROBE=1 not set");
        return;
    }
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN3_DIR"), cuda()) else { return };
    let m = load(&dir, be, 2, 2048);
    let codec = intarsia_models::HfCodec::load(&dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    const GEN: usize = 64;
    const FORCED: usize = 4;
    let ids: Vec<Vec<u32>> = EQ_PROMPTS.iter().map(|p| codec.encode(p, false).unwrap()).collect();
    // lane rip: prompt passes run as the engines run them (`forward_prefill`: row-invariant prefill on CUDA), or with the ordinary
    // prefill under INTARSIA_RIP_PLAIN=1 (the BI8 probe as it was)
    let plain = std::env::var("INTARSIA_RIP_PLAIN").is_ok_and(|v| v == "1");
    let pf = |st: &mut intarsia_core::state::SequenceState, ids: &[u32], mode: HeadMode| {
        if plain { m.forward(st, ids, mode, None) } else { m.forward_prefill(st, ids, mode, None) }.unwrap()
    };
    // greedy continuation of `st` from the logits row `first`: (tokens, logits rows incl. `first`)
    let greedy = |st: &mut intarsia_core::state::SequenceState, first: Vec<f32>, n: usize| -> (Vec<u32>, Vec<Vec<f32>>) {
        let mut rows = vec![first];
        let mut toks = Vec::new();
        for _ in 0..n {
            let t = argmax_of(rows.last().unwrap());
            toks.push(t);
            rows.push(logits_of(m.forward(st, &[t], HeadMode::LastLogits, None).unwrap()));
        }
        (toks, rows)
    };
    let first_diff = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).position(|(x, y)| x != y);
    let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let mut cold: Vec<(Vec<u32>, Vec<Vec<f32>>)> = Vec::new();
    for (i, p) in ids.iter().enumerate() {
        let mut st = m.new_state().unwrap();
        let l = logits_of(pf(&mut st, p, HeadMode::LastLogits));
        let (toks, rows) = greedy(&mut st, l, GEN);
        eprintln!("COLD {i} plen {} ids {:?}", p.len(), toks);
        // the repeated prompt on the same slot: the cold state (prompt + GEN tokens) truncated to plen - 1, last token alone
        st.truncate(p.len() - 1, &*m.backend).unwrap();
        let l = logits_of(pf(&mut st, &p[p.len() - 1..], HeadMode::LastLogits));
        let (rt, rr) = greedy(&mut st, l, GEN);
        let d = rt.iter().zip(&toks).position(|(a, b)| a != b);
        eprintln!(
            "SELFREUSE {i} plen {}: logits row0 {} (maxabs {:.3e}), first logits diff row {:?}, first token diff {:?}",
            p.len(),
            if rr[0] == rows[0] { "EQUAL" } else { "DIFFER" },
            maxabs(&rr[0], &rows[0]),
            first_diff(&rr, &rows),
            d
        );
        cold.push((toks, rows));
    }
    // every split k: forward(P[..k]) then forward(P[k..]) and FORCED teacher-forced decode rows
    for (i, p) in ids.iter().enumerate() {
        let mut line = String::new();
        for k in 1..p.len() {
            let mut st = m.new_state().unwrap();
            pf(&mut st, &p[..k], HeadMode::None);
            let mut rows = vec![logits_of(pf(&mut st, &p[k..], HeadMode::LastLogits))];
            for &t in &cold[i].0[..FORCED] {
                rows.push(logits_of(m.forward(&mut st, &[t], HeadMode::LastLogits, None).unwrap()));
            }
            let fd = first_diff(&rows, &cold[i].1[..=FORCED]);
            line.push_str(&format!(" k{k}(tail {}):{}", p.len() - k, fd.map_or("=".to_string(), |r| format!("x{r}"))));
        }
        eprintln!("SPLIT {i} plen {}:{line}", p.len());
    }
    // cross-prompt reuse: prompt j's cold state truncated to the prefix it shares with i, then P_i[c..]
    for (i, p) in ids.iter().enumerate() {
        for (j, q) in ids.iter().enumerate() {
            let c = p.iter().zip(q).take_while(|(a, b)| a == b).count().min(p.len() - 1);
            if i == j || c == 0 {
                continue;
            }
            let mut st = m.new_state().unwrap();
            let l = logits_of(pf(&mut st, q, HeadMode::LastLogits));
            let _ = greedy(&mut st, l, FORCED);
            st.truncate(c, &*m.backend).unwrap();
            let mut rows = vec![logits_of(pf(&mut st, &p[c..], HeadMode::LastLogits))];
            for &t in &cold[i].0[..FORCED] {
                rows.push(logits_of(m.forward(&mut st, &[t], HeadMode::LastLogits, None).unwrap()));
            }
            eprintln!("CROSS {i}<-{j} shared {c} of {}: first logits diff row {:?}", p.len(), first_diff(&rows, &cold[i].1[..=FORCED]));
        }
    }
}

/// The KV cache rows `0..len` of every attention slot of `st`, as bytes (K then V, kv head by kv head).
fn kv_rows(m: &Model, st: &intarsia_core::state::SequenceState, len: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for s in &st.slots {
        if let intarsia_core::state::StateSlot::AttnKv(a) = s {
            let row = a.head_dim * a.dtype.size_bytes();
            for buf in [&a.k, &a.v] {
                for h in 0..a.n_kv_heads {
                    let mut b = vec![0u8; len * row];
                    m.backend.download(buf, (h * a.cap) * row, &mut b, 0).unwrap();
                    out.extend_from_slice(&b);
                }
            }
        }
    }
    out
}

/// Deterministic text of about `words` words, every sentence different (a long prompt for the row-invariance gate).
fn long_text(words: usize) -> String {
    let mut s = String::new();
    let mut i = 0;
    while s.split_whitespace().count() < words {
        s.push_str(&format!("{} (part {i}, {} items). ", EQ_PROMPTS[i % EQ_PROMPTS.len()].trim(), 3 * i + 7));
        i += 1;
    }
    s
}

/// Lane rip (docs/results/2026-10-29-rip-row-invariant-prefill.md): on Qwen3-8B, a prompt prefilled in one `forward_prefill`
/// (the cold run: 4096-row chunks) has bitwise the same last-row logits and KV rows as the same prompt prefilled in two passes
/// split at every sampled k, as in 256-row busy chunks, and as a resume from a held prefix (the cold state truncated to k, then
/// the rest: what prefix reuse does), and the greedy decode after each is bitwise the cold one. Prompts: a 1500-token one (one
/// pass cold) and a 4700-token one (4096 + 604 cold). `INTARSIA_RIP_PLAIN=1`: the same with the ordinary prefill (`forward`, no
/// row-invariant kernels), printing the differences instead of asserting (the lane's "before" probe, and the mutant).
#[test]
fn real_qwen3_8b_prefill_is_row_invariant() {
    let _serial = serial();
    use intarsia_core::engine::TextCodec;
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN3_DIR"), cuda()) else { return };
    let plain = std::env::var("INTARSIA_RIP_PLAIN").is_ok_and(|v| v == "1");
    let m = load(&dir, be, 2, 8192);
    assert!(plain || m.row_invariant_prefill(), "row-invariant prefill is off for the 8B");
    let codec = intarsia_models::HfCodec::load(&dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    let pf = |st: &mut intarsia_core::state::SequenceState, ids: &[u32], mode: HeadMode| {
        if plain { m.forward(st, ids, mode, None) } else { m.forward_prefill(st, ids, mode, None) }.unwrap()
    };
    const GEN: usize = 8;
    let decode = |st: &mut intarsia_core::state::SequenceState, first: Vec<f32>| -> Vec<Vec<f32>> {
        let mut rows = vec![first];
        for _ in 0..GEN {
            let t = argmax_of(rows.last().unwrap());
            rows.push(logits_of(m.forward(st, &[t], HeadMode::LastLogits, None).unwrap()));
        }
        rows
    };
    let mut failures = Vec::new();
    for words in [1100usize, 3500] {
        let p = codec.encode(&long_text(words), false).unwrap();
        let plen = p.len();
        let mut cold = m.new_state().unwrap();
        let first = logits_of(pf(&mut cold, &p, HeadMode::LastLogits));
        let kv_cold = kv_rows(&m, &cold, plen);
        let cold_rows = decode(&mut cold, first);
        let kv_after = kv_rows(&m, &cold, plen + GEN);
        eprintln!("RIP prompt {plen} tokens (cold chunk {}), cold greedy {:?}", m.prefill_chunk(), cold_rows.iter().map(|r| argmax_of(r)).collect::<Vec<_>>());
        let mut judge = |what: String, st: &mut intarsia_core::state::SequenceState, first: Vec<f32>| {
            let same_logits = first == cold_rows[0];
            let same_kv = kv_rows(&m, st, plen) == kv_cold;
            let rows = decode(st, first);
            let dec = rows.iter().zip(&cold_rows).position(|(a, b)| a != b);
            let same_after = kv_rows(&m, st, plen + GEN) == kv_after;
            let ok = same_logits && same_kv && dec.is_none() && same_after;
            eprintln!(
                "RIP {plen} {what}: logits {} kv {} decode {} kv+decode {}",
                if same_logits { "EQUAL" } else { "DIFFER" },
                if same_kv { "EQUAL" } else { "DIFFER" },
                dec.map_or("EQUAL".to_string(), |r| format!("DIFFER from row {r}")),
                if same_after { "EQUAL" } else { "DIFFER" },
            );
            if !ok {
                failures.push(format!("{plen} {what}"));
            }
        };
        let mut ks = vec![1usize, 7, 255, 256, 1000, plen - 1];
        if plen > 4097 {
            ks.extend([4095, 4096, 4097]);
        }
        for &k in &ks {
            let mut st = m.new_state().unwrap();
            pf(&mut st, &p[..k], HeadMode::None);
            let l = logits_of(pf(&mut st, &p[k..], HeadMode::LastLogits));
            judge(format!("split k={k}"), &mut st, l);
        }
        // 256-row busy chunks (the batched engine's chunk while others decode), then the last row's head
        {
            let mut st = m.new_state().unwrap();
            let mut at = 0;
            let mut out = None;
            while at < plen {
                let n = 256.min(plen - at);
                let mode = if at + n == plen { HeadMode::LastLogits } else { HeadMode::None };
                out = Some(pf(&mut st, &p[at..at + n], mode));
                at += n;
            }
            judge("busy 256-row chunks".into(), &mut st, logits_of(out.unwrap()));
        }
        // prefix reuse: a held state (the prompt and a few decode steps) truncated to k, then the rest of the prompt
        for &k in &[plen - 1, 1000, 1] {
            let mut st = m.new_state().unwrap();
            let l = logits_of(pf(&mut st, &p, HeadMode::LastLogits));
            let _ = decode(&mut st, l);
            st.truncate(k, &*m.backend).unwrap();
            let l = logits_of(pf(&mut st, &p[k..], HeadMode::LastLogits));
            judge(format!("reuse from {k}"), &mut st, l);
        }
    }
    if plain {
        eprintln!("RIP plain prefill: {} of the cases differ: {failures:?}", failures.len());
    } else {
        assert!(failures.is_empty(), "row-invariant prefill is not row-invariant: {failures:?}");
    }
}

// ---- engine level ---------------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    text: String,
    reason: Option<FinishReason>,
}

fn collect(rx: &Receiver<GenEvent>) -> Out {
    let (mut ids, mut text) = (vec![], String::new());
    loop {
        match rx.recv_timeout(Duration::from_secs(600)).expect("engine stalled") {
            GenEvent::Token { id, text: t } => {
                ids.push(id);
                text.push_str(&t);
            }
            GenEvent::Done { finish_reason, .. } => return Out { ids, text, reason: Some(finish_reason) },
            GenEvent::Error { message } => panic!("request failed: {message}"),
        }
    }
}

const PROMPTS: [&str; 6] = [
    "The history of the printing press begins",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which",
    "In 1969, the Apollo 11 mission",
    "Explain, step by step, why the sky is blue. First,",
    "The three laws of thermodynamics state that",
];

fn requests(engine: &dyn Engine, n: usize, max_tokens: usize) -> Vec<GenRequest> {
    (0..n)
        .map(|i| {
            let mut ids = engine.tokenize(PROMPTS[i % PROMPTS.len()], false).unwrap();
            if i >= PROMPTS.len() {
                // distinct prompts beyond the six texts: a different tail
                ids.extend(engine.tokenize(&format!(" ({i})"), false).unwrap());
            }
            let mut r = GenRequest { prompt: ids, max_tokens, ignore_eos: true, ..Default::default() };
            if i % 4 == 3 {
                r.temperature = 0.8;
                r.top_k = 40;
                r.top_p = 0.95;
                r.seed = Some(11 + i as u64);
            }
            r
        })
        .collect()
}

#[test]
fn real_35b_batched_engine_equals_the_local_engine() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let mk = |max_batch: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 2048, kv_fraction: 0.95, profile_every: 0, ..Default::default() },
        max_batch,
        ..Default::default()
    };
    // the reference: every request alone on the single-sequence engine
    let reference: Vec<Out> = {
        let l = engine(&dir, &be, &mk(None));
        assert_eq!(l.max_batch, 1);
        let reqs = requests(&*l.engine, 20, 130);
        reqs.into_iter().map(|r| collect(&l.engine.submit(r).unwrap())).collect()
    };
    assert!(reference.iter().all(|o| o.ids.len() == 130));
    for slots in [4usize, 16] {
        let l = engine(&dir, &be, &mk(Some(slots)));
        assert_eq!(l.max_batch, slots);
        // one stream at a time: the batched engine at N = 1 is the local engine
        for (i, r) in requests(&*l.engine, 3, 130).into_iter().enumerate() {
            assert_eq!(collect(&l.engine.submit(r).unwrap()), reference[i], "one stream, request {i}, {slots} slots");
        }
        // all at once: 20 requests on `slots` slots (queueing, overlap, slot reuse)
        let rxs: Vec<_> = requests(&*l.engine, 20, 130).into_iter().map(|r| l.engine.submit(r).unwrap()).collect();
        for (i, rx) in rxs.iter().enumerate() {
            let got = collect(rx);
            let want = &reference[i];
            // prefill of a request that starts while others decode runs in busy chunks; a prompt this short is one chunk
            // either way, so the whole output is compared
            assert_eq!(&got, want, "concurrent request {i} on {slots} slots");
        }
    }
}

/// Lane bi8 (docs/results/2026-10-29-bi8-serve-batch-invariance.md): on Qwen3-8B every request computes the bits of its cold
/// run whatever the engine held before. The reference is each equal.py prompt alone on a fresh single-sequence engine. The same
/// engine then runs the prompts again in an order full of repeats and shared first tokens (a slot's prefix reuse resumed a
/// repeated prompt with its last token alone, a 1-row pass, which flipped greedy tokens); then the batched engine at 8 and 16
/// slots runs three rounds of all 16 at once without a restart (rounds 2 and 3 find every prompt's own earlier slot idle) and a
/// round of 5. Every request's tokens must be the reference's. Greedy, and seeded sampling on every fourth prompt.
#[test]
fn real_qwen3_8b_batched_engine_equals_the_cold_reference() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN3_DIR"), cuda()) else { return };
    const GEN: usize = 64;
    let mk = |max_batch: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 2048, kv_fraction: 0.95, profile_every: 0, ..Default::default() },
        max_batch,
        ..Default::default()
    };
    let req = |e: &dyn Engine, i: usize| {
        let mut r = GenRequest { prompt: e.tokenize(EQ_PROMPTS[i], false).unwrap(), max_tokens: GEN, ignore_eos: true, ..Default::default() };
        if i % 4 == 3 {
            r.temperature = 0.8;
            r.top_k = 40;
            r.top_p = 0.95;
            r.seed = Some(100 + i as u64);
        }
        r
    };
    // the cold reference: the prompts in order on one single-sequence engine. It holds only the previous request, and no two
    // consecutive prompts share their first token, so no request finds a prefix to reuse, under any reuse rule
    let reference: Vec<Out> = {
        let l = engine(&dir, &be, &mk(None));
        let reqs: Vec<GenRequest> = (0..EQ_PROMPTS.len()).map(|i| req(&*l.engine, i)).collect();
        for w in reqs.windows(2) {
            assert_ne!(w[0].prompt[0], w[1].prompt[0], "consecutive reference prompts must not share a prefix");
        }
        reqs.into_iter().map(|r| collect(&l.engine.submit(r).unwrap())).collect()
    };
    assert!(reference.iter().all(|o| o.ids.len() == GEN));
    {
        let l = engine(&dir, &be, &mk(None));
        for (n, &i) in [4usize, 4, 5, 0, 5, 8, 8, 7, 12, 5, 14, 1, 1].iter().enumerate() {
            assert_eq!(collect(&l.engine.submit(req(&*l.engine, i)).unwrap()), reference[i], "single-sequence engine, request {n} (prompt {i})");
        }
    }
    for slots in [8usize, 16] {
        let l = engine(&dir, &be, &mk(Some(slots)));
        assert_eq!(l.max_batch, slots);
        let mut rounds: Vec<Vec<usize>> = vec![(0..16).collect(), (0..16).collect(), (0..16).rev().collect(), (0..5).collect()];
        rounds.push(vec![4, 5, 8, 4, 5, 8]);
        for (round, idx) in rounds.iter().enumerate() {
            let rxs: Vec<_> = idx.iter().map(|&i| l.engine.submit(req(&*l.engine, i)).unwrap()).collect();
            for (&i, rx) in idx.iter().zip(&rxs) {
                assert_eq!(&collect(rx), &reference[i], "{slots} slots, round {round}, prompt {i}");
            }
        }
        eprintln!("bi8 engine identity: {slots} slots, {} requests equal the cold reference", rounds.iter().map(|r| r.len()).sum::<usize>());
    }
}

/// Lane rip: why `Model::reuse_len` stops at `SequenceState::inv_len` (the rows prefill passes wrote) rather than at the common
/// prefix. A held sequence (a ~600-token prompt, then 16 greedy decode steps) holds the reply's rows as decode steps wrote them:
/// the M = 1 GEMV and the split-KV decode attention. A cold row-invariant prefill of the same tokens must give the prompt rows
/// bitwise, and other bits for the decode-written rows (else reusing them would be exact and the limit needless: this test then
/// fails to say so). Resuming at `inv_len` (what reuse does) gives the cold bits: logits and every KV row.
#[test]
fn real_qwen3_8b_decode_written_rows_are_not_prefill_rows() {
    let _serial = serial();
    use intarsia_core::engine::TextCodec;
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN3_DIR"), cuda()) else { return };
    let m = load(&dir, be, 2, 8192);
    assert!(m.row_invariant_prefill(), "row-invariant prefill is off for the 8B");
    let codec = intarsia_models::HfCodec::load(&dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    let p = codec.encode(&long_text(300), false).unwrap();
    const GEN: usize = 16;
    let mut held = m.new_state().unwrap();
    let mut l = logits_of(m.forward_prefill(&mut held, &p, HeadMode::LastLogits, None).unwrap());
    let mut q = p.clone();
    for _ in 0..GEN {
        let t = argmax_of(&l);
        q.push(t);
        l = logits_of(m.forward(&mut held, &[t], HeadMode::LastLogits, None).unwrap());
    }
    assert_eq!(held.len, q.len());
    assert_eq!(held.inv_len, p.len(), "decode steps advanced the row-invariant prefix");
    // the next turn's prompt holds q whole: reuse resumes at the prompt, not at q
    assert_eq!(m.reuse_len(&held, q.len() + 8, q.len()), p.len());
    let mut cold = m.new_state().unwrap();
    let lc = logits_of(m.forward_prefill(&mut cold, &q, HeadMode::LastLogits, None).unwrap());
    let prompt_rows = kv_rows(&m, &held, p.len()) == kv_rows(&m, &cold, p.len());
    let all_rows = kv_rows(&m, &held, q.len()) == kv_rows(&m, &cold, q.len());
    let next = l == lc;
    let eq = |b: bool| if b { "EQUAL" } else { "DIFFER" };
    eprintln!(
        "RIP decode-written rows: {} prompt rows + {GEN} decode rows; vs a cold prefill of the same tokens: KV of the prompt rows {}, KV of all rows {}, next-token logits {}",
        p.len(),
        eq(prompt_rows),
        eq(all_rows),
        eq(next)
    );
    assert!(prompt_rows, "the held prompt rows are not the cold prefill rows");
    assert!(!(all_rows && next), "decode-written rows are bitwise prefill rows: the inv_len limit is stricter than needed");
    // what reuse does: back to inv_len, then the rest as a prefill pass
    held.truncate(held.inv_len, &*m.backend).unwrap();
    let lr = logits_of(m.forward_prefill(&mut held, &q[p.len()..], HeadMode::LastLogits, None).unwrap());
    let exact = lr == lc && kv_rows(&m, &held, q.len()) == kv_rows(&m, &cold, q.len());
    eprintln!("RIP resume at inv_len {} then prefill the {GEN} reply tokens: logits and KV {}", p.len(), eq(exact));
    assert!(exact, "resuming at inv_len is not the cold prefill");
}

/// (ids, cached prompt tokens) of a request.
fn collect_cached(rx: &Receiver<GenEvent>) -> (Vec<u32>, usize) {
    let mut ids = vec![];
    loop {
        match rx.recv_timeout(Duration::from_secs(600)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { cached_tokens, .. } => return (ids, cached_tokens),
            GenEvent::Error { message } => panic!("request failed: {message}"),
        }
    }
}

/// Lane rip (docs/results/2026-10-29-rip-row-invariant-prefill.md), engine level on Qwen3-8B: with row-invariant prefill a
/// request reuses any held prefix prefill passes wrote and still computes its cold output.
/// - Multi-turn chat on the single-sequence engine: turn 2 (turn 1's prompt, its 48-token reply, a new user message) resumes from
///   turn 1's whole prompt (not from the reply's rows, which decode steps wrote) and equals turn 2 on a fresh engine; turn 3 resumes
///   from turn 2's whole prompt.
/// - The batched engine (16 slots): the same turns, and a 1700-token prompt admitted while four others decode (256-row busy
///   chunks), equal the fresh engine's outputs.
/// Why turn 2 does not resume inside turn 1's reply: `real_qwen3_8b_decode_written_rows_are_not_prefill_rows` (bitwise; greedy tokens
/// alone did not tell: with `INTARSIA_PREFIX_REUSE=any` the 48-token turns still matched here).
#[test]
fn real_qwen3_8b_reuse_and_busy_chunks_equal_the_cold_reference() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN3_DIR"), cuda()) else { return };
    let mk = |max_batch: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 8192, kv_fraction: 0.95, profile_every: 0, ..Default::default() },
        max_batch,
        ..Default::default()
    };
    let chat = |e: &dyn Engine, history: &[u32], user: &str| -> Vec<u32> {
        let mut p = history.to_vec();
        let pre = if history.is_empty() { "" } else { "<|im_end|>\n" };
        p.extend(e.tokenize(&format!("{pre}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"), false).unwrap());
        p
    };
    let greedy = |prompt: Vec<u32>, n: usize| GenRequest { prompt, max_tokens: n, ignore_eos: true, ..Default::default() };
    let u1 = format!("Summarise the following notes in three sentences.\n\n{}", long_text(1400));
    let u2 = "Now list the three most important items, one per line.";
    let u3 = "Which of them came first?";
    // turns on one engine: (prompts, outputs, cached)
    let turns = |e: &dyn Engine| -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>) {
        let (mut ps, mut outs, mut cached) = (vec![], vec![], vec![]);
        let mut hist: Vec<u32> = vec![];
        for u in [u1.as_str(), u2, u3] {
            let p = chat(e, &hist, u);
            let (ids, c) = collect_cached(&e.submit(greedy(p.clone(), 48)).unwrap());
            hist = p.clone();
            hist.extend(&ids);
            ps.push(p);
            outs.push(ids);
            cached.push(c);
        }
        (ps, outs, cached)
    };
    let long = |e: &dyn Engine| e.tokenize(&format!("Translate to French:\n{}", long_text(1200)), false).unwrap();
    // warm: the single-sequence engine runs the three turns in a row (each resumes from the one before)
    let (ps, warm, cached) = {
        let l = engine(&dir, &be, &mk(None));
        turns(&*l.engine)
    };
    eprintln!("RIP turns: prompts {:?} tokens, cached {:?}", ps.iter().map(|p| p.len()).collect::<Vec<_>>(), cached);
    assert_eq!(cached[0], 0);
    assert_eq!(cached[1], ps[0].len(), "turn 2 resumes from turn 1's whole prompt, not from its reply's rows");
    assert_eq!(cached[2], ps[1].len(), "turn 3 resumes from turn 2's whole prompt");
    // cold: each turn's prompt alone on a fresh engine (the previous request shares no prefix with it), and the long prompt
    let (cold, cold_long) = {
        let l = engine(&dir, &be, &mk(None));
        let mut outs = vec![];
        let filler = l.engine.tokenize("Zebra crossings are", false).unwrap();
        for p in &ps {
            let (ids, c) = collect_cached(&l.engine.submit(greedy(p.clone(), 48)).unwrap());
            assert_eq!(c, 0, "the cold reference reused a prefix");
            outs.push(ids);
            let _ = collect_cached(&l.engine.submit(greedy(filler.clone(), 2)).unwrap());
        }
        let (ids, c) = collect_cached(&l.engine.submit(greedy(long(&*l.engine), 48)).unwrap());
        assert_eq!(c, 0);
        (outs, ids)
    };
    for t in 0..3 {
        let same = warm[t] == cold[t];
        eprintln!("RIP single-sequence turn {}: warm {} cold", t + 1, if same { "==" } else { "!=" });
        assert!(same, "turn {} on the single-sequence engine differs from its cold run", t + 1);
    }
    // the batched engine: the turns, then the long prompt while four others decode
    let l = engine(&dir, &be, &mk(Some(16)));
    assert_eq!(l.max_batch, 16);
    let (ps_b, warm_b, cached_b) = turns(&*l.engine);
    assert_eq!(ps_b, ps);
    eprintln!("RIP batched turns cached {cached_b:?}");
    for t in 0..3 {
        assert_eq!(warm_b[t], cold[t], "turn {} on the batched engine differs from its cold run", t + 1);
    }
    assert!(cached_b[1] >= ps[0].len() && cached_b[2] >= ps[1].len(), "the batched engine did not resume the turns: {cached_b:?}");
    let bg: Vec<_> = (0..4).map(|i| l.engine.submit(greedy(l.engine.tokenize(EQ_PROMPTS[i], false).unwrap(), 400)).unwrap()).collect();
    std::thread::sleep(Duration::from_millis(1500));
    let (ids, _) = collect_cached(&l.engine.submit(greedy(long(&*l.engine), 48)).unwrap());
    for rx in &bg {
        let _ = collect_cached(rx);
    }
    assert_eq!(ids, cold_long, "a long prompt prefilled in busy chunks differs from its cold run");
    eprintln!("RIP batched: turns and the busy-chunked long prompt equal the cold reference");
}

/// Device-side sampling is the host sampling (lane htk, docs/results/2026-10-27-htk-hip-topk.md): `INTARSIA_DEVICE_SAMPLING=1` on a backend with
/// `topk_rows`. The same seeded `Sampler`s draw from the whole logits row (`HeadMode::LastLogits`, the host walk) in one arm and from the device's
/// top-k candidates (`HeadMode::LastTopK`) in the other, over 32 steps of batched decode with 1, 4 and 16 rows and five temperature / top_k /
/// top_p settings (k from 1 to the 64 maximum); the two arms must draw the same token at every step of every row. Opt-in like the other 35B gates.
#[test]
fn real_35b_device_sampling_draws_the_host_tokens() {
    let _serial = serial();
    use intarsia_core::sampler::{Sampler, SamplingParams};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    if !m.device_sampling() {
        skip("INTARSIA_DEVICE_SAMPLING=1 is not set (or the backend has no device top-k)");
        return;
    }
    let settings = [(0.7f32, 20usize, 1.0f32), (1.0, 40, 0.9), (0.8, 64, 0.5), (1.3, 8, 0.95), (0.6, 1, 1.0)];
    let steps = 32;
    let mut cells = 0usize;
    for b in [1usize, 4, 16] {
        for &(temperature, top_k, top_p) in &settings {
            let params = SamplingParams { temperature, top_k, top_p };
            let run = |device: bool| -> Vec<Vec<u32>> {
                let mut states: Vec<_> = (0..b)
                    .map(|i| {
                        let mut st = m.new_state().unwrap();
                        let p: Vec<u32> = (0..48 + 5 * i).map(|j| 1000 + ((j * 7 + i * 13) % 20000) as u32).collect();
                        m.forward(&mut st, &p, HeadMode::None, None).unwrap();
                        st
                    })
                    .collect();
                let mut samplers: Vec<Sampler> = (0..b).map(|i| Sampler::new(params.clone(), Some(7 + i as u64))).collect();
                let mut toks: Vec<u32> = (0..b).map(|i| 1000 + 37 * i as u32).collect();
                let modes = vec![if device { HeadMode::LastTopK(top_k) } else { HeadMode::LastLogits }; b];
                let mut drawn = vec![Vec::new(); b];
                for _ in 0..steps {
                    let mut refs: Vec<&mut _> = states.iter_mut().collect();
                    let out = m.forward_batch(&mut refs, &toks, &modes).unwrap();
                    for (r, o) in out.iter().enumerate() {
                        toks[r] = match o {
                            StepOut::Logits(l) => samplers[r].sample(l) as u32,
                            StepOut::TopK { ids, vals } => {
                                assert!(device, "a top-k row came back in the host arm");
                                samplers[r].sample_topk(ids, vals) as u32
                            }
                            o => panic!("unexpected {o:?}"),
                        };
                        drawn[r].push(toks[r]);
                    }
                }
                drawn
            };
            let host = run(false);
            let dev = run(true);
            assert_eq!(dev, host, "b {b} temperature {temperature} top_k {top_k} top_p {top_p}");
            cells += b * steps;
        }
    }
    eprintln!("{cells} (row, step) tokens drawn from the device top-k equal the host draws (B 1 / 4 / 16, {} settings)", settings.len());
}

// ---- timing (not a gate): `cargo test ... -- --ignored --nocapture real_35b_batch_step_timing` --------------------------

/// Wall time of `forward_batch` (all rows `LastArgmax`, the serving mode) at B = 1, 2, 4, 8, 16 over 24 steps, the solo
/// `forward` step beside it, and the per-op profile of one profiled B = 4 and B = 16 step.
#[test]
#[ignore]
fn real_35b_batch_step_timing() {
    let _serial = serial();
    use intarsia_core::model::{HeadMode, StepOut};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    let tok = |i: usize| 1000 + 37 * i as u32;
    let mut states: Vec<_> = (0..16)
        .map(|i| {
            let mut st = m.new_state().unwrap();
            let p: Vec<u32> = (0..900 + 5 * i).map(|j| 1000 + ((j * 7 + i * 13) % 20000) as u32).collect();
            m.forward(&mut st, &p, HeadMode::None, None).unwrap();
            st
        })
        .collect();
    let steps = 24;
    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        let StepOut::Token(_) = m.forward(&mut states[0], &[tok(0)], HeadMode::LastArgmax, None).unwrap() else {
            panic!()
        };
    }
    let solo = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
    eprintln!("TIMING solo step {solo:.2} ms ({:.1} tok/s)", 1e3 / solo);
    // the row counts timed: 2, 4, 8, 16, or INTARSIA_TIMING_ROWS="2,3,4,..." (test-only)
    let rows: Vec<usize> = std::env::var("INTARSIA_TIMING_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 4, 8, 16]);
    for b in rows {
        let ids: Vec<u32> = (0..b).map(tok).collect();
        let modes = vec![HeadMode::LastArgmax; b];
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
            m.forward_batch(&mut refs, &ids, &modes).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        eprintln!(
            "TIMING batch B={b:>2}: {ms:.2} ms/step, {:.1} tok/s aggregate ({:.2}x solo step)",
            b as f64 * 1e3 / ms,
            ms / solo
        );
    }
}

/// B2 (Fable review): the cost of a batched step whose rows are SAMPLED, on the host path (`LastLogits`: one [B, V] download and a
/// host top-k/top-p walk per row) against the device path (`LastTopK(40)`: the k candidates only; opt-in: `INTARSIA_DEVICE_SAMPLING=1` turns it on).
/// Same seeds, so the two draw the same tokens when the states are the same; the token parity is gated by the engine test
/// (`real_35b_batched_engine_equals_the_local_engine`, seeded-sampled requests, run with `INTARSIA_DEVICE_SAMPLING=1`).
#[test]
#[ignore]
fn real_35b_batch_step_sampled_timing() {
    let _serial = serial();
    use intarsia_core::model::{HeadMode, StepOut};
    use intarsia_core::sampler::{Sampler, SamplingParams};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
        let m = load(&dir, be, 16, 2048);
    let params = SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.9 };
    let mut states: Vec<_> = (0..16)
        .map(|i| {
            let mut st = m.new_state().unwrap();
            let p: Vec<u32> = (0..900 + 5 * i).map(|j| 1000 + ((j * 7 + i * 13) % 20000) as u32).collect();
            m.forward(&mut st, &p, HeadMode::None, None).unwrap();
            st
        })
        .collect();
    let steps = 24;
    for b in [4usize, 16] {
        for device in [false, true] {
            let mut samplers: Vec<Sampler> = (0..b).map(|i| Sampler::new(params.clone(), Some(7 + i as u64))).collect();
            let mut toks: Vec<u32> = (0..b).map(|i| 1000 + 37 * i as u32).collect();
            let modes = vec![if device { HeadMode::LastTopK(40) } else { HeadMode::LastLogits }; b];
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
                let out = m.forward_batch(&mut refs, &toks, &modes).unwrap();
                for (r, o) in out.iter().enumerate() {
                    toks[r] = match o {
                        StepOut::Logits(l) => samplers[r].sample(l) as u32,
                        StepOut::TopK { ids, vals } => samplers[r].sample_topk(ids, vals) as u32,
                        o => panic!("unexpected {o:?}"),
                    };
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
            eprintln!("TIMING sampled B={b:>2} {}: {ms:.2} ms/step incl. host sampling, {:.1} tok/s aggregate", if device { "device top-k" } else { "host      " }, b as f64 * 1e3 / ms);
        }
    }
}

/// Per-op calls and time of 8 profiled batched steps at B = 1, 4 and 16, or at `INTARSIA_TIMING_ROWS="1,8,16"` (profiling
/// syncs after each op, so the sum is above the unprofiled step; the shares are what to read). Two unprofiled-in-the-table
/// warm-up steps per B come first.
#[test]
#[ignore]
fn real_35b_batch_step_profile() {
    let _serial = serial();
    use intarsia_core::model::HeadMode;
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let (m, tel) = load_with(&dir, be, 16, 2048, 1);
    let mut states: Vec<_> = (0..16)
        .map(|i| {
            let mut st = m.new_state().unwrap();
            let p: Vec<u32> = (0..900 + 5 * i).map(|j| 1000 + ((j * 7 + i * 13) % 20000) as u32).collect();
            m.forward(&mut st, &p, HeadMode::None, None).unwrap();
            st
        })
        .collect();
    let rows_list: Vec<usize> = std::env::var("INTARSIA_TIMING_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 4, 16]);
    for b in rows_list {
        let ids: Vec<u32> = (0..b).map(|i| 1000 + 37 * i as u32).collect();
        let modes = vec![HeadMode::LastArgmax; b];
        // B = 1 is the solo step (`forward`, eager, the same per-op syncs): the batched path replays a captured graph at one row,
        // whose ops are not timed one by one
        let step = |states: &mut Vec<_>| {
            if b == 1 {
                m.forward(&mut states[0], &ids, HeadMode::LastArgmax, None).unwrap();
            } else {
                let mut refs: Vec<&mut _> = states.iter_mut().take(b).collect();
                m.forward_batch(&mut refs, &ids, &modes).unwrap();
            }
        };
        for _ in 0..2 {
            step(&mut states);
        }
        let before = tel.snapshot().ops;
        let t0 = std::time::Instant::now();
        for _ in 0..8 {
            step(&mut states);
        }
        let wall = t0.elapsed().as_secs_f64() * 1e3 / 8.0;
        let after = tel.snapshot().ops;
        let mut rows: Vec<(String, u64, f64)> = after
            .iter()
            .map(|(k, a)| {
                let b0 = before.get(k);
                (k.clone(), a.calls - b0.map_or(0, |x| x.calls), a.time_ms - b0.map_or(0.0, |x| x.time_ms))
            })
            .filter(|r| r.1 > 0)
            .collect();
        rows.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
        let tot: f64 = rows.iter().map(|r| r.2).sum::<f64>() / 8.0;
        eprintln!("PROFILE B={b}: wall {wall:.2} ms/step (profiled), op time sum {tot:.2} ms/step");
        for (k, calls, ms) in rows.iter().take(40) {
            eprintln!("PROFILE B={b}   {k:<40} {:>6.1} calls/step {:>7.3} ms/step", *calls as f64 / 8.0, ms / 8.0);
        }
    }
}

/// The repository's `docs/*.md` as token ids of the model's tokenizer (about 106k tokens): realistic English / markdown text for
/// prompts of any length up to 100k.
fn corpus_tokens(dir: &std::path::Path) -> Vec<u32> {
    use intarsia_core::engine::TextCodec;
    let codec = intarsia_models::HfCodec::load(dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut names: Vec<_> = std::fs::read_dir(&docs)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    names.sort();
    let text: String = names.iter().map(|p| std::fs::read_to_string(p).unwrap()).collect::<Vec<_>>().join("\n\n");
    codec.encode(&text, false).unwrap()
}

// ---- busy-chunk prefill KL gate (not bitwise by design) -------------------------------------------------------------------

/// Gate of the one non-bitwise step of the batching engine: a request that starts prefilling while others decode runs in busy
/// chunks (1024 tokens for the 35B on CUDA) instead of the solo chunk (2048), and the GEMM tiling depends on the chunk. For
/// prompts longer than the busy chunk (1.5k, 2.5k, 4k, 8k, 16k tokens of repository docs text) this prefills the same prompt
/// both ways, then decodes 64 greedy tokens from the solo state and teacher-forces the same tokens through the busy state.
/// Compared per row over the full vocabulary: KL(solo || busy) and top-1 agreement. Floor: the 35B short golden's ref-noise
/// (`golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors.json`, kl21_mean 3.161e-3: vLLM's own run-to-run spread on
/// the same prompts). Pass: the mean KL over the 64 decode rows of every prompt is at most that floor (factor 1, the
/// default of the ref-noise rule). Controls: the solo chunk driven by hand equals `forward` of the whole prefix (harness
/// check, KL exactly 0), and a chunk of 256 (the old default) is reported beside 1024.
#[test]
#[ignore]
fn real_35b_busy_chunk_prefill_kl_gate() {
    let _serial = serial();
    use intarsia_core::engine::TextCodec;
    use intarsia_core::model::{HeadMode, StepOut};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    const FLOOR: f64 = 3.161e-3; // ref-noise kl21_mean of the 35B short golden
    const SOLO: usize = 2048;
    const DECODE_ROWS: usize = 64;
    let lens = [1536usize, 2560, 4096, 8192, 16384];
    let (m, _) = load_with(&dir, be, 2, 16384 + DECODE_ROWS + 8, 0);
    assert_eq!(m.prefill_chunk(), SOLO, "the solo prefill chunk of the 35B recipe");

    // text: the repository docs, tokenized by the model's tokenizer
    let codec = intarsia_models::HfCodec::load(&dir, &intarsia_models::ChatTemplateSource::Auto).unwrap();
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut names: Vec<_> = std::fs::read_dir(&docs)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    names.sort();
    let text: String = names.iter().map(|p| std::fs::read_to_string(p).unwrap()).collect::<Vec<_>>().join("\n\n");
    let all = codec.encode(&text, false).unwrap();
    eprintln!("KLGATE corpus: {} docs files, {} tokens", names.len(), all.len());
    let stride = (all.len() - 16384 - 1) / (lens.len() - 1);

    // prefill `prompt` in slices of `chunk` tokens (0: one `forward` over the whole prefix, the solo engine's schedule), the
    // last prompt token alone (the recurrent schedule of the engine), then `rows` decode steps; returns (rows of logits, tokens)
    let run = |prompt: &[u32], chunk: usize, forced: Option<&[u32]>| -> (Vec<Vec<f32>>, Vec<u32>) {
        let mut st = m.new_state().unwrap();
        let (head, last) = prompt.split_at(prompt.len() - 1);
        if chunk == 0 {
            m.forward(&mut st, head, HeadMode::None, None).unwrap();
        } else {
            for c in head.chunks(chunk) {
                m.forward(&mut st, c, HeadMode::None, None).unwrap();
            }
        }
        let logits = |o: StepOut| match o {
            StepOut::Logits(l) => l,
            o => panic!("expected logits, got {o:?}"),
        };
        let mut rows = vec![logits(m.forward(&mut st, last, HeadMode::LastLogits, None).unwrap())];
        let mut toks = Vec::new();
        for i in 0..DECODE_ROWS {
            let next = match forced {
                Some(f) => f[i],
                None => {
                    rows.last()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                        .0 as u32
                }
            };
            toks.push(next);
            rows.push(logits(m.forward(&mut st, &[next], HeadMode::LastLogits, None).unwrap()));
        }
        (rows, toks)
    };
    // KL(p || q) over the full vocabulary in f64, and top-1 agreement, for rows[1..] (the decode rows) and row 0
    let kl = |p: &[f32], q: &[f32]| -> f64 {
        let lse = |x: &[f32]| {
            let mx = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
            mx + x.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln()
        };
        let (lp, lq) = (lse(p), lse(q));
        p.iter()
            .zip(q)
            .map(|(&a, &b)| {
                let lpa = a as f64 - lp;
                lpa.exp() * (lpa - (b as f64 - lq))
            })
            .sum()
    };
    let top1 =
        |x: &[f32]| x.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
    // the estimator of the ref-noise floor: 21-way KL(p || q), the 20 largest probabilities of p plus the tail bucket
    let kl21 = |p: &[f32], q: &[f32]| -> f64 {
        let norm = |x: &[f32]| {
            let mx = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let z: f64 = x.iter().map(|&v| (v as f64 - mx).exp()).sum();
            move |v: f32| (v as f64 - mx).exp() / z
        };
        let (np, nq) = (norm(p), norm(q));
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.select_nth_unstable_by(20, |&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        let (mut sp, mut sq, mut kl) = (0f64, 0f64, 0f64);
        for &i in &idx[..20] {
            let (a, b) = (np(p[i]), nq(q[i]));
            sp += a;
            sq += b;
            kl += a * (a / b).ln();
        }
        let (tp, tq) = ((1.0 - sp).max(1e-12), (1.0 - sq).max(1e-12));
        kl + tp * (tp / tq).ln()
    };
    let mut worst_mean = 0f64;
    let mut sum_all = (0f64, 0usize, 0usize, 0usize);
    let mut ctl_max = 0f64;
    let mut failed: Vec<String> = Vec::new();
    let mut failed_full: Vec<String> = Vec::new();
    let mut sum21 = 0f64;
    for (k, &len) in lens.iter().enumerate() {
        let prompt = &all[k * stride..k * stride + len];
        let (solo, toks) = run(prompt, 0, None);
        // control: the solo chunk by hand is the same computation
        let (hand, _) = run(prompt, SOLO, Some(&toks));
        let ctl = (0..solo.len()).map(|i| kl(&solo[i], &hand[i])).fold(0f64, f64::max);
        ctl_max = ctl_max.max(ctl);
        for chunk in [1024usize, 256] {
            let (busy, _) = run(prompt, chunk, Some(&toks));
            let kls: Vec<f64> = (1..solo.len()).map(|i| kl(&solo[i], &busy[i])).collect();
            let k21: Vec<f64> = (1..solo.len()).map(|i| kl21(&solo[i], &busy[i])).collect();
            let mean21 = k21.iter().sum::<f64>() / k21.len() as f64;
            let flips = (1..solo.len()).filter(|&i| top1(&solo[i]) != top1(&busy[i])).count();
            let mean = kls.iter().sum::<f64>() / kls.len() as f64;
            let max = kls.iter().cloned().fold(0f64, f64::max);
            let kl0 = kl(&solo[0], &busy[0]);
            eprintln!(
                "KLGATE len {len:>5} chunk {chunk:>4} vs solo {SOLO}: decode rows {} mean KL {mean:.3e} (21-way {mean21:.3e}) max {max:.3e} top-1 flips {flips}/{} | prefill-last row KL {kl0:.3e} flip {} | control (hand 2048 vs forward) max KL {ctl:.1e}",
                kls.len(), kls.len(), top1(&solo[0]) != top1(&busy[0])
            );
            let head: Vec<String> = kls.iter().take(8).map(|v| format!("{v:.1e}")).collect();
            let tail = kls[8..].iter().sum::<f64>() / (kls.len() - 8) as f64;
            eprintln!("KLGATE   rows 1-8 KL [{}] rows 9-64 mean {tail:.3e}", head.join(" "));
            if chunk == 1024 {
                worst_mean = worst_mean.max(mean);
                sum_all.0 += kls.iter().sum::<f64>();
                sum_all.1 += kls.len();
                sum_all.2 += flips;
                sum_all.3 += 1;
                sum21 += k21.iter().sum::<f64>();
                if mean21 > FLOOR {
                    failed.push(format!("len {len}: mean 21-way KL {mean21:.3e}"));
                }
                if mean > FLOOR {
                    failed_full.push(format!("len {len}: mean full-vocab KL {mean:.3e}"));
                }
            }
        }
    }
    eprintln!(
        "KLGATE SUMMARY chunk 1024 vs 2048: {} prompts, {} decode rows, mean KL {:.3e} (21-way {:.3e}), worst prompt mean {:.3e}, top-1 flips {}/{} | floor {FLOOR:.3e} | control max KL {ctl_max:.1e}",
        sum_all.3, sum_all.1, sum_all.0 / sum_all.1 as f64, sum21 / sum_all.1 as f64, worst_mean, sum_all.2, sum_all.1
    );
    assert_eq!(ctl_max, 0.0, "the harness: the solo chunk by hand must equal forward() of the prefix");
    eprintln!("KLGATE full-vocab prompts above the floor: {failed_full:?}");
    assert!(failed.is_empty(), "above the ref-noise floor {FLOOR:.3e}: {failed:?}");
}

// ---- engine identity on prompts longer than the solo chunk ----------------------------------------------------------------

/// The busy chunk of the batching engine defaults to the solo chunk (2048) for the 35B on CUDA, so a request that prefills while
/// others decode runs the solo chunk schedule: its output must be token for token the `LocalEngine` run, for prompts of 2.6k, 4k
/// and 16k tokens (more than one chunk), with 2 and 4 slots. The decoding neighbour (R0, a short prompt, 900 tokens) is still
/// generating while the long prompts prefill (checked from the scheduler's mixed-iteration counter). Control, printed and not
/// asserted: the same scenario with `--busy-chunk 1024` (the opt-in knob), which is KL-gated and not bitwise.
#[test]
#[ignore]
fn real_35b_batched_engine_long_prompts_equal_the_local_engine() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let all = corpus_tokens(&dir);
    let window = |start: usize, len: usize| all[start..start + len].to_vec();
    let mk = |max_batch: Option<usize>, busy_chunk: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 16384 + 256, kv_fraction: 0.95, profile_every: 0, ..Default::default() },
        max_batch,
        busy_chunk,
        ..Default::default()
    };
    // (name, request): R0 the decoding neighbour, then the long prompts
    let req = |prompt: Vec<u32>, gen: usize, sampled: Option<u64>| {
        let mut r = GenRequest { prompt, max_tokens: gen, ignore_eos: true, ..Default::default() };
        if let Some(seed) = sampled {
            r.temperature = 0.8;
            r.top_k = 40;
            r.top_p = 0.95;
            r.seed = Some(seed);
        }
        r
    };
    let reqs = || {
        vec![
            ("R0 200 tok, 900 gen", req(window(500, 200), 900, None)),
            ("R1 4096 tok", req(window(3000, 4096), 130, None)),
            ("R2 16384 tok (sampled)", req(window(20000, 16384), 130, Some(21))),
            ("R3 2600 tok", req(window(60000, 2600), 130, None)),
        ]
    };
    let reference: Vec<Out> = {
        let l = engine(&dir, &be, &mk(None, None));
        assert_eq!(l.max_batch, 1);
        reqs().into_iter().map(|(_, r)| collect(&l.engine.submit(r).unwrap())).collect()
    };
    assert!(reference.iter().zip(reqs()).all(|(o, (_, r))| o.ids.len() == r.max_tokens));
    // `slots` slots; request indices to run (R0 first, the rest together once R0 is decoding)
    let scenario = |slots: usize,
                    which: &[usize],
                    busy_chunk: Option<usize>|
     -> (Vec<Out>, intarsia_core::scheduler::BatchMetrics) {
        let l = engine(&dir, &be, &mk(Some(slots), busy_chunk));
        assert_eq!(l.max_batch, slots);
        let all_reqs = reqs();
        let rx0 = l.engine.submit(all_reqs[which[0]].1.clone()).unwrap();
        // R0 decoding: its first token is out
        let first = rx0.recv_timeout(Duration::from_secs(600)).expect("R0 first token");
        let rest: Vec<_> = which[1..].iter().map(|&i| l.engine.submit(all_reqs[i].1.clone()).unwrap()).collect();
        let mut outs = Vec::new();
        std::thread::scope(|sc| {
            let h0 = sc.spawn(move || {
                // the event already taken, then the rest of the stream
                let (mut ids, mut text, mut reason) = (vec![], String::new(), None);
                let mut ev = Some(first);
                loop {
                    let e = ev
                        .take()
                        .unwrap_or_else(|| rx0.recv_timeout(Duration::from_secs(600)).expect("engine stalled"));
                    match e {
                        GenEvent::Token { id, text: t } => {
                            ids.push(id);
                            text.push_str(&t);
                        }
                        GenEvent::Done { finish_reason, .. } => {
                            reason = Some(finish_reason);
                            break;
                        }
                        GenEvent::Error { message } => panic!("request failed: {message}"),
                    }
                }
                Out { ids, text, reason }
            });
            let hs: Vec<_> = rest.into_iter().map(|rx| sc.spawn(move || collect(&rx))).collect();
            outs.push(h0.join().unwrap());
            outs.extend(hs.into_iter().map(|h| h.join().unwrap()));
        });
        (outs, l.engine.batch_metrics().expect("a batching engine has metrics"))
    };
    let mut failures = Vec::new();
    for (slots, which) in [(2usize, vec![0usize, 1]), (4, vec![0, 1, 2, 3]), (4, vec![0, 2, 3, 1])] {
        let (outs, m) = scenario(slots, &which, None);
        let hist: Vec<String> =
            m.batch_hist.iter().enumerate().filter(|(_, &n)| n > 0).map(|(b, n)| format!("B{b}:{n}")).collect();
        eprintln!(
            "LONGENG {slots} slots, requests {which:?}, busy chunk default: mixed iterations {}, prefill chunks {}, decode batches {}",
            m.mixed_iterations, m.prefill_chunks, hist.join(" ")
        );
        assert!(
            m.mixed_iterations > 0,
            "no iteration prefilled beside decoding: the scenario did not exercise the busy path"
        );
        for (k, &i) in which.iter().enumerate() {
            let same = outs[k] == reference[i];
            eprintln!(
                "LONGENG   {} : {}",
                reqs()[i].0,
                if same { "IDENTICAL to the LocalEngine run" } else { "DIFFERS" }
            );
            if !same {
                let first = outs[k].ids.iter().zip(&reference[i].ids).position(|(a, b)| a != b);
                failures.push(format!("{slots} slots, {}: first differing token {first:?}", reqs()[i].0));
            }
        }
    }
    // control: the opt-in 1024 chunk
    let (outs, _) = scenario(4, &[0, 1, 2, 3], Some(1024));
    for (k, i) in [0usize, 1, 2, 3].iter().enumerate() {
        let first = outs[k].ids.iter().zip(&reference[*i].ids).position(|(a, b)| a != b);
        eprintln!(
            "LONGENG control busy chunk 1024: {} : {}",
            reqs()[*i].0,
            match first {
                None => "identical".to_string(),
                Some(t) => format!("first differing token {t}"),
            }
        );
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// Diagnostic (ignored): the ops and kernels of one solo decode step and of a 4-row and a 16-row batched step of the 35B, with
/// their call counts. What the R3 work replaces (per-row launches) is the difference between the two lists.
#[test]
#[ignore]
fn real_35b_batch_op_listing() {
    let _serial = serial();
    use intarsia_core::plan::Phase;
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let m = load(&dir, be, 16, 2048);
    let count = |rec: &intarsia_core::plan::Recorder| {
        let mut v: std::collections::BTreeMap<(String, String), usize> = Default::default();
        for c in rec.calls.iter().filter(|c| !c.extra) {
            *v.entry((c.call.op.to_string(), c.kernel.clone())).or_default() += 1;
        }
        v
    };
    for (name, phase, rows) in
        [("solo", Phase::Decode, 1usize), ("batch4", Phase::Batch, 4), ("batch16", Phase::Batch, 16)]
    {
        let rec = m.dry_run(phase, rows, 100).unwrap();
        let v = count(&rec);
        let total: usize = v.values().sum();
        eprintln!("OPLIST {name}: {total} launches");
        for ((op, k), n) in &v {
            eprintln!("OPLIST   {name:<8} {n:>5} x {op:<26} {k}");
        }
    }
}

// ---- packed prefill (INTARSIA_PACK_PREFILL=1) ----------------------------------------------------------------------------

/// Prompts that prefill together run their chunks in one pass (`Model::forward_packed`); each request must come out exactly as
/// on the single-sequence engine. 24 requests of 40..1100 tokens of repository docs text (some sampled), submitted at once on
/// 8 and 16 slots, so several are prefilling together; at least one pass must have packed two or more chunks. `--ignored`: it
/// loads the model four times.
#[test]
#[ignore]
fn real_35b_packed_prefill_equals_the_local_engine() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let all = corpus_tokens(&dir);
    let lens = [300usize, 1100, 40, 700, 520, 90, 1024, 260, 33, 880, 410, 150];
    let reqs = || -> Vec<GenRequest> {
        (0..24)
            .map(|i| {
                let len = lens[i % lens.len()] + (i / lens.len()) * 17;
                let start = 1000 + i * 1300;
                let mut r = GenRequest { prompt: all[start..start + len].to_vec(), max_tokens: 100, ignore_eos: true, ..Default::default() };
                if i % 4 == 3 {
                    r.temperature = 0.8;
                    r.top_k = 40;
                    r.top_p = 0.95;
                    r.seed = Some(31 + i as u64);
                }
                r
            })
            .collect()
    };
    let mk = |max_batch: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 2048, kv_fraction: 0.95, profile_every: 0, ..Default::default() },
        max_batch,
        ..Default::default()
    };
    std::env::remove_var("INTARSIA_PACK_PREFILL");
    let reference: Vec<Out> = {
        let l = load_engine(&dir, be.clone(), &mk(None)).unwrap();
        reqs().into_iter().map(|r| collect(&l.engine.submit(r).unwrap())).collect()
    };
    for slots in [8usize, 16] {
        std::env::set_var("INTARSIA_PACK_PREFILL", "1");
        let l = load_engine(&dir, be.clone(), &mk(Some(slots))).unwrap();
        std::env::remove_var("INTARSIA_PACK_PREFILL");
        let rxs: Vec<_> = reqs().into_iter().map(|r| l.engine.submit(r).unwrap()).collect();
        for (i, rx) in rxs.iter().enumerate() {
            assert_eq!(&collect(rx), &reference[i], "request {i} ({} prompt tokens) on {slots} slots, packed prefill", reqs()[i].prompt.len());
        }
        let m = l.engine.batch_metrics().expect("batched engine metrics");
        eprintln!("{slots} slots: {} packed passes carried {} chunks ({} chunks in all)", m.pack_passes, m.packed_chunks, m.prefill_chunks);
        assert!(m.pack_passes > 0 && m.packed_chunks >= 2 * m.pack_passes, "nothing was packed: {m:?}");
    }
}

// ---- macro prefill (INTARSIA_PREFILL_MACRO) on the real 35B ---------------------------------------------------------------

/// Macro prefill (several chunks of one prompt in one layer-by-layer pass, `Model::macro_rows`) must leave the bits of the plain
/// chunk schedule: for prompts of 1.5k, 4k, 8.3k and 16.6k tokens of repository docs text, the same weights prefill the prompt with
/// the macro schedule off, then on (macro 4096, 8192 and 16384 rows), and the logits of the last prompt row and of 16 greedy
/// decode steps (which read the KV cache and the recurrent state the prefill wrote) must be bit for bit the same. Also a
/// single `forward` over the whole prompt with a head. Opt-in: `INTARSIA_QWEN36_DIR`, a fleetq slot on the Spark.
#[test]
#[ignore]
fn real_35b_macro_prefill_is_bitwise_the_plain_schedule() {
    let _serial = serial();
    use intarsia_core::model::{HeadMode, StepOut};
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    const DECODE_ROWS: usize = 16;
    let lens = [1536usize, 4096, 8192 + 100, 16384 + 271];
    let (_, recipe, dims, _) = read_model_spec(&dir, &EngineOptions::default()).unwrap();
    let weights = SchemedSource::open(&dir).unwrap();
    let chunk = recipe.resolve_prefill_chunk(None, be.name()).value;
    let opts = LoadOptions {
        max_seq: 16384 + 271 + DECODE_ROWS + 8,
        prefill_chunk: chunk,
        profile_every: 0,
        kv_fraction: 0.95,
        prefill_macro: Some(16384),
        ..Default::default()
    };
    let mut m = Model::load(be, &recipe, &dims, &weights, &opts, Arc::new(Telemetry::new())).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    assert_eq!(m.macro_rows(), 16384);
    let all = corpus_tokens(&dir);
    assert!(all.len() > 16384 + 271, "{} corpus tokens", all.len());
    let logits = |o: StepOut| match o {
        StepOut::Logits(l) => l,
        o => panic!("expected logits, got {o:?}"),
    };
    // the engine's recurrent schedule: everything but the last token headless, the last alone with the head, then greedy decode
    let run = |m: &Model, prompt: &[u32], whole: bool| -> Vec<Vec<f32>> {
        let mut st = m.new_state().unwrap();
        let mut rows = Vec::new();
        if whole {
            rows.push(logits(m.forward(&mut st, prompt, HeadMode::LastLogits, None).unwrap()));
        } else {
            let (head, last) = prompt.split_at(prompt.len() - 1);
            m.forward(&mut st, head, HeadMode::None, None).unwrap();
            rows.push(logits(m.forward(&mut st, last, HeadMode::LastLogits, None).unwrap()));
        }
        for _ in 0..DECODE_ROWS {
            let next = rows
                .last()
                .unwrap()
                .iter()
                .enumerate()
                .fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                .0 as u32;
            rows.push(logits(m.forward(&mut st, &[next], HeadMode::LastLogits, None).unwrap()));
        }
        rows
    };
    let bits = |r: &Vec<Vec<f32>>| -> Vec<Vec<u32>> { r.iter().map(|l| l.iter().map(|x| x.to_bits()).collect()).collect() };
    for (k, &len) in lens.iter().enumerate() {
        let prompt = &all[k * 997..k * 997 + len];
        for whole in [false, true] {
            m.set_macro_rows(0);
            let want = bits(&run(&m, prompt, whole));
            for rows in [4096usize, 8192, 16384] {
                m.set_macro_rows(rows);
                let before = m.macro_passes();
                let got = bits(&run(&m, prompt, whole));
                let passes = m.macro_passes() - before;
                eprintln!("MACRO len {len} whole {whole} macro {rows}: {passes} macro passes, logits of {} rows", got.len());
                // a macro size below two chunks is the plain schedule by definition (`effective_macro_rows`; HIP: chunk 4096, so
                // macro 4096 is one chunk): only then may a long prompt run without a macro pass. `rows` and `chunk` are the test's
                // own numbers, not what the model reports, so a model that silently dropped the macro schedule still fails here.
                let macro_expected = len >= 2 * chunk && rows >= 2 * chunk;
                assert!(!macro_expected || passes > 0, "len {len}, macro {rows}, chunk {chunk}: the macro schedule did not run");
                assert!(macro_expected || passes == 0, "len {len}, macro {rows}, chunk {chunk}: a macro pass ran below two chunks");
                assert!(got == want, "len {len}, whole {whole}, macro {rows}: logits differ from the plain schedule");
            }
        }
    }
}

/// The batched engine's macro prefill (a prompt prefilling alone with nothing decoding runs several 2048-row chunks as one
/// pass) is bitwise the plain schedule: the reference is the LocalEngine with the macro schedule OFF (`prefill_macro: Some(0)`),
/// the batched engine runs the recipe default (8192). Prompts of 2.6k, 4k, 8.3k and 16.4k tokens, each submitted alone (macro
/// steps), then all four at once (plain chunks while several prefill, macro steps for the last one standing); a sampled one
/// among them. The scheduler's chunk counter must show fewer prefill calls than the plain schedule's.
#[test]
#[ignore]
fn real_35b_batched_macro_prefill_equals_the_plain_local_engine() {
    let _serial = serial();
    let (Some(dir), Some(be)) = (model_dir("INTARSIA_QWEN36_DIR"), cuda()) else { return };
    let all = corpus_tokens(&dir);
    let mk = |max_batch: Option<usize>, macro_rows: Option<usize>| EngineOptions {
        load: LoadOptions { max_seq: 16384 + 256, kv_fraction: 0.95, profile_every: 0, prefill_macro: macro_rows, ..Default::default() },
        max_batch,
        ..Default::default()
    };
    let lens = [2600usize, 4096, 8192 + 100, 16384];
    let reqs = || -> Vec<GenRequest> {
        lens.iter()
            .enumerate()
            .map(|(i, &n)| {
                let start = 1000 + i * 21000;
                let mut r = GenRequest { prompt: all[start..start + n].to_vec(), max_tokens: 100, ignore_eos: true, ..Default::default() };
                if i == 2 {
                    r.temperature = 0.8;
                    r.top_k = 40;
                    r.top_p = 0.95;
                    r.seed = Some(21);
                }
                r
            })
            .collect()
    };
    let reference: Vec<Out> = {
        let l = load_engine(&dir, be.clone(), &mk(None, Some(0))).unwrap();
        reqs().into_iter().map(|r| collect(&l.engine.submit(r).unwrap())).collect()
    };
    let plain_chunks: usize = lens.iter().map(|&n| n.div_ceil(2048)).sum::<usize>();
    let l = load_engine(&dir, be.clone(), &mk(Some(4), None)).unwrap();
    let before = l.engine.batch_metrics().expect("batched metrics").prefill_chunks;
    for (i, r) in reqs().into_iter().enumerate() {
        let got = collect(&l.engine.submit(r).unwrap());
        assert_eq!(&got, &reference[i], "prompt of {} tokens, alone", lens[i]);
    }
    let alone = l.engine.batch_metrics().unwrap().prefill_chunks - before;
    eprintln!("MACRO batched alone: {alone} prefill calls (plain schedule: {} chunks)", plain_chunks);
    assert!(alone < plain_chunks as u64, "no macro step ran: {alone} calls");
    let rxs: Vec<_> = reqs().into_iter().map(|r| l.engine.submit(r).unwrap()).collect();
    for (i, rx) in rxs.iter().enumerate() {
        assert_eq!(&collect(rx), &reference[i], "prompt of {} tokens, together", lens[i]);
    }
}
