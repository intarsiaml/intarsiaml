//! Fused all-reduce + residual add + rmsnorm (`Comm::all_reduce_add_rmsnorm`, `INTARSIA_TP_ARNORM`) at the model level on the
//! cpu backend: two shards joined by a host-staged communicator that fuses by running the reduction and then the backend's
//! own `RmsNormAdd` (tests/common/arnorm_comm.rs). Whatever the model defers, fuses or flushes, the logits must equal the
//! unfused run's bit for bit; the communicator's counters show which all-reduces took the fused path (no vacuous pass).

mod common;
use common::arnorm_comm::ArNormComm;
use common::*;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::comm::{Comm, HostCommGroup};
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::plan::Phase;
use intarsia_core::{Backend, DType, Op, Telemetry};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard};

/// The loads read `INTARSIA_NO_FUSION` once per process: the tests run one at a time in this binary's env anyway.
static ENV: Mutex<()> = Mutex::new(());

fn env() -> MutexGuard<'static, ()> {
    ENV.lock().unwrap_or_else(|e| e.into_inner())
}

const CHUNK: usize = 4;
const PROMPT: usize = 11; // chunks of 4, 4, 3
const STEPS: usize = 5;

fn opts(dt: DType, arnorm: bool) -> LoadOptions {
        LoadOptions { kv_dtype: Some(dt), prefill_chunk: CHUNK, tp_fusion: true, tp_arnorm: arnorm, ..Default::default() }
}

fn backends() -> Vec<Arc<dyn Backend>> {
    (0..2).map(|_| CpuBackend::arc()).collect()
}

/// A fusing group (payloads above `max_elems` refused).
fn fusing(t: &Tiny, o: &LoadOptions, max_elems: usize) -> (Model, Vec<Arc<ArNormComm>>) {
    let bs = backends();
    let (comms, mine) = ArNormComm::group(bs.clone(), max_elems);
    (Model::load_tp(bs, comms, &recipe(), &dims(), &t.weights, o, Arc::new(Telemetry::new())).unwrap(), mine)
}

/// A plain host-staged group: the unfused sequence.
fn plain(t: &Tiny, o: &LoadOptions) -> Model {
    let bs = backends();
    let comms = HostCommGroup::new(bs.clone());
    Model::load_tp(bs, comms, &recipe(), &dims(), &t.weights, o, Arc::new(Telemetry::new())).unwrap()
}

/// Every logit row of a chunked prefill and of `STEPS` single-token decode steps, as bits.
fn run(m: &Model) -> Vec<Vec<u32>> {
    let mut st = m.new_state().unwrap();
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let ids: Vec<u32> = (0..PROMPT).map(|i| ((i * 37 + 11) % V) as u32).collect();
    let mut push = |_p: usize, r: &[f32]| {
        rows.push(r.iter().map(|x| x.to_bits()).collect());
        Ok(())
    };
    m.forward(&mut st, &ids, HeadMode::AllLogits, Some(&mut push)).unwrap();
    for s in 0..STEPS {
        m.forward(&mut st, &[((s * 53 + 7) % V) as u32], HeadMode::AllLogits, Some(&mut push)).unwrap();
    }
    rows
}

#[test]
fn fused_run_is_bitwise_the_unfused_run_and_every_reduction_fused() {
    let _g = env();
    for dt in [DType::F32, DType::Bf16] {
        let t = tiny_weights(dt, 41);
        let (m, comms) = fusing(&t, &opts(dt, true), usize::MAX);
        let (a, b) = (run(&m), run(&plain(&t, &opts(dt, false))));
        assert_eq!(a.len(), PROMPT + STEPS);
        assert!(a == b, "{dt:?}: the fused run's logits differ from the unfused run's");
        for (r, c) in comms.iter().enumerate() {
            let fused = c.fused.load(Ordering::Relaxed);
            // 2 blocks per layer, each reduction followed by a standalone RmsNormAdd (the next block's or the final norm)
            assert_eq!(fused, 2 * LAYERS * (3 + STEPS), "{dt:?} rank {r}");
            assert_eq!(c.stats().calls as usize, fused, "{dt:?} rank {r}: a reduction ran unfused");
        }
    }
}

#[test]
fn a_payload_the_communicator_refuses_is_reduced_and_normed_separately() {
    let _g = env();
    let dt = DType::Bf16;
    let t = tiny_weights(dt, 42);
    // one row only: the prefill chunks (4 and 3 rows) are refused, the decode steps fuse
    let (m, comms) = fusing(&t, &opts(dt, true), H);
    assert!(run(&m) == run(&plain(&t, &opts(dt, false))));
    for c in &comms {
        assert_eq!(c.fused.load(Ordering::Relaxed), 2 * LAYERS * STEPS);
        assert!(c.rows.lock().unwrap().iter().all(|&r| r == 1));
        assert_eq!(c.stats().calls as usize, 2 * LAYERS * (3 + STEPS), "refused reductions still run, plain");
    }
}

#[test]
fn off_means_the_unfused_sequence() {
    let _g = env();
    let dt = DType::Bf16;
    let t = tiny_weights(dt, 43);
    // the model option off: a fusing communicator is never asked
    let (m, comms) = fusing(&t, &opts(dt, false), usize::MAX);
    let want = run(&plain(&t, &opts(dt, false)));
    assert!(run(&m) == want);
    assert!(comms.iter().all(|c| c.fused.load(Ordering::Relaxed) == 0));
    // the option on, a communicator without the fused call (the host one): nothing defers, same bits
    assert!(run(&plain(&t, &opts(dt, true))) == want);
    // fusion off under TP: the option does nothing
    let o = LoadOptions { tp_fusion: false, ..opts(dt, true) };
    let (m, comms) = fusing(&t, &o, usize::MAX);
    assert!(run(&m) == want);
    assert!(comms.iter().all(|c| c.fused.load(Ordering::Relaxed) == 0));
}

#[test]
fn the_plan_counts_the_fused_collectives_and_keeps_the_ops() {
    let _g = env();
    let dt = DType::Bf16;
    let t = tiny_weights(dt, 44);
    let (m, _c) = fusing(&t, &opts(dt, true), usize::MAX);
    let p = plain(&t, &opts(dt, false));
    // every block-end reduction of the decode step fuses; the count of reductions and of RmsNormAdd calls is the unfused plan's
    assert_eq!(m.resolved.comm_calls_per_decode_step, p.resolved.comm_calls_per_decode_step);
    assert_eq!(m.resolved.comm_fused_norm_per_decode_step, 2 * LAYERS);
    assert_eq!(p.resolved.comm_fused_norm_per_decode_step, 0);
    let norms = |m: &Model| m.resolved.ops.iter().filter(|o| o.op == Op::RmsNormAdd).map(|o| o.calls_per_decode_step).sum::<usize>();
    assert_eq!(norms(&m), norms(&p));
    assert!(norms(&m) > 0);
    let plan = intarsia_core::plan::plan_json(&m, "tiny");
    assert_eq!(plan["tp"]["all_reduce_fused_rmsnorm_per_decode_step"], 2 * LAYERS);
    // a prefill probe past the payload limit would not fuse; the recorder asks the communicator the same question
    let (small, _c) = fusing(&t, &opts(dt, true), H);
    let rec = small.dry_run(Phase::Decode, CHUNK, 0).unwrap();
    assert_eq!(rec.comm.iter().filter(|c| c.fused_norm).count(), 0);
    let rec = small.dry_run(Phase::Decode, 1, 0).unwrap();
    assert_eq!(rec.comm.iter().filter(|c| c.fused_norm).count(), 2 * LAYERS);
}

/// Rank 0 profiles a step (times and synchronises around each collective) while rank 1 never does: both must still issue the
/// same collective, so profiling cannot change whether a reduction fuses.
#[test]
fn profiled_steps_fuse_like_the_others() {
    let _g = env();
    let dt = DType::Bf16;
    let t = tiny_weights(dt, 45);
    let o = LoadOptions { profile_every: 1, ..opts(dt, true) };
    let (m, comms) = fusing(&t, &o, usize::MAX);
    assert!(run(&m) == run(&plain(&t, &opts(dt, false))));
    for c in &comms {
        assert_eq!(c.fused.load(Ordering::Relaxed), 2 * LAYERS * (3 + STEPS));
    }
}
