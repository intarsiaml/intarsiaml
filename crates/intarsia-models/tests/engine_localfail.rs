//! Failure paths of the single-sequence engine: one bad request or one device error must never leave an engine that
//! reports healthy while it fails every request (docs/results/2026-11-03-engine-localfail.md).
//!
//! `Flaky` (tests/common) wraps the cpu backend and fails launches on demand, with the error of the test's choosing.
mod common;
use common::*;
use intarsia_core::backend::Backend;
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::{DType, Engine, EngineMeta, Error, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn opts() -> LoadOptions {
    LoadOptions { max_seq: 64, kv_dtype: Some(DType::F32), act_dtype: Some(DType::F32), ..Default::default() }
}

fn engine_of(m: Model) -> LocalEngine {
    let tel = Arc::new(Telemetry::new());
    LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), eos_ids: vec![], ..Default::default() }, tel).unwrap()
}

/// One engine over one `Flaky` device.
fn solo() -> (LocalEngine, Arc<Flaky>) {
    let t = tiny_weights(DType::F32, 1);
    let be = Flaky::new();
    let m = Model::load(be.clone(), &recipe(), &dims(), &t.weights, &opts(), Arc::new(Telemetry::new())).unwrap();
    (engine_of(m), be)
}

/// One engine over two ranks joined by the host communicator; rank 0 runs on the `Flaky` device.
fn tp2() -> (LocalEngine, Arc<Flaky>) {
    let (e, bes) = tp2_flaky();
    (e, bes[0].clone())
}

/// [`tp2`] with a `Flaky` device on each rank.
fn tp2_flaky() -> (LocalEngine, [Arc<Flaky>; 2]) {
    let t = tiny_weights(DType::F32, 1);
    let bes = [Flaky::new(), Flaky::new()];
    let backends: Vec<Arc<dyn Backend>> = bes.iter().map(|b| b.clone() as Arc<dyn Backend>).collect();
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    let m = Model::load_tp(backends, comms, &recipe(), &dims(), &t.weights, &opts(), Arc::new(Telemetry::new())).unwrap();
    (engine_of(m), bes)
}

fn req(prompt: Vec<u32>) -> GenRequest {
    GenRequest { prompt, max_tokens: 3, ..Default::default() }
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 10 + i * 3).collect()
}

/// The events of a request up to its end: `Ok((finish reason, cached_tokens))` for `Done`, `Err(message)` for `Error`.
fn run(e: &dyn Engine, r: GenRequest) -> std::result::Result<(FinishReason, usize), String> {
    let rx = e.submit(r).map_err(|e| format!("rejected: {e}"))?;
    loop {
        match rx.recv().expect("the worker answers every request") {
            GenEvent::Token { .. } => {}
            GenEvent::Done { finish_reason, cached_tokens, .. } => return Ok((finish_reason, cached_tokens)),
            GenEvent::Error { message } => return Err(message),
        }
    }
}

/// [`run`] to the cached token count.
fn finish(e: &dyn Engine, r: GenRequest) -> std::result::Result<usize, String> {
    run(e, r).map(|(_, cached)| cached)
}

/// A prompt id at or past the vocabulary is the client's mistake: it must not abort the tensor-parallel communicator for the
/// requests after it (before the fix every later request failed with "communicator aborted" while `/health` stayed 200).
#[test]
fn out_of_vocab_prompt_leaves_the_tp_group_working() {
    let (e, _be) = tp2();
    assert!(finish(&e, req(prompt(6))).is_ok(), "the group works");
    for bad in [V as u32, V as u32 + 5, u32::MAX] {
        let mut p = prompt(6);
        p[3] = bad;
        assert!(finish(&e, req(p)).is_err(), "id {bad} is refused");
        // the one-token prompt takes the graph-step path
        assert!(finish(&e, req(vec![bad])).is_err(), "id {bad} alone is refused");
        assert_eq!(e.failed(), None, "a bad request does not fail a healthy engine");
        finish(&e, req(prompt(8))).unwrap_or_else(|m| panic!("after id {bad} the group fails: {m}"));
    }
}

/// Refused on `submit`, as a `BadRequest` the server answers with 400 (it was a 500, or a 200 stream ending in an error
/// event), before the request queues or touches the cache; the id just below the vocabulary is a token like any other.
#[test]
fn out_of_vocab_prompt_is_a_bad_request() {
    for (e, _be) in [solo(), tp2()] {
        assert!(finish(&e, req(prompt(6))).is_ok());
        for bad in [V as u32, V as u32 + 5, u32::MAX] {
            let mut p = prompt(6);
            p[0] = bad;
            match e.submit(req(p)) {
                Err(Error::BadRequest(m)) => assert!(m.contains(&bad.to_string()) && m.contains("vocab"), "{m}"),
                other => panic!("id {bad}: expected a BadRequest, got {:?}", other.map(|_| ())),
            }
        }
        let mut p = prompt(6);
        p[0] = V as u32 - 1;
        assert!(finish(&e, req(p)).is_ok());
        assert_eq!(e.failed(), None);
        let t = e.telemetry();
        assert_eq!((t.running, t.waiting), (0, 0), "a refused request leaves no count behind");
    }
}

/// A request that only the client got wrong (too long, empty) is an error for it and leaves a healthy engine as it was,
/// on one device and on a group.
#[test]
fn bad_requests_are_not_fatal() {
    for (e, _be) in [solo(), tp2()] {
        assert!(finish(&e, req(prompt(6))).is_ok());
        let too_long = finish(&e, req(prompt(70))).unwrap_err();
        assert!(too_long.contains("context window"), "{too_long}");
        let empty = finish(&e, req(Vec::new())).unwrap_err();
        assert!(empty.contains("empty prompt"), "{empty}");
        assert_eq!(e.failed(), None);
        finish(&e, req(prompt(8))).expect("the engine still serves");
    }
}

/// A device error stays (a wedged context fails every launch after it): the engine fails with the first one, so the
/// server answers 503, `/health` says so and the supervisor restarts the process, instead of 500 for every request.
#[test]
fn device_error_fails_the_engine() {
    let (e, be) = solo();
    assert!(finish(&e, req(prompt(6))).is_ok());
    be.set(FAIL_BACKEND);
    let err = finish(&e, req(prompt(6))).unwrap_err();
    assert!(err.contains("injected device error"), "{err}");
    let why = e.failed().expect("the device error fails the engine");
    assert!(why.contains("injected device error"), "{why}");
    be.set(OK);
    assert!(e.submit(req(prompt(6))).is_err(), "a failed engine accepts no work");
    assert!(e.forward_logits(&[65]).is_err());
}

/// An error the backend calls one-off (a Metal command buffer that ended in Error status: the next buffer runs, the
/// speculative pipeline resets after it) is that request's error: the engine keeps serving instead of reloading.
#[test]
fn a_transient_backend_error_does_not_fail_the_engine() {
    let (e, be) = solo();
    be.set_transient(true);
    assert!(finish(&e, req(prompt(6))).is_ok());
    be.set(FAIL_BACKEND);
    let err = finish(&e, req(prompt(6))).unwrap_err();
    assert!(err.contains("injected device error"), "{err}");
    be.set(OK);
    assert_eq!(e.failed(), None, "a one-off backend error fails the engine");
    finish(&e, req(prompt(6))).expect("serves again");
    // the same error from a backend that calls it sticky does fail it (`device_error_fails_the_engine`)
    be.set_transient(false);
    be.set(FAIL_BACKEND);
    assert!(finish(&e, req(prompt(6))).is_err());
    assert!(e.failed().is_some());
}

/// Requests already queued behind the one that hits the device error are answered, each with an error, not dropped.
#[test]
fn queued_requests_drain_after_a_device_error() {
    let (e, be) = solo();
    be.set(FAIL_BACKEND);
    let rxs: Vec<_> = (0..4).filter_map(|_| e.submit(req(prompt(6))).ok()).collect();
    assert!(!rxs.is_empty(), "the first request is accepted");
    for rx in rxs {
        match rx.recv().expect("answered") {
            GenEvent::Error { .. } => {}
            other => panic!("expected an error, got {other:?}"),
        }
    }
    assert!(e.failed().is_some());
    let t = e.telemetry();
    assert_eq!((t.running, t.waiting), (0, 0), "counters stay consistent");
}

/// A failed pass aborts a tensor-parallel communicator for good, whatever the error that ended it: the engine fails instead
/// of answering every later request with "communicator aborted" while `/health` stays 200. The backends here call their errors
/// transient, so it is the group's abort alone that fails the engine (a peer's failure reaches rank 0 as the communicator's
/// own "aborted" `Backend` error, which a default backend would call a device fault).
#[test]
fn aborted_tp_group_fails_the_engine() {
    for rank in [0, 1] {
        let (e, bes) = tp2_flaky();
        bes[0].set_transient(true);
        assert!(finish(&e, req(prompt(6))).is_ok());
        assert_eq!(e.failed(), None);
        bes[rank].set(FAIL_MEMORY);
        assert!(finish(&e, req(prompt(6))).is_err(), "rank {rank}");
        bes[rank].set(OK);
        let why = e.failed().unwrap_or_else(|| panic!("rank {rank}: the aborted group left the engine healthy"));
        assert!(why.contains("tensor-parallel group aborted"), "{why}");
        assert!(e.submit(req(prompt(6))).is_err(), "a failed engine accepts no work");
    }
}

/// The same error on one device is the request's own: a failed allocation does not fail an engine without a group.
#[test]
fn memory_error_on_one_device_is_not_fatal() {
    let (e, be) = solo();
    be.set(FAIL_MEMORY);
    assert!(finish(&e, req(prompt(6))).unwrap_err().contains("injected allocation failure"));
    be.set(OK);
    assert_eq!(e.failed(), None);
    finish(&e, req(prompt(6))).expect("serves again");
}

/// A request cancelled while it waited (its client went away) is answered without touching the cache: the conversation
/// before it keeps its prefix (it was cut to nothing, and the next turn prefilled from scratch).
#[test]
fn cancelled_queued_request_keeps_the_warm_cache() {
    let a = prompt(20);
    let mut next = a.clone();
    next.extend([7, 8, 9]);
    // what the follow-up reuses with nothing in between
    let reference = {
        let (e, _) = solo();
        finish(&e, req(a.clone())).unwrap();
        finish(&e, req(next.clone())).unwrap()
    };
    assert!(reference >= a.len(), "the follow-up reuses the conversation ({reference} tokens)");

    let (e, _) = solo();
    finish(&e, req(a)).unwrap();
    // a different prompt whose cancel flag is set before the worker pops it
    let mut other = prompt(20);
    other[0] += 1;
    let cancel = Arc::new(AtomicBool::new(true));
    let (reason, cached) = run(&e, GenRequest { cancel: Some(cancel), ..req(other) }).unwrap();
    assert_eq!((reason, cached), (FinishReason::Cancelled, 0));
    assert_eq!(finish(&e, req(next)).unwrap(), reference, "the cancelled request wiped the cache");
    assert_eq!(e.failed(), None);
    let t = e.telemetry();
    assert_eq!((t.running, t.waiting), (0, 0), "counters stay consistent");
}
