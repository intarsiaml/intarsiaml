//! Failure paths of the single-sequence engine: one bad request or one device error must never leave an engine that
//! reports healthy while it fails every request (docs/results/2026-11-03-engine-localfail.md).
//!
//! `Flaky` wraps the cpu backend and fails launches on demand, with the error of the test's choosing: a stand-in for a
//! sticky device error (`Error::Backend`) or an allocation failure on one rank (`Error::Memory`).
mod common;
use common::*;
use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::{
    DType, Engine, EngineMeta, Error, FinishReason, GenEvent, GenRequest, KernelImpl, LocalEngine, OpCall, Result, Telemetry,
};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

const OK: u8 = 0;
const FAIL_BACKEND: u8 = 1;
const FAIL_MEMORY: u8 = 2;

/// The cpu backend that fails every launch while `mode` is not `OK`.
struct Flaky {
    inner: Arc<dyn Backend>,
    mode: AtomicU8,
}

impl Flaky {
    fn new() -> Arc<Flaky> {
        Arc::new(Flaky { inner: intarsia_backend_cpu::CpuBackend::arc(), mode: AtomicU8::new(OK) })
    }
    fn set(&self, mode: u8) {
        self.mode.store(mode, Ordering::SeqCst);
    }
}

impl Backend for Flaky {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn info(&self) -> DeviceInfo {
        self.inner.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> Result<Buffer> {
        self.inner.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> Result<()> {
        self.inner.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> Result<()> {
        self.inner.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> Result<()> {
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.inner.kernels()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        match self.mode.load(Ordering::SeqCst) {
            FAIL_BACKEND => return Err(Error::backend("injected device error (test)")),
            FAIL_MEMORY => return Err(Error::Memory("injected allocation failure (test)".into())),
            _ => {}
        }
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.inner.synchronize(s)
    }
}

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
    let t = tiny_weights(DType::F32, 1);
    let be = Flaky::new();
    let backends: Vec<Arc<dyn Backend>> = vec![be.clone(), intarsia_backend_cpu::CpuBackend::arc()];
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    let m = Model::load_tp(backends, comms, &recipe(), &dims(), &t.weights, &opts(), Arc::new(Telemetry::new())).unwrap();
    (engine_of(m), be)
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
