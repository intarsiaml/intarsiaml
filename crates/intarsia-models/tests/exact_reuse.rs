//! Exact prefix reuse (lane bi8, docs/results/2026-10-29-bi8-serve-batch-invariance.md): on a backend whose dense prefill
//! bits depend on a pass's row count (`Backend::prefix_reuse_any_cut` false: CUDA, Metal since lane prb, HIP since lane arx) a request resumes from a held KV prefix
//! only where its own cold chunk schedule cuts, and only over rows the held sequence wrote with exactly those passes, in the
//! single-sequence engine and in the batched engine's slot choice alike. The cpu kernels are chunk-invariant, so here the
//! bits agree either way: these tests pin the policy (which prefix is reused), the CUDA gates in `batch_cuda_real.rs` the bits.
mod common;
use common::*;
use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::{
    BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, GenEvent, GenRequest, KernelImpl, LocalEngine, OpCall, Result,
    Telemetry,
};
use std::sync::Arc;
use std::time::Duration;

const CHUNK: usize = 8;

/// The cpu backend, reporting a chunk-sensitive prefill as CUDA does.
struct Strict {
    inner: Arc<dyn Backend>,
}

impl Backend for Strict {
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
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.inner.synchronize(s)
    }
    fn prefix_reuse_any_cut(&self) -> bool {
        false
    }
}

fn model(strict: bool, rows: usize) -> (Model, Arc<Telemetry>) {
    let cpu = intarsia_backend_cpu::CpuBackend::arc();
    let backend: Arc<dyn Backend> = if strict { Arc::new(Strict { inner: cpu }) } else { cpu };
    let t = tiny_weights(DType::Bf16, 5);
    let tel = Arc::new(Telemetry::new());
    let mut m = Model::load(
        backend,
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 160, prefill_chunk: CHUNK, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    if rows > 1 {
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    (m, tel)
}

fn meta() -> EngineMeta {
    EngineMeta { id: "tiny".into(), eos_ids: vec![299], ..Default::default() }
}

fn req(prompt: &str) -> GenRequest {
    GenRequest { prompt: prompt.bytes().map(|b| b as u32).collect(), max_tokens: 6, ignore_eos: true, ..Default::default() }
}

/// (generated ids, cached prompt tokens)
fn collect(rx: &std::sync::mpsc::Receiver<GenEvent>) -> (Vec<u32>, usize) {
    let mut ids = vec![];
    loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { cached_tokens, .. } => return (ids, cached_tokens),
            GenEvent::Error { message } => panic!("request failed: {message}"),
        }
    }
}

// a 30-byte prompt: cold chunks (0, 8) (8, 16) (16, 24) (24, 30), so 8, 16 and 24 are the reuse points
const LONG: &str = "abcdefghijklmnopqrstuvwxyz0123";
// shares 20 bytes with LONG: the largest point it may resume from is 16
const SHARED20: &str = "abcdefghijklmnopqrstQQQQQQQQQQQQ";
// a short prompt (one cold chunk): never resumed
const SHORT: &str = "hey you";

/// The sequence of requests both engines run, with the prefix each must reuse on a chunk-sensitive backend and on cpu.
fn cases() -> Vec<(&'static str, usize, usize)> {
    vec![
        // (prompt, cached when strict, cached on cpu: any cut, plen - 1 for a repeat)
        (SHORT, 0, 0),
        (SHORT, 0, SHORT.len() - 1),
        (LONG, 0, 0),
        (LONG, 24, LONG.len() - 1),
        (SHARED20, 16, 20),
    ]
}

#[test]
fn single_sequence_engine_resumes_only_at_the_cold_schedule() {
    for strict in [true, false] {
        let (m, tel) = model(strict, 1);
        let e = LocalEngine::new(m, Arc::new(ByteCodec), meta(), tel).unwrap();
        let cold = |p: &str| {
            let (m, tel) = model(strict, 1);
            collect(&LocalEngine::new(m, Arc::new(ByteCodec), meta(), tel).unwrap().submit(req(p)).unwrap()).0
        };
        for (p, want_strict, want_any) in cases() {
            let (ids, cached) = collect(&e.submit(req(p)).unwrap());
            assert_eq!(cached, if strict { want_strict } else { want_any }, "{p:?} strict={strict}");
            assert_eq!(ids, cold(p), "{p:?} strict={strict}: the warm request computes its cold output");
        }
        // a follow-up turn: LONG plus its generated tokens plus more. Rows 24..30 were the final chunk (6 rows), rows from 30
        // on were written by decode steps: only the 8-row ladder up to 24 matches the new prompt's own cold chunks
        let (first, _) = collect(&e.submit(req(LONG)).unwrap());
        let mut follow = req(LONG);
        follow.prompt.extend(&first);
        follow.prompt.extend("and more text here".bytes().map(|b| b as u32));
        let (_, cached) = collect(&e.submit(follow.clone()).unwrap());
        // the held sequence: the prompt and every generated token but the last (never fed back)
        let any_cut = LONG.len() + req(LONG).max_tokens - 1;
        assert_eq!(cached, if strict { 24 } else { any_cut }, "follow-up turn strict={strict}");
    }
}

#[test]
fn batched_engine_picks_slots_by_the_same_rule() {
    for strict in [true, false] {
        let (m, tel) = model(strict, 2);
        let mut o = BatchedOptions::new(2, 16);
        o.prefix_reuse = true;
        let e = BatchedEngine::new(m, Arc::new(ByteCodec), meta(), tel, o).unwrap();
        // one at a time, so each request finds the previous ones' slots idle
        for (p, want_strict, want_any) in cases() {
            let (_, cached) = collect(&e.submit(req(p)).unwrap());
            if strict {
                assert_eq!(cached, want_strict, "{p:?} strict");
            } else {
                // two slots: the least recently used one may hold the other prompt, so at most the single-sequence figure
                assert!(cached <= want_any, "{p:?}: cached {cached} > {want_any}");
            }
        }
    }
}

/// Lane prb (docs/results/2026-10-29-prb-prefix-reuse-probe.md): the cpu backend's prefill is row-count invariant, which is why
/// it keeps `prefix_reuse_any_cut` (default true): every output element is one fixed-order dot product of a row of activations
/// with a row of weights (and attention is per (row, head)), whatever the number of rows or the first row of a pass. Pinned bit
/// for bit: a 30-token prompt prefilled in one `forward` (chunks 8, 8, 8, 6) and split anywhere (k, 30 - k), the 1-row tail
/// (k = 29) included, gives the same logits for the last prompt row and for the decode steps after it, in bf16 and f32.
#[test]
fn cpu_prefill_is_bitwise_row_count_invariant() {
    use intarsia_core::model::{HeadMode, StepOut};
    let logits = |o: StepOut| match o {
        StepOut::Logits(l) => l,
        o => panic!("expected logits, got {o:?}"),
    };
    for dt in [DType::Bf16, DType::F32] {
        let t = tiny_weights(dt, 5);
        let m = Model::load(
            intarsia_backend_cpu::CpuBackend::arc(),
            &recipe(),
            &dims(),
            &t.weights,
            &LoadOptions { max_seq: 160, prefill_chunk: CHUNK, ..Default::default() },
            Arc::new(Telemetry::new()),
        )
        .unwrap();
        assert!(m.backend.prefix_reuse_any_cut());
        let p: Vec<u32> = LONG.bytes().map(|b| b as u32).collect();
        let steps = [7u32, 8, 9];
        let run = |k: usize| {
            let mut st = m.new_state().unwrap();
            let mut rows = vec![];
            if k == 0 {
                rows.push(logits(m.forward(&mut st, &p, HeadMode::LastLogits, None).unwrap()));
            } else {
                m.forward(&mut st, &p[..k], HeadMode::None, None).unwrap();
                rows.push(logits(m.forward(&mut st, &p[k..], HeadMode::LastLogits, None).unwrap()));
            }
            for &s in &steps {
                rows.push(logits(m.forward(&mut st, &[s], HeadMode::LastLogits, None).unwrap()));
            }
            rows
        };
        let cold = run(0);
        for k in 1..p.len() {
            assert!(run(k) == cold, "{dt:?}: split ({k}, {}) differs from the one-pass prefill", p.len() - k);
        }
    }
}
