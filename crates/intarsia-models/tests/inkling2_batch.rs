//! Inkling-2 batched decode (lane ink-batch, docs/results/2026-11-21-inkling2-batch.md) on the tiny fixture
//! (`golden/inkling2-tiny`): every row of a `Model::forward_batch` step is bitwise the solo decode step of its sequence
//! (logits, softcapped, and tokens), for 2, 5 and 16 sequences at mixed positions (before, at and past the SWA window of 8,
//! the rel extent of 12, the 2-layer attn-res blocks), with sequences leaving and joining the batch, rolled back per
//! sequence, and a failed batched pass resetting every sequence of the batch.
//!
//! `INTARSIA_INKLING2_BACKEND=metal` runs the same tests on Metal (row-exact kernels for the B-row ops, the `_rows`
//! row-table kernels for the stateful ones). Skipped (loudly) without the fixture; fails under `CI=1`.

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
    println!("SKIPPED: inkling2_batch: {why}");
    eprintln!("SKIPPED: inkling2_batch: {why}");
    if std::env::var("CI").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("inkling2_batch was SKIPPED ({why}) but CI is set");
    }
}

fn on_metal() -> bool {
    std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() == Ok("metal")
}

fn backend() -> Arc<dyn intarsia_core::Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        Ok("metal") => intarsia_backend_metal::MetalBackend::arc().unwrap(),
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu or metal"),
    }
}

/// Prefill chunk: one pass on cpu; 16 on Metal (passes of 2..=16 rows take the row-exact kernels).
fn chunk() -> usize {
    if on_metal() {
        16
    } else {
        64
    }
}

fn load_on(dir: &Path, backend: Arc<dyn intarsia_core::Backend>, rows: usize) -> intarsia_core::Result<Model> {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(dir).unwrap();
    let mut m = Model::load(
        backend,
        &recipe,
        &dims,
        &src,
        &LoadOptions { max_seq: 128, prefill_chunk: chunk(), graphs: false, ..Default::default() },
        Arc::new(Telemetry::new()),
    )?;
    m.enable_batch(rows)?;
    m.configure_plan(&Default::default(), None)?;
    Ok(m)
}

/// The test backend (cpu: its default batch cap of 8 lifted to 16, the Metal cap, so the 16-sequence case runs there too).
fn load(dir: &Path, rows: usize) -> Model {
    load_on(dir, Arc::new(Failing { inner: backend(), fail: Arc::new(AtomicBool::new(false)) }), rows).unwrap()
}

/// A deterministic token stream per sequence (ids 6..300, away from eos 5).
fn tok(seq: usize, i: usize) -> u32 {
    (6 + (seq * 37 + i * 11 + (seq * i) % 7) % 290) as u32
}

fn logits(o: StepOut) -> Vec<f32> {
    match o {
        StepOut::Logits(v) => v,
        other => panic!("expected logits, got {other:?}"),
    }
}

/// Prefill sequence `seq` to `len` tokens (in chunks of the model's prefill chunk).
fn prefill(m: &Model, st: &mut SequenceState, seq: usize, len: usize) {
    let ids: Vec<u32> = (0..len).map(|i| tok(seq, i)).collect();
    for c in ids.chunks(chunk()) {
        m.forward(st, c, HeadMode::LastLogits, None).unwrap();
    }
}

/// `n` sequences prefilled to `lens[i]`, twice: (solo states, batched states).
fn pair(m: &Model, lens: &[usize]) -> (Vec<SequenceState>, Vec<SequenceState>) {
    let mk = || -> Vec<SequenceState> {
        lens.iter()
            .enumerate()
            .map(|(s, &l)| {
                let mut st = m.new_state().unwrap();
                prefill(m, &mut st, s, l);
                st
            })
            .collect()
    };
    (mk(), mk())
}

/// One step of the sequences `which`: solo on `solo`, batched on `bat`; asserts the rows are bitwise the solo steps.
fn step(m: &Model, solo: &mut [SequenceState], bat: &mut [SequenceState], which: &[usize], what: &str) {
    let ids: Vec<u32> = which.iter().map(|&s| tok(s, solo[s].len)).collect();
    let want: Vec<Vec<f32>> = which.iter().zip(&ids).map(|(&s, &id)| logits(m.forward(&mut solo[s], &[id], HeadMode::LastLogits, None).unwrap())).collect();
    // the batch's states in `which` order
    let mut refs: Vec<&mut SequenceState> = Vec::new();
    {
        let mut rest: Vec<Option<&mut SequenceState>> = bat.iter_mut().map(Some).collect();
        for &s in which {
            refs.push(rest[s].take().expect("a sequence twice in one batch"));
        }
    }
    let modes = vec![HeadMode::LastLogits; which.len()];
    let got = m.forward_batch(&mut refs, &ids, &modes).unwrap();
    for (k, (g, w)) in got.into_iter().zip(&want).enumerate() {
        let g = logits(g);
        assert!(g == *w, "{what}: row {k} (sequence {}, position {}) differs from its solo step", which[k], solo[which[k]].len - 1);
    }
    for &s in which {
        assert_eq!(bat[s].len, solo[s].len, "{what}: sequence {s} length");
    }
}

/// Batched == solo for 2, 5 and 16 sequences at mixed positions: lengths before the window (8), at it, past the rel extent
/// (12), on and off the 2-layer block edges, and far past the window (ring wrapped several times).
#[test]
fn inkling2_batch_equals_solo_2_5_16() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir, 16);
    let all = [1usize, 7, 8, 9, 12, 13, 3, 30, 17, 64, 5, 41, 2, 24, 50, 11];
    for n in [2usize, 5, 16] {
        let lens = &all[..n];
        let (mut solo, mut bat) = pair(&m, lens);
        let which: Vec<usize> = (0..n).collect();
        for k in 0..6 {
            step(&m, &mut solo, &mut bat, &which, &format!("{n} sequences, step {k}"));
        }
    }
}

/// Argmax rows (device argmax of the batched head) are the solo step's tokens.
#[test]
fn inkling2_batch_argmax_equals_solo() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir, 4);
    let lens = [3usize, 9, 20, 33];
    let (mut solo, mut bat) = pair(&m, &lens);
    for k in 0..4 {
        let ids: Vec<u32> = (0..4).map(|s| tok(s, solo[s].len)).collect();
        let want: Vec<StepOut> = (0..4).map(|s| m.forward(&mut solo[s], &[ids[s]], HeadMode::LastArgmax, None).unwrap()).collect();
        let mut refs: Vec<&mut SequenceState> = bat.iter_mut().collect();
        let got = m.forward_batch(&mut refs, &ids, &[HeadMode::LastArgmax; 4]).unwrap();
        for s in 0..4 {
            assert_eq!(format!("{:?}", got[s]), format!("{:?}", want[s]), "step {k} sequence {s}");
        }
    }
}

/// Sequences leave the batch (one finishes) and come back, in a different row order: every row stays its solo step.
#[test]
fn inkling2_batch_membership_changes() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir, 4);
    let (mut solo, mut bat) = pair(&m, &[4, 10, 15, 27]);
    for k in 0..3 {
        step(&m, &mut solo, &mut bat, &[0, 1, 2], &format!("abc {k}"));
    }
    // b finishes: a and c go on
    for k in 0..3 {
        step(&m, &mut solo, &mut bat, &[0, 2], &format!("ac {k}"));
    }
    // d joins, row order shuffled, then a single row (the solo path) and back
    step(&m, &mut solo, &mut bat, &[3, 2, 0], "dca");
    step(&m, &mut solo, &mut bat, &[2], "c alone");
    step(&m, &mut solo, &mut bat, &[1, 3, 0, 2], "bdac");
}

/// Rollback per sequence: one sequence of the batch is cut back by length (a rejected draft) while the others keep going.
#[test]
fn inkling2_batch_rollback_per_sequence() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir, 3);
    let (mut solo, mut bat) = pair(&m, &[6, 14, 31]);
    for k in 0..4 {
        step(&m, &mut solo, &mut bat, &[0, 1, 2], &format!("before {k}"));
    }
    for (s, cut) in [(1usize, 3usize), (2, 2)] {
        let keep = solo[s].len - cut;
        solo[s].truncate(keep, &*m.backend).unwrap();
        bat[s].truncate(keep, &*m.backend).unwrap();
    }
    for k in 0..4 {
        step(&m, &mut solo, &mut bat, &[0, 1, 2], &format!("after rollback {k}"));
    }
}

use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::registry::KernelImpl;
use intarsia_core::{Op, OpCall};
use std::sync::atomic::{AtomicBool, Ordering};

/// The test backend whose row-table `ConvMix` launches fail while `fail` is set (a failed batched pass).
struct Failing {
    inner: Arc<dyn Backend>,
    fail: Arc<AtomicBool>,
}

impl Backend for Failing {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn info(&self) -> DeviceInfo {
        self.inner.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> intarsia_core::Result<Buffer> {
        self.inner.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> intarsia_core::Result<()> {
        self.inner.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> intarsia_core::Result<()> {
        self.inner.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> intarsia_core::Result<()> {
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.inner.kernels()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> intarsia_core::Result<()> {
        if call.op == Op::ConvMix && call.rows.is_some() && self.fail.load(Ordering::SeqCst) {
            return Err(intarsia_core::Error::invalid("injected failure"));
        }
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.synchronize(s)
    }
    fn supports_row_tables(&self) -> bool {
        self.inner.supports_row_tables()
    }
    fn device_address(&self, buf: &Buffer) -> Option<u64> {
        self.inner.device_address(buf)
    }
    fn batch_row_cap(&self) -> usize {
        self.inner.batch_row_cap().max(16)
    }
    fn avoid_single_row_prefill(&self) -> bool {
        self.inner.avoid_single_row_prefill()
    }
    fn prefix_reuse_any_cut(&self) -> bool {
        self.inner.prefix_reuse_any_cut()
    }
    fn prefix_reuse_min_pass_rows(&self) -> Option<usize> {
        self.inner.prefix_reuse_min_pass_rows()
    }
    fn zero(&self, dst: &Buffer, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.zero(dst, s)
    }
    fn hold(&self, s: StreamId, after: usize) -> intarsia_core::Result<()> {
        self.inner.hold(s, after)
    }
    fn release(&self, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.release(s)
    }
    fn discard(&self, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.discard(s)
    }
    fn wrap_host(&self, mem: Arc<dyn intarsia_core::backend::HostMemory>, offset: usize, len: usize) -> Option<Buffer> {
        self.inner.wrap_host(mem, offset, len)
    }
}

/// A batched pass that fails half-way resets every sequence of the batch (their rings are half-written); each one then
/// prefills and decodes as a fresh sequence.
#[test]
fn inkling2_batch_failed_pass_resets_every_sequence() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let fail = Arc::new(AtomicBool::new(false));
    let m = load_on(&dir, Arc::new(Failing { inner: backend(), fail: fail.clone() }), 3).unwrap();
    let (mut solo, mut bat) = pair(&m, &[5, 12, 22]);
    step(&m, &mut solo, &mut bat, &[0, 1, 2], "before");
    fail.store(true, Ordering::SeqCst);
    let ids = [7u32, 8, 9];
    let mut refs: Vec<&mut SequenceState> = bat.iter_mut().collect();
    assert!(m.forward_batch(&mut refs, &ids, &[HeadMode::LastLogits; 3]).is_err());
    fail.store(false, Ordering::SeqCst);
    for (s, st) in bat.iter().enumerate() {
        assert_eq!(st.len, 0, "sequence {s} is reset");
    }
    // fresh solo references, then batched again from the reset states
    let lens = [5usize, 12, 22];
    let mut solo: Vec<SequenceState> = lens.iter().enumerate().map(|(s, &l)| {
        let mut st = m.new_state().unwrap();
        prefill(&m, &mut st, s, l);
        st
    }).collect();
    for (s, st) in bat.iter_mut().enumerate() {
        prefill(&m, st, s, lens[s]);
    }
    for k in 0..3 {
        step(&m, &mut solo, &mut bat, &[0, 1, 2], &format!("after reset {k}"));
    }
}
