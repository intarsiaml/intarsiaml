//! Inkling-2 packed prefill (lane ink-hip-bpf, docs/results/2026-11-26-inkling2-hip-bpf.md): one `Model::ink_prefill_packed` pass
//! over the next chunks of K sequences leaves every sequence bitwise as its own `forward_prefill` of that chunk does: the head
//! (last-row logits / argmax), every state byte a later pass reads (the conv and SWA rings whole, the full-attention K / V rows),
//! and the decode step after it. Segments of 17 .. 2048 rows, 2 .. 16 segments, fresh and resumed sequences (a segment that starts
//! mid-prompt), heads on some segments only (a non-final chunk has none), and the expert calls split by form (HIP: the
//! tensor-core prefill from `Backend::moe_split_rows` rows, the decode kernels below it).
//!
//! The tiny fixture (`golden/inkling2-tiny`) on the cpu reference, or with `INTARSIA_INKLING2_BACKEND=hip` on one R9700. The real
//! checkpoint at real shapes: `INTARSIA_INK_PACK_REAL=<model dir>` (HIP; no-op without it). Every test here sets
//! `INTARSIA_INK_PACK=1` (the knob is read once per process) and `INTARSIA_INK_PACK_ROWS=4096` (the packed workspace) before it
//! loads a model.

mod common;
use common::ByteCodec;
use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{GumbelHead, HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::registry::KernelImpl;
use intarsia_core::state::{SequenceState, StateSlot};
use intarsia_core::{BatchedEngine, BatchedOptions, Engine, EngineMeta, GenEvent, GenRequest, LocalEngine, OpCall, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

const PACK_ROWS: usize = 4096;

fn tiny_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("INTARSIA_INKLING2_TINY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/inkling2-tiny"));
    dir.join("config.json").exists().then_some(dir)
}

fn skip(why: &str) {
    println!("SKIPPED: inkling2_packed: {why}");
    eprintln!("SKIPPED: inkling2_packed: {why}");
    if std::env::var("CI").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("inkling2_packed was SKIPPED ({why}) but CI is set");
    }
}

fn pack_env() {
    // every test sets the same values, so threads racing here agree
    std::env::set_var("INTARSIA_INK_PACK", "1");
    std::env::set_var("INTARSIA_INK_PACK_ROWS", PACK_ROWS.to_string());
}

/// The test backend: cpu (its batch cap of 8 lifted to 16) or, with `INTARSIA_INKLING2_BACKEND=hip`, one R9700 as it is.
fn backend() -> Arc<dyn Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        Ok("hip") => intarsia_backend_hip::create(0).unwrap(),
        Ok("cpu") | Err(_) => Arc::new(Cap16 { inner: intarsia_backend_cpu::CpuBackend::arc() }),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu or hip"),
    }
}

fn load_on(dir: &Path, be: Arc<dyn Backend>, rows: usize, chunk: usize, max_seq: usize) -> Model {
    pack_env();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(dir).unwrap();
    let opts = LoadOptions { max_seq, prefill_chunk: chunk, graphs: false, ..Default::default() };
    let mut m = Model::load(be, &recipe, &dims, &src, &opts, Arc::new(Telemetry::new())).unwrap();
    if rows > 1 {
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

/// A deterministic token stream per sequence, ids 6 .. 6 + `vocab`.
fn tok(seq: usize, i: usize, vocab: usize) -> u32 {
    (6 + (seq * 7919 + i * 104_729 + (seq * i) % 13) % vocab) as u32
}

/// Every state byte a later pass reads: rings whole, the full-attention K / V rows `0..len` per KV head.
fn state_bytes(be: &dyn Backend, st: &SequenceState) -> Vec<Vec<u8>> {
    be.synchronize(0).unwrap();
    let mut out = Vec::new();
    for s in &st.slots {
        match s {
            StateSlot::Ring(r) => {
                let mut d = vec![0u8; r.buf.bytes()];
                be.download(&r.buf, 0, &mut d, 0).unwrap();
                out.push(d);
            }
            StateSlot::AttnKv(a) => {
                let row = a.head_dim * a.dtype.size_bytes();
                for buf in [&a.k, &a.v] {
                    let mut d = vec![0u8; a.n_kv_heads * a.cap * row];
                    be.download(buf, 0, &mut d, 0).unwrap();
                    for h in 0..a.n_kv_heads {
                        out.push(d[h * a.cap * row..(h * a.cap + st.len) * row].to_vec());
                    }
                }
            }
            StateSlot::Recurrent(r) => {
                let mut d = vec![0u8; r.bytes];
                be.download(&r.buf, 0, &mut d, 0).unwrap();
                out.push(d);
            }
            _ => {}
        }
    }
    out
}

fn bits(o: &StepOut) -> Vec<u32> {
    match o {
        StepOut::Logits(v) => v.iter().map(|x| x.to_bits()).collect(),
        StepOut::Token(t) => vec![*t],
        StepOut::None => vec![],
        other => panic!("unexpected head output {other:?}"),
    }
}

/// The head of segment `i` of a case: a final chunk's (logits, argmax or, with the device sampler, a sampled token) or none
/// (a non-final chunk).
fn mode(m: &Model, i: usize) -> HeadMode {
    match i % 4 {
        0 => HeadMode::LastLogits,
        1 => HeadMode::LastArgmax,
        2 if m.gumbel_ok() => HeadMode::LastGumbel(GumbelHead::new(77 + i as u64, 0.8)),
        2 => HeadMode::LastArgmax,
        _ => HeadMode::None,
    }
}

/// Sequence `i` (tokens `tok(seed + i, ..)`) prefilled solo to `prior[i]` tokens, then its next `lens[i]` tokens: solo
/// (`forward_prefill`) on one set of states, as one packed pass on another; heads, states and one decode step after must be
/// bitwise equal.
fn check(m: &Model, prior: &[usize], lens: &[usize], seed: usize, what: &str) {
    // every segment but the last at a multiple of the backend's row alignment (`Model::ink_pack_align_rows`: 1 on the cpu
    // reference and at real shapes; the tiny fixture's 972-byte rows on a GPU need 4)
    let al = m.ink_pack_align_rows();
    let lens: Vec<usize> = lens.iter().enumerate().map(|(i, &l)| if i + 1 < lens.len() { l.div_ceil(al) * al } else { l }).collect();
    let lens = &lens[..];
    let vocab = m.dims.vocab.min(200_000) - 6;
    let be = m.backend.clone();
    let chunk = m.prefill_chunk();
    let ids = |i: usize, a: usize, b: usize| -> Vec<u32> { (a..b).map(|k| tok(seed + i, k, vocab)).collect() };
    let fresh = |i: usize| -> SequenceState {
        let mut st = m.new_state().unwrap();
        let p = ids(i, 0, prior[i]);
        for c in p.chunks(chunk) {
            m.forward_prefill(&mut st, c, HeadMode::None, None).unwrap();
        }
        st
    };
    let n = lens.len();
    let chunks: Vec<Vec<u32>> = (0..n).map(|i| ids(i, prior[i], prior[i] + lens[i])).collect();
    // the solo side one sequence at a time (head, state bytes, restorable lengths, the decode step after), so a case holds n + 1
    // states, not 2n
    struct Solo {
        head: Vec<u32>,
        bytes: Vec<Vec<u8>>,
        len: usize,
        restorable: Vec<usize>,
        next: Vec<u32>,
    }
    let probes = |st: &SequenceState, i: usize| -> Vec<usize> {
        [0, prior[i], st.len.saturating_sub(1), st.len].iter().map(|&w| st.restorable_len(w)).collect()
    };
    let next_tok = |i: usize| [tok(seed + i, prior[i] + lens[i], vocab)];
    let solo: Vec<Solo> = (0..n)
        .map(|i| {
            let mut st = fresh(i);
            let head = bits(&m.forward_prefill(&mut st, &chunks[i], mode(m, i), None).unwrap());
            let bytes = state_bytes(&*be, &st);
            let (len, restorable) = (st.len, probes(&st, i));
            let next = bits(&m.forward(&mut st, &next_tok(i), HeadMode::LastLogits, None).unwrap());
            Solo { head, bytes, len, restorable, next }
        })
        .collect();
    let mut pk: Vec<SequenceState> = (0..n).map(fresh).collect();
    let before = m.ink_packed_passes();
    let got: Vec<Vec<u32>> = {
        let mut items: Vec<(&mut SequenceState, &[u32], HeadMode)> =
            pk.iter_mut().enumerate().map(|(i, st)| (st, &chunks[i][..], mode(m, i))).collect();
        m.ink_prefill_packed(&mut items).unwrap_or_else(|e| panic!("{what}: packed pass refused: {e}")).iter().map(bits).collect()
    };
    assert_eq!(m.ink_packed_passes(), before + 1, "{what}: one packed pass");
    for i in 0..n {
        let w = &solo[i];
        assert_eq!(got[i], w.head, "{what}: segment {i} ({} rows at {}) head", lens[i], prior[i]);
        assert_eq!(pk[i].len, w.len, "{what}: segment {i} length");
        // what idle-slot prefix reuse may restore (the passes, the ring snapshot)
        assert_eq!(probes(&pk[i], i), w.restorable, "{what}: segment {i} restorable_len");
        let a = state_bytes(&*be, &pk[i]);
        assert_eq!(a.len(), w.bytes.len());
        for (k, (x, y)) in a.iter().zip(&w.bytes).enumerate() {
            if x != y {
                let first = x.iter().zip(y).position(|(p, q)| p != q).unwrap_or(x.len().min(y.len()));
                panic!("{what}: segment {i} ({} rows at {}) state part {k} differs from byte {first} of {}", lens[i], prior[i], x.len());
            }
        }
        // the next decode step
        let g = bits(&m.forward(&mut pk[i], &next_tok(i), HeadMode::LastLogits, None).unwrap());
        assert_eq!(g, w.next, "{what}: segment {i} decode step after the pass");
    }
    eprintln!("{what}: {n} segments, {} rows: packed == solo", lens.iter().sum::<usize>());
}

/// Packs of 2 .. 16 segments of 17 .. 2048 rows on the tiny fixture, fresh and resumed, against the solo prefill of each.
#[test]
fn inkling2_packed_equals_solo_tiny() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load_on(&dir, backend(), 16, 2048, 8192);
    assert!(m.ink_pack_capable(), "packed prefill not capable on this model");
    assert_eq!(m.ink_pack_rows_max(), PACK_ROWS);
    assert_eq!(m.prefill_chunk(), 2048, "the recipe chunk, not the packed workspace");
    assert_eq!(m.pack_min_rows(), 17, "a 16-row chunk takes the row-exact decode numerics solo");
    check(&m, &[0, 0], &[17, 17], 1, "2 x 17");
    check(&m, &[0, 9, 0], &[17, 40, 300], 2, "17 / 40 / 300, one resumed");
    check(&m, &[0, 0, 100], &[2048, 1000, 17], 3, "2048 / 1000 / 17");
    check(&m, &[2048, 5, 0, 30], &[2048, 999, 600, 448], 4, "4096 rows, two resumed");
    check(&m, &[0, 0, 0], &[258, 261, 513], 6, "blend tails");
    let lens: Vec<usize> = (0..16).map(|i| 17 + (i * 37) % 230).collect();
    let prior: Vec<usize> = (0..16).map(|i| [0, 3, 20, 100][i % 4]).collect();
    check(&m, &prior, &lens, 5, "16 mixed");
}

/// A chunk under `pack_min_rows` (or a pack of one) is refused and touches no state.
#[test]
fn inkling2_packed_refuses_what_cannot_pack() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load_on(&dir, backend(), 4, 2048, 4096);
    let ids: Vec<u32> = (0..40).map(|k| tok(9, k, 300)).collect();
    let (mut a, mut b) = (m.new_state().unwrap(), m.new_state().unwrap());
    let mut items: Vec<(&mut SequenceState, &[u32], HeadMode)> = vec![(&mut a, &ids[..16], HeadMode::LastLogits), (&mut b, &ids[..], HeadMode::LastLogits)];
    assert!(m.ink_prefill_packed(&mut items).is_err(), "a 16-row segment");
    let mut one: Vec<(&mut SequenceState, &[u32], HeadMode)> = vec![(&mut b, &ids[..], HeadMode::LastLogits)];
    assert!(m.ink_prefill_packed(&mut one).is_err(), "a pack of one");
    assert_eq!((a.len, b.len), (0, 0));
}

/// The real checkpoint at real shapes (`INTARSIA_INK_PACK_REAL=<model dir>`, HIP): the experts' two forms (segments at and
/// around `Backend::moe_split_rows`), 2048-row chunks, 16 segments, resumed sequences.
#[test]
fn inkling2_packed_equals_solo_real() {
    let Some(dir) = std::env::var_os("INTARSIA_INK_PACK_REAL").map(PathBuf::from) else { return };
    let m = load_on(&dir, backend(), 16, 2048, 4224);
    assert!(m.ink_pack_capable(), "packed prefill not capable on this model");
    let split = m.backend.moe_split_rows(128, 8, m.dims.hidden, 768);
    let cap = m.new_state().unwrap().cap;
    eprintln!("inkling2_packed_equals_solo_real: moe_split_rows = {split:?}, pack_rows_max = {}, kv cap {cap}", m.ink_pack_rows_max());
    check(&m, &[0, 0], &[17, 2048], 1, "17 + 2048");
    check(&m, &[0, 0, 0, 0], &[255, 256, 257, 17], 2, "around the split");
    // the blend's 256-row chunks: tails of 2 and 5 rows, and a 4096-row pass
    check(&m, &[0, 0, 0], &[258, 261, 513], 6, "blend tails");
    check(&m, &[0, 0], &[2048, 2048], 7, "2 x 2048");
    // (the plan's KV capacity per sequence is ~2.7k tokens here: every case stays within it)
    check(&m, &[0, 1024, 100, 0, 7, 0], &[2048, 1024, 600, 300, 40, 17], 3, "6 mixed, 4029 rows");
    let lens: Vec<usize> = (0..16).map(|i| 17 + (i * 97) % 400).collect();
    let prior: Vec<usize> = (0..16).map(|i| [0, 49, 0, 1500][i % 4]).collect();
    check(&m, &prior, &lens, 4, "16 mixed");
    check(&m, &[0, 0], &[1024, 1024], 5, "2 x 1024");
}

/// The rows `lo .. lo + len` along the axis where `packed` has `t` and `solo` has `len` (the solo pass's tensor shape).
fn seg_slice(solo: &[usize], packed: &[usize], v: &[f32], t: usize, lo: usize, len: usize) -> Option<Vec<f32>> {
    if solo.len() != packed.len() {
        return None;
    }
    let ax = (0..packed.len()).find(|&a| packed[a] == t && solo[a] == len && (0..packed.len()).all(|b| b == a || packed[b] == solo[b]))?;
    let outer: usize = packed[..ax].iter().product();
    let inner: usize = packed[ax + 1..].iter().product();
    let mut out = Vec::with_capacity(outer * len * inner);
    for o in 0..outer {
        let base = (o * t + lo) * inner;
        out.extend_from_slice(&v[base..base + len * inner]);
    }
    Some(out)
}

/// Every oracle-named intermediate of each segment of a packed pass (`Model::ink_dump_packed`) against its solo pass
/// (`Model::ink_dump`), bitwise; on a difference it names the first visit's differing tensors (which op differs).
fn dump_diff(m: &Model, prior: &[usize], lens: &[usize], seed: usize, what: &str) {
    let vocab = m.dims.vocab.min(200_000) - 6;
    let chunk = m.prefill_chunk();
    let ids = |i: usize, a: usize, b: usize| -> Vec<u32> { (a..b).map(|k| tok(seed + i, k, vocab)).collect() };
    let fresh = |i: usize| -> SequenceState {
        let mut st = m.new_state().unwrap();
        for c in ids(i, 0, prior[i]).chunks(chunk) {
            m.forward_prefill(&mut st, c, HeadMode::None, None).unwrap();
        }
        st
    };
    let n = lens.len();
    let chunks: Vec<Vec<u32>> = (0..n).map(|i| ids(i, prior[i], prior[i] + lens[i])).collect();
    let solo: Vec<intarsia_core::model::InkTensors> = (0..n).map(|i| m.ink_dump(&mut fresh(i), &chunks[i], None).unwrap()).collect();
    let mut pk: Vec<SequenceState> = (0..n).map(fresh).collect();
    let (ranges, dump) = {
        let mut items: Vec<(&mut SequenceState, &[u32], HeadMode)> =
            pk.iter_mut().enumerate().map(|(i, st)| (st, &chunks[i][..], HeadMode::None)).collect();
        m.ink_dump_packed(&mut items).unwrap_or_else(|e| panic!("{what}: packed dump refused: {e}"))
    };
    let t: usize = lens.iter().sum();
    let mut diffs: Vec<String> = Vec::new();
    let mut checked = 0;
    for i in 0..n {
        let (lo, len) = ranges[i];
        for (k, (ss, sv)) in &solo[i] {
            let Some((ps, pv)) = dump.get(k) else { continue };
            let Some(got) = seg_slice(ss, ps, pv, t, lo, len) else {
                diffs.push(format!("{k}: solo {ss:?} / packed {ps:?} not comparable"));
                continue;
            };
            checked += 1;
            if let Some(j) = got.iter().zip(sv).position(|(a, b)| a.to_bits() != b.to_bits()) {
                let nd = got.iter().zip(sv).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                diffs.push(format!("{k} (segment {i}: {len} rows at {}): {nd} of {} values differ, first at {j}: {} vs {}", prior[i], sv.len(), got[j], sv[j]));
            }
        }
    }
    diffs.sort();
    for d in diffs.iter().take(40) {
        eprintln!("{what}: DIFF {d}");
    }
    eprintln!("{what}: {checked} segment tensors compared, {} differ", diffs.len());
    assert!(diffs.is_empty(), "{what}: packed intermediates differ from solo ({} tensors)", diffs.len());
}

/// The per-op diff on the tiny fixture (and, with `INTARSIA_INK_PACK_REAL`, the real checkpoint): every intermediate of each
/// segment bitwise its solo pass's.
#[test]
fn inkling2_packed_intermediates_equal_solo() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load_on(&dir, backend(), 16, 2048, 8192);
    let al = m.ink_pack_align_rows();
    dump_diff(&m, &[0, 0], &[20, 20], 1, "tiny 2 x 20");
    dump_diff(&m, &[0, 7], &[40usize.div_ceil(al) * al, 300], 2, "tiny 40 + 300, one resumed");
    drop(m);
    if let Some(real) = std::env::var_os("INTARSIA_INK_PACK_REAL").map(PathBuf::from) {
        let m = load_on(&real, backend(), 16, 2048, 4224);
        dump_diff(&m, &[0, 0], &[17, 300], 3, "real 17 + 300");
        dump_diff(&m, &[0, 9], &[257, 600], 4, "real 257 + 600, one resumed");
    }
}

fn meta() -> EngineMeta {
    EngineMeta { id: "inkling2-tiny".into(), eos_ids: vec![], ..Default::default() }
}

fn collect(rx: &Receiver<GenEvent>) -> (Vec<u32>, usize) {
    let mut ids = vec![];
    loop {
        match rx.recv_timeout(Duration::from_secs(300)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { completion_tokens, .. } => return (ids, completion_tokens),
            GenEvent::Error { message } => panic!("engine error: {message}"),
        }
    }
}

fn requests(n: usize) -> Vec<GenRequest> {
    let lens = [20usize, 63, 17, 45, 90, 33, 120, 18, 70];
    (0..n)
        .map(|i| {
            let mut r = GenRequest {
                prompt: (0..lens[i % lens.len()]).map(|k| tok(i, k, 290)).collect(),
                max_tokens: 24 + (i % 3) * 5,
                ignore_eos: true,
                ..Default::default()
            };
            if i % 4 == 3 {
                r.temperature = 0.9;
                r.top_k = 20;
                r.seed = Some(11 + i as u64);
            }
            r
        })
        .collect()
}

/// The batching engine (4 slots, 9 requests at once) with packed prefill serves every request token for token as the
/// one-sequence engine, and packed passes did run.
#[test]
fn inkling2_packed_engine_equals_solo_engine() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let n = 9;
    let solo = LocalEngine::new(load_on(&dir, backend(), 1, 64, 256), Arc::new(ByteCodec), meta(), Arc::new(Telemetry::new())).unwrap();
    let want: Vec<(Vec<u32>, usize)> = requests(n)
        .into_iter()
        .map(|r| {
            let flush = GenRequest { prompt: vec![95, 94, 93], max_tokens: 1, ignore_eos: true, ..Default::default() };
            collect(&solo.submit(flush).unwrap());
            collect(&solo.submit(r).unwrap())
        })
        .collect();
    let mut o = BatchedOptions::new(4, 256);
    o.prefix_reuse = false;
    o.pack_prefill = Some(true);
    let model = load_on(&dir, backend(), 4, 64, 256);
    let eng = BatchedEngine::new(model, Arc::new(ByteCodec), meta(), Arc::new(Telemetry::new()), o).unwrap();
    let rxs: Vec<_> = requests(n).into_iter().map(|r| eng.submit(r).unwrap()).collect();
    for (i, (rx, w)) in rxs.iter().zip(&want).enumerate() {
        assert_eq!(&collect(rx), w, "request {i}");
    }
    let mt = eng.metrics();
    assert!(mt.pack_passes > 0, "no packed prefill pass ran: {} passes, {} chunks", mt.pack_passes, mt.packed_chunks);
}

/// The cpu backend with the batch cap lifted to 16 (everything else forwarded, as `inkling2_batch.rs`'s wrapper does).
struct Cap16 {
    inner: Arc<dyn Backend>,
}

impl Backend for Cap16 {
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
    fn supports_topk(&self) -> bool {
        self.inner.supports_topk()
    }
    #[allow(clippy::too_many_arguments)]
    fn topk_rows(
        &self,
        logits: &intarsia_core::TensorRef,
        k: usize,
        ids: &intarsia_core::TensorRef,
        vals: &intarsia_core::TensorRef,
        nan: &intarsia_core::TensorRef,
        scratch: &intarsia_core::TensorRef,
        s: StreamId,
    ) -> intarsia_core::Result<()> {
        self.inner.topk_rows(logits, k, ids, vals, nan, scratch, s)
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
    fn pack_min_rows(&self) -> usize {
        self.inner.pack_min_rows()
    }
    fn pack_prefill_default(&self) -> bool {
        self.inner.pack_prefill_default()
    }
    fn moe_split_rows(&self, e: usize, k: usize, h: usize, i: usize) -> Option<usize> {
        self.inner.moe_split_rows(e, k, h, i)
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
