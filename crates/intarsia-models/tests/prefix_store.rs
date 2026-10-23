//! The content-addressed prefix store (`intarsia_core::prefix_store`) on the cpu backend:
//!
//! - store mechanics: keys, page sharing, eviction, pinning, disk persistence, rejection of corrupt and foreign files;
//! - engine byte identity: a request that restores a stored prefix (after another conversation replaced the held state,
//!   from a prewarm, from disk after a restart) gives bit-identical logits and tokens to a cold prefill of the same
//!   prompt, on the hybrid model (GatedDeltaNet recurrent state + attention KV) and an attention-only model, against a
//!   cold run with the store on and one with the store off (the cpu backend is chunk-invariant, so both must match).

mod common;
use intarsia_core::backend::Backend;
use intarsia_core::engine::{ChatOpts, TextCodec};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::prefix_store::{PrefixStore, PrefixStoreConfig};
use intarsia_core::recipe::ModelDims;
use intarsia_core::state::{SequenceState, SlotSpec};
use intarsia_core::{DType, Engine, EngineMeta, GenEvent, GenRequest, LocalEngine, Prewarm, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ------------------------------------------------------------------------------------------------- store mechanics

fn cpu() -> Arc<dyn Backend> {
    intarsia_backend_cpu::CpuBackend::arc()
}

fn specs() -> Vec<SlotSpec> {
    vec![
        SlotSpec::Recurrent { bytes: 64 },
        SlotSpec::AttnKv { n_kv_heads: 2, head_dim: 4, dtype: DType::F32 },
        SlotSpec::Recurrent { bytes: 32 },
        SlotSpec::AttnKv { n_kv_heads: 1, head_dim: 8, dtype: DType::F32 },
    ]
}

/// A state of `tokens` whose every byte is a function of (slot, position, token) and, for recurrent slots, of the
/// whole sequence: what restoring must reproduce exactly.
fn fake_state(b: &dyn Backend, tokens: &[u32], cap: usize) -> SequenceState {
    let mut st = SequenceState::new(b, &specs(), cap).unwrap();
    st.commit(tokens);
    fill(b, &st);
    st
}

fn fill(b: &dyn Backend, st: &SequenceState) {
    let sum: u32 = st.tokens.iter().fold(7u32, |a, &t| a.wrapping_mul(31).wrapping_add(t));
    for (si, s) in st.slots.iter().enumerate() {
        match s {
            intarsia_core::state::StateSlot::AttnKv(a) => {
                let row = a.head_dim * 4;
                for (kv, buf) in [&a.k, &a.v].into_iter().enumerate() {
                    let mut v = vec![0u8; buf.bytes()];
                    for h in 0..a.n_kv_heads {
                        for p in 0..st.len {
                            for j in 0..row {
                                v[(h * a.cap + p) * row + j] =
                                    (st.tokens[p] as usize * 3 + p * 5 + h * 7 + j + si * 11 + kv * 13) as u8;
                            }
                        }
                    }
                    b.upload(buf, 0, &v, 0).unwrap();
                }
            }
            intarsia_core::state::StateSlot::Recurrent(r) => {
                let v: Vec<u8> = (0..r.bytes).map(|j| (sum as usize + j * 3 + si) as u8).collect();
                b.upload(&r.buf, 0, &v, 0).unwrap();
            }
        }
    }
}

/// Every filled byte of `st` (attention rows `0..len`, recurrent slots).
fn dump(b: &dyn Backend, st: &SequenceState) -> Vec<u8> {
    let mut out = Vec::new();
    for s in &st.slots {
        match s {
            intarsia_core::state::StateSlot::AttnKv(a) => {
                let row = a.head_dim * a.dtype.size_bytes();
                for buf in [&a.k, &a.v] {
                    for h in 0..a.n_kv_heads {
                        let mut v = vec![0u8; st.len * row];
                        b.download(buf, h * a.cap * row, &mut v, 0).unwrap();
                        out.extend(v);
                    }
                }
            }
            intarsia_core::state::StateSlot::Recurrent(r) => {
                let mut v = vec![0u8; r.bytes];
                b.download(&r.buf, 0, &mut v, 0).unwrap();
                out.extend(v);
            }
        }
    }
    out
}

fn cfg(page: usize) -> PrefixStoreConfig {
    PrefixStoreConfig { page_tokens: page, min_tokens: 1, ..Default::default() }
}

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("intarsia-ps-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn store_restores_exact_bytes_and_shares_pages() {
    let b = cpu();
    let mut s = PrefixStore::new(cfg(16), &specs(), "test").unwrap();
    let t: Vec<u32> = (0..50).map(|i| (i * 7 % 23) as u32).collect();
    let a = fake_state(&*b, &t[..40], 64);
    let copied = s.insert(&*b, &a, 40, None).unwrap();
    assert!(copied > 0);
    // a longer prefix shares the first two full pages (32 tokens) with the first entry
    let a2 = fake_state(&*b, &t[..50], 64);
    let copied2 = s.insert(&*b, &a2, 50, None).unwrap();
    let page = (2 * 2 * 4 + 2 * 8) * 4; // bytes per token row over the attention slots
    assert_eq!(copied2 as usize, 18 * page + 96, "only the pages past 32 tokens and the recurrent blob are new");
    // re-inserting is a no-op
    assert_eq!(s.insert(&*b, &a2, 50, None).unwrap(), 0);

    let mut prompt = t.clone();
    prompt.extend([1, 2, 3]);
    let hit = s.lookup(&prompt, prompt.len() - 1).unwrap();
    assert_eq!(hit.len, 50);
    let hit40 = s.lookup(&prompt, 45).unwrap();
    assert_eq!(hit40.len, 40);
    // a prompt that diverges inside the first entry finds nothing
    let mut other = t.clone();
    other[10] = 99;
    assert!(s.lookup(&other, 49).is_none());

    let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
    st.commit(&[5, 5, 5]);
    let r = s.restore(&*b, &mut st, hit).unwrap();
    assert_eq!(r.len, 50);
    assert_eq!(st.len, 50);
    assert_eq!(st.clean, 50);
    assert_eq!(st.tokens, t[..50]);
    assert_eq!(dump(&*b, &st), dump(&*b, &a2));
    let snap = s.snapshot();
    assert_eq!((snap.hits, snap.tokens_saved, snap.entries), (1, 50, 2));
}

#[test]
fn store_evicts_lru_and_keeps_pins() {
    let b = cpu();
    let one = |n: usize| -> Vec<u32> { (0..n).map(|i| (i * 13 % 31 + 100) as u32).collect() };
    // each entry of 32 tokens: 32 rows of 128 B + 96 B recurrent = 4192 B; budget for two
    let mut c = cfg(16);
    c.host_bytes = 9_000;
    let mut s = PrefixStore::new(c, &specs(), "test").unwrap();
    let mk = |seed: u32| -> Vec<u32> { one(32).into_iter().map(|t| t + seed).collect() };
    let (p1, p2, p3) = (mk(0), mk(1000), mk(2000));
    s.insert(&*b, &fake_state(&*b, &p1, 64), 32, Some("system")).unwrap();
    s.insert(&*b, &fake_state(&*b, &p2, 64), 32, None).unwrap();
    s.insert(&*b, &fake_state(&*b, &p3, 64), 32, None).unwrap();
    // p2 is the LRU unpinned entry: evicted; the pinned p1 stays although it is older
    assert!(s.lookup(&p1, 32).is_some());
    assert!(s.lookup(&p2, 32).is_none());
    assert!(s.lookup(&p3, 32).is_some());
    let snap = s.snapshot();
    assert_eq!(snap.pinned_entries, 1);
    assert!(snap.host.evictions >= 1);
    assert!(snap.host.bytes <= 9_000);
}

#[test]
fn store_device_tier_demotes_to_host() {
    let b = cpu();
    let mut c = cfg(16);
    c.device_bytes = 7000;
    let mut s = PrefixStore::new(c, &specs(), "test").unwrap();
    let t1: Vec<u32> = (0..32).map(|i| i as u32).collect();
    let t2: Vec<u32> = (0..32).map(|i| 500 + i as u32).collect();
    let a1 = fake_state(&*b, &t1, 64);
    s.insert(&*b, &a1, 32, None).unwrap();
    s.insert(&*b, &fake_state(&*b, &t2, 64), 32, None).unwrap();
    let snap = s.snapshot();
    assert!(snap.device.bytes <= 7000 && snap.host.bytes > 0, "{snap:?}");
    let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
    let h = s.lookup(&t1, 32).unwrap();
    s.restore(&*b, &mut st, h).unwrap();
    assert_eq!(dump(&*b, &st), dump(&*b, &a1));
}

#[test]
fn store_disk_survives_restart_and_rejects_corruption_and_other_builds() {
    let b = cpu();
    let dir = tmpdir("disk");
    let mut c = cfg(16);
    c.disk_dir = Some(dir.clone());
    c.disk_bytes = 1 << 30;
    let t: Vec<u32> = (0..40).map(|i| (i * 3) as u32).collect();
    let u: Vec<u32> = (0..40).map(|i| (i * 5 + 1) as u32).collect();
    let a = fake_state(&*b, &t, 64);
    {
        let mut s = PrefixStore::new(c.clone(), &specs(), "build-1").unwrap();
        s.insert(&*b, &a, 40, None).unwrap();
        s.insert(&*b, &fake_state(&*b, &u, 64), 40, None).unwrap();
        s.flush();
    }
    // the same build after a restart: both entries come back from disk, bit-exact
    {
        let mut s = PrefixStore::new(c.clone(), &specs(), "build-1").unwrap();
        assert_eq!(s.snapshot().entries, 2);
        let h = s.lookup(&t, 40).unwrap();
        let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
        let r = s.restore(&*b, &mut st, h).unwrap();
        assert_eq!(r.tier, Some(intarsia_core::prefix_store::Tier::Disk));
        assert_eq!(dump(&*b, &st), dump(&*b, &a));
        assert_eq!(s.snapshot().disk.hits, 1);
    }
    // another build never sees them
    {
        let mut s = PrefixStore::new(c.clone(), &specs(), "build-2").unwrap();
        assert_eq!(s.snapshot().entries, 0);
        assert!(s.lookup(&t, 40).is_none());
    }
    // a flipped byte in an object file: the restore is rejected (state left reset), the entry dropped
    let sub = dir.join(format!("{:016x}", intarsia_core::prefix_store::build_fingerprint("build-1")));
    let mut corrupted = 0;
    for f in std::fs::read_dir(&sub).unwrap() {
        let p = f.unwrap().path();
        if p.extension().is_some_and(|x| x == "bin") {
            let mut d = std::fs::read(&p).unwrap();
            let n = d.len();
            d[n - 3] ^= 0x40;
            std::fs::write(&p, d).unwrap();
            corrupted += 1;
        }
    }
    assert!(corrupted > 0);
    {
        let mut s = PrefixStore::new(c.clone(), &specs(), "build-1").unwrap();
        let h = s.lookup(&t, 40).unwrap();
        let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
        assert!(s.restore(&*b, &mut st, h).is_err());
        assert_eq!(st.len, 0);
        assert!(s.snapshot().rejected >= 1);
        assert!(s.lookup(&t, 40).is_none(), "the rejected entry is gone");
    }
    // a manifest of another format version is rejected on load
    {
        for f in std::fs::read_dir(&sub).unwrap() {
            let p = f.unwrap().path();
            if p.extension().is_some_and(|x| x == "entry") {
                let mut d = std::fs::read(&p).unwrap();
                d[4] = d[4].wrapping_add(1); // format version
                std::fs::write(&p, d).unwrap();
            }
        }
        let mut s = PrefixStore::new(c, &specs(), "build-1").unwrap();
        assert_eq!(s.snapshot().entries, 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn store_refuses_a_state_of_another_layout() {
    let b = cpu();
    let mut s = PrefixStore::new(cfg(16), &specs(), "test").unwrap();
    let other = vec![SlotSpec::AttnKv { n_kv_heads: 2, head_dim: 4, dtype: DType::F32 }];
    let mut st = SequenceState::new(&*b, &other, 64).unwrap();
    st.commit(&[1, 2, 3]);
    assert!(s.insert(&*b, &st, 3, None).is_err());
}

// ------------------------------------------------------------------------------------------------- engine byte identity

const BOUND_TEXT: &str = "<B>";
const THINK_TEXT: &str = "<T>";

/// `<B>`, `<T>` and a newline are one token each (`bound`, `think`, `nl`), everything else is bytes modulo the
/// vocabulary.
struct TestCodec {
    vocab: u32,
    bound: u32,
    think: u32,
    nl: u32,
}

impl TextCodec for TestCodec {
    fn encode(&self, text: &str, _s: bool) -> intarsia_core::Result<Vec<u32>> {
        Ok(match text {
            BOUND_TEXT => vec![self.bound],
            THINK_TEXT => vec![self.think],
            "\n" => vec![self.nl],
            _ => text.bytes().map(|b| b as u32 % self.vocab).collect(),
        })
    }
    fn decode(&self, ids: &[u32], _s: bool) -> intarsia_core::Result<String> {
        Ok(ids.iter().map(|i| format!("<{i}>")).collect())
    }
    fn apply_chat_template(&self, _m: &[serde_json::Value], _o: &ChatOpts) -> intarsia_core::Result<String> {
        Ok(String::new())
    }
    fn has_chat_template(&self) -> bool {
        false
    }
}

fn hybrid_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn hybrid(chunk: usize) -> Model {
    let dir = hybrid_dir();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&dir).unwrap();
    Model::load(
        cpu(),
        &recipe,
        &dims,
        &src,
        &LoadOptions {
            max_seq: 240,
            prefill_chunk: chunk,
            act_dtype: Some(DType::F32),
            kv_dtype: Some(DType::F32),
            ..Default::default()
        },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

fn dense(chunk: usize) -> Model {
    let t = common::tiny_weights(DType::Bf16, 11);
    Model::load(
        cpu(),
        &common::recipe(),
        &common::dims(),
        &t.weights,
        &LoadOptions { max_seq: 240, prefill_chunk: chunk, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Hybrid,
    Dense,
}

impl Kind {
    fn codec(self) -> TestCodec {
        match self {
            Kind::Hybrid => TestCodec { vocab: 96, bound: 3, think: 4, nl: 5 },
            Kind::Dense => TestCodec { vocab: 300, bound: 290, think: 291, nl: 292 },
        }
    }
    fn vocab(self) -> u32 {
        self.codec().vocab
    }
}

fn engine(kind: Kind, chunk: usize, store: Option<PrefixStoreConfig>) -> LocalEngine {
    let m = match kind {
        Kind::Hybrid => hybrid(chunk),
        Kind::Dense => dense(chunk),
    };
    let store = store.map(|mut c| {
        c.boundary_tokens = vec![BOUND_TEXT.into()];
        c.header_end_tokens = vec!["\n".into()];
        c
    });
    LocalEngine::new(
        m,
        Arc::new(kind.codec()),
        EngineMeta { id: "tiny".into(), prefix_store: store, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap()
}

struct Out {
    ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
    cached: usize,
}

fn run(e: &LocalEngine, prompt: &[u32], n: usize) -> Out {
    let trace = Arc::new(Mutex::new(Vec::new()));
    let req = GenRequest {
        prompt: prompt.to_vec(),
        max_tokens: n,
        ignore_eos: true,
        trace_logits: Some(trace.clone()),
        ..Default::default()
    };
    let (mut ids, mut cached) = (vec![], 0);
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Done { cached_tokens, .. } => cached = cached_tokens,
        GenEvent::Error { message } => panic!("{message}"),
    })
    .unwrap();
    let logits = trace.lock().unwrap().clone();
    Out { ids, logits, cached }
}

fn prewarm(e: &LocalEngine, prompt: &[u32], pin: Option<&str>) -> usize {
    let req = GenRequest {
        prompt: prompt.to_vec(),
        max_tokens: 1,
        prewarm: Some(Prewarm { pin: pin.map(String::from) }),
        ..Default::default()
    };
    let mut cached = usize::MAX;
    let mut toks = 0;
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { .. } => toks += 1,
        GenEvent::Done { cached_tokens, completion_tokens, .. } => {
            cached = cached_tokens;
            assert_eq!(completion_tokens, 0);
        }
        GenEvent::Error { message } => panic!("{message}"),
    })
    .unwrap();
    assert_eq!(toks, 0, "a prewarm generates nothing");
    cached
}

/// Against a cold run with the store off: bitwise for the dense model (its cpu prefill is chunk-invariant); the hybrid's
/// GatedDeltaNet prefill is not chunk-invariant on cpu (`hybrid_cpu_prefill_chunk_sensitivity`), and the store-on
/// schedule cuts chunks at message boundaries, so there it is only reported.
fn off_check(kind: Kind, a: &Out, b: &Out, what: &str) {
    if kind == Kind::Dense {
        same_bits(a, b, what);
    } else {
        let eq = a.ids == b.ids && a.logits == b.logits;
        println!("{what}: bitwise {eq}, ids equal {}", a.ids == b.ids);
    }
}

fn same_bits(a: &Out, b: &Out, what: &str) {
    assert_eq!(a.ids, b.ids, "{what}: generated ids differ");
    assert_eq!(a.logits.len(), b.logits.len(), "{what}");
    for (i, (x, y)) in a.logits.iter().zip(&b.logits).enumerate() {
        let xb: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
        let yb: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
        assert!(xb == yb, "{what}: logits of step {i} differ");
    }
}

/// A ChatML-like conversation: `<B>` starts each message, the generation prompt ends in `<T>`, earlier assistant
/// turns lose their thinking (as the Qwen3.6 template renders them).
struct Chat {
    kind: Kind,
    toks: Vec<u32>,
}

impl Chat {
    fn new(kind: Kind) -> Chat {
        Chat { kind, toks: Vec::new() }
    }
    fn msg(&mut self, seed: u32, n: usize) -> &mut Chat {
        let c = self.kind.codec();
        self.toks.push(c.bound);
        self.toks.push(25); // role
        self.toks.push(c.nl);
        for i in 0..n as u32 {
            // ids 10.. (clear of the boundary ids)
            self.toks.push(10 + (seed * 17 + i * 7) % (self.kind.vocab() - 20));
        }
        self
    }
    /// the prompt of the next request: this conversation + `<B>` assistant `<T>`
    fn prompt(&self) -> Vec<u32> {
        let c = self.kind.codec();
        let mut p = self.toks.clone();
        p.push(c.bound);
        p.push(20);
        p.push(c.nl);
        p.push(c.think);
        p.push(21);
        p
    }
    /// append the assistant's answer (without its thinking) to the history
    fn answer(&mut self, ans: &[u32]) -> &mut Chat {
        let c = self.kind.codec();
        self.toks.push(c.bound);
        self.toks.push(20);
        self.toks.push(c.nl);
        // answer tokens: drop the boundary ids so they do not add cuts the answer did not have
        self.toks.extend(ans.iter().map(|&t| if t == c.bound || t == c.think || t == c.nl { 22 } else { t }));
        self
    }
}

fn store_cfg() -> PrefixStoreConfig {
    PrefixStoreConfig { min_tokens: 4, grid_tokens: 16, page_tokens: 16, ..Default::default() }
}

/// Turn 1 of chat A, then chat B (replaces the held state), then turn 2 of chat A: turn 2 restores A's history from
/// the store, and its output is bitwise the cold output (store on and store off).
fn multi_turn_identity(kind: Kind, chunk: usize) {
    let warm = engine(kind, chunk, Some(store_cfg()));
    let mut a = Chat::new(kind);
    a.msg(1, 23).msg(2, 9);
    let t1 = run(&warm, &a.prompt(), 6);
    let mut b = Chat::new(kind);
    b.msg(7, 30).msg(8, 5);
    let _ = run(&warm, &b.prompt(), 4);
    a.answer(&t1.ids).msg(3, 11);
    let p2 = a.prompt();
    let restored = run(&warm, &p2, 12);
    assert!(restored.cached > 30, "turn 2 restored only {} tokens", restored.cached);
    let cold_on = run(&engine(kind, chunk, Some(store_cfg())), &p2, 12);
    assert_eq!(cold_on.cached, 0);
    let cold_off = run(&engine(kind, chunk, None), &p2, 12);
    same_bits(&restored, &cold_on, "restored vs cold (store on)");
    off_check(kind, &restored, &cold_off, "restored vs cold (store off)");
    // a third turn: restores turn 2's history
    a.answer(&restored.ids).msg(4, 6);
    let p3 = a.prompt();
    let r3 = run(&warm, &p3, 8);
    assert!(r3.cached > restored.cached);
    same_bits(&r3, &run(&engine(kind, chunk, Some(store_cfg())), &p3, 8), "turn 3 restored vs cold (store on)");
    off_check(kind, &r3, &run(&engine(kind, chunk, None), &p3, 8), "turn 3 restored vs cold (store off)");
    let snap = warm.telemetry().prefix_store.unwrap();
    assert!(snap.hits + snap.local_wins >= 2 && snap.tokens_saved > 0, "{snap:?}");
}

#[test]
fn hybrid_multi_turn_restore_is_bitwise_cold() {
    multi_turn_identity(Kind::Hybrid, 8);
    multi_turn_identity(Kind::Hybrid, 64);
}

#[test]
fn dense_multi_turn_restore_is_bitwise_cold() {
    multi_turn_identity(Kind::Dense, 8);
    multi_turn_identity(Kind::Dense, 64);
}

/// The agent / RAG pattern: one long fixed prefix (system prompt, tool schema, documents), many different questions.
/// The prefix is pinned by a prewarm; every question restores it.
fn shared_prefix_identity(kind: Kind) {
    let chunk = 8;
    let warm = engine(kind, chunk, Some(store_cfg()));
    let mut sys = Chat::new(kind);
    sys.msg(40, 60);
    let pinned = sys.toks.clone();
    assert_eq!(prewarm(&warm, &pinned, Some("system")), 0);
    // prewarming again is a hit on the whole prefix
    assert_eq!(prewarm(&warm, &pinned, Some("system")), pinned.len());
    for q in 0..3u32 {
        let mut c = Chat { kind, toks: pinned.clone() };
        c.msg(100 + q, 5 + q as usize * 3);
        let p = c.prompt();
        let r = run(&warm, &p, 6);
        assert!(r.cached >= pinned.len(), "question {q} reused {} of the {}-token prefix", r.cached, pinned.len());
        same_bits(&r, &run(&engine(kind, chunk, Some(store_cfg())), &p, 6), "shared prefix restored vs cold (store on)");
        off_check(kind, &r, &run(&engine(kind, chunk, None), &p, 6), "shared prefix restored vs cold (store off)");
    }
    let snap = warm.telemetry().prefix_store.unwrap();
    assert_eq!(snap.pinned_entries, 1);
}

#[test]
fn hybrid_shared_prefix_prewarm_is_bitwise_cold() {
    shared_prefix_identity(Kind::Hybrid);
}

#[test]
fn dense_shared_prefix_prewarm_is_bitwise_cold() {
    shared_prefix_identity(Kind::Dense);
}

/// A long message with no boundary inside (a pasted document): the grid entries inside it let a question that shares
/// only part of the document reuse that part.
#[test]
fn hybrid_grid_points_inside_a_long_message() {
    let kind = Kind::Hybrid;
    let warm = engine(kind, 8, Some(store_cfg()));
    let mut a = Chat::new(kind);
    a.msg(50, 90);
    let _ = run(&warm, &a.prompt(), 2);
    let mut b = Chat::new(kind);
    b.msg(50, 90);
    // same document, edited near its end
    let n = b.toks.len();
    b.toks[n - 10] = 33;
    let p = b.prompt();
    let r = run(&warm, &p, 6);
    assert!(r.cached >= 64, "reused {}", r.cached);
    same_bits(&r, &run(&engine(kind, 8, Some(store_cfg())), &p, 6), "grid restore vs cold (store on)");
}

/// The store on disk: a new engine of the same build restores from the disk tier, bitwise.
#[test]
fn hybrid_restore_from_disk_after_restart() {
    let kind = Kind::Hybrid;
    let dir = tmpdir("engine-disk");
    let mut c = store_cfg();
    c.disk_dir = Some(dir.clone());
    c.disk_bytes = 1 << 30;
    let mut a = Chat::new(kind);
    a.msg(61, 40).msg(62, 12);
    let p1 = a.prompt();
    {
        let e = engine(kind, 8, Some(c.clone()));
        let _ = run(&e, &p1, 4);
        // let the writer finish (it is asynchronous)
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
    a.msg(63, 7);
    let p2 = a.prompt();
    let e = engine(kind, 8, Some(c));
    let r = run(&e, &p2, 6);
    assert!(r.cached > 40, "reused {}", r.cached);
    let snap = e.telemetry().prefix_store.unwrap();
    assert_eq!(snap.disk.hits, 1, "{snap:?}");
    same_bits(&r, &run(&engine(kind, 8, Some(store_cfg())), &p2, 6), "disk restore vs cold (store on)");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Store off: no prefix store in the telemetry, prewarm refused.
#[test]
fn store_off_is_off() {
    let e = engine(Kind::Dense, 8, None);
    assert!(e.telemetry().prefix_store.is_none());
    let req = GenRequest { prompt: vec![1, 2, 3], max_tokens: 1, prewarm: Some(Prewarm::default()), ..Default::default() };
    let mut err = None;
    let _ = e.generate(req, &mut |ev| {
        if let GenEvent::Error { message } = ev {
            err = Some(message);
        }
    });
    assert!(err.is_some_and(|m| m.contains("prefix store")));
}

// ------------------------------------------------------------------------------------------------- dead-tail skip

/// `Model::set_skip_dead`: non-final prefill chunks skip the last layer's work after its cache writes. Every state byte
/// after the chunks, and the logits of the last chunk, are bitwise what the full pass gives.
fn skip_dead_identity(kind: Kind) {
    use intarsia_core::model::{HeadMode, StepOut};
    let chunk = 8;
    let m = match kind {
        Kind::Hybrid => hybrid(chunk),
        Kind::Dense => dense(chunk),
    };
    let ids: Vec<u32> = (0..45u32).map(|i| 10 + (i * 37) % (kind.vocab() - 20)).collect();
    let mut outs = Vec::new();
    for skip in [false, true] {
        m.set_skip_dead(skip);
        let mut st = m.new_state().unwrap();
        // three non-final chunks, a partial one, then the last token with logits
        for c in [&ids[..8], &ids[8..16], &ids[16..24], &ids[24..44]] {
            assert!(matches!(m.forward(&mut st, c, HeadMode::None, None).unwrap(), StepOut::None));
        }
        let state = dump(&*m.backend, &st);
        let StepOut::Logits(l) = m.forward(&mut st, &ids[44..], HeadMode::LastLogits, None).unwrap() else { panic!() };
        outs.push((state, l.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), dump(&*m.backend, &st)));
    }
    m.set_skip_dead(false);
    assert!(outs[0].0 == outs[1].0, "state after the non-final chunks differs with the dead tail skipped");
    assert!(outs[0].1 == outs[1].1, "logits differ");
    assert!(outs[0].2 == outs[1].2, "final state differs");
}

#[test]
fn hybrid_skip_dead_tail_is_bitwise() {
    skip_dead_identity(Kind::Hybrid);
}

#[test]
fn dense_skip_dead_tail_is_bitwise() {
    skip_dead_identity(Kind::Dense);
}

#[test]
fn hybrid_cpu_prefill_chunk_sensitivity() {
    // a report, not a gate: whether the cpu hybrid's prefill bits depend on the chunking (they did on 2026-10-22)
    use intarsia_core::model::{HeadMode, StepOut};
    let m = hybrid(8);
    let ids: Vec<u32> = (0..40u32).map(|i| 10 + (i * 37) % 70).collect();
    let b = &*m.backend;
    // A: forward 0..24 in chunks of 8, capture, then 24..40
    let mut st = m.new_state().unwrap();
    for c in ids[..24].chunks(8) { m.forward(&mut st, c, HeadMode::None, None).unwrap(); }
    let mut s = PrefixStore::new(cfg(16), m.slot_specs(), "dbg").unwrap();
    s.insert(b, &st, 24, None).unwrap();
    let d24 = dump(b, &st);
    let mut st2 = m.new_state().unwrap();
    let h = s.lookup(&ids, 39).unwrap();
    s.restore(b, &mut st2, h).unwrap();
    assert!(dump(b, &st2) == d24, "a restored state is the captured state");
    let StepOut::Logits(l1) = m.forward(&mut st, &ids[24..], HeadMode::LastLogits, None).unwrap() else { panic!() };
    let StepOut::Logits(l2) = m.forward(&mut st2, &ids[24..], HeadMode::LastLogits, None).unwrap() else { panic!() };
    assert!(l1 == l2, "a restored state continues bitwise");
    // chunk invariance: one pass of 0..40 vs chunks
    let mut st3 = m.new_state().unwrap();
    let StepOut::Logits(l3) = m.forward(&mut st3, &ids[..], HeadMode::LastLogits, None).unwrap() else { panic!() };
    println!("chunked vs model-chunked equal: {}", l1 == l3);
    let mut st4 = m.new_state().unwrap();
    for c in [&ids[..5], &ids[5..24], &ids[24..39]] { m.forward(&mut st4, c, HeadMode::None, None).unwrap(); }
    let StepOut::Logits(l4) = m.forward(&mut st4, &ids[39..], HeadMode::LastLogits, None).unwrap() else { panic!() };
    // the state itself after the same 40 tokens in 8 x 5 rows vs 5 + 19 + 16 rows (no 1-row pass in either)
    let (mut sa, mut sb) = (m.new_state().unwrap(), m.new_state().unwrap());
    for c in ids.chunks(8) { m.forward(&mut sa, c, HeadMode::None, None).unwrap(); }
    for c in [&ids[..5], &ids[5..24], &ids[24..40]] { m.forward(&mut sb, c, HeadMode::None, None).unwrap(); }
    println!("CHUNK-SENSITIVITY cpu hybrid (tiny fixture): state after 40 tokens, 8x5 vs 5+19+16 rows, bitwise equal: {}", dump(b, &sa) == dump(b, &sb));
    println!("CHUNK-SENSITIVITY cpu hybrid: chunks (5, 19, 15, 1) vs (8 x 5) bitwise equal: {}", l3 == l4);
}
