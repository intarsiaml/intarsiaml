//! The content-addressed prefix store (`intarsia_core::prefix_store`) on the cpu backend:
//!
//! - store mechanics: keys, page sharing, eviction, pinning, disk persistence, rejection of corrupt and foreign files;
//! - engine byte identity: a request that restores a stored prefix (after another conversation replaced the held state,
//!   from a prewarm, from disk after a restart) gives bit-identical logits and tokens to a cold prefill of the same
//!   prompt, on the hybrid model (GatedDeltaNet recurrent state + attention KV) and an attention-only model, against a
//!   cold run with the store on and one with the store off (the cpu backend is chunk-invariant, so both must match);
//! - the batched engine (`--max-batch 2`) with the store: every reply bitwise the single-sequence engine's with the store on, with
//!   several multi-turn conversations in flight on two slots (restores, captures and settles interleaved with the other slot's
//!   prefill and decode).

mod common;
use intarsia_core::backend::Backend;
use intarsia_core::engine::{ChatOpts, TextCodec};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::prefix_store::{PrefixStore, PrefixStoreConfig};
use intarsia_core::recipe::ModelDims;
use intarsia_core::state::{SequenceState, SlotSpec};
use intarsia_core::{BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, GenEvent, GenRequest, LocalEngine, Prewarm, Telemetry};
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
            _ => unreachable!("the prefix store refuses DeepSeek-V4.1 ring / shared slots"),
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
            _ => unreachable!("the prefix store refuses DeepSeek-V4.1 ring / shared slots"),
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
    let copied = s.insert(&*b, &a, 40, None, &[16, 32, 40]).unwrap();
    assert!(copied > 0);
    // a longer prefix shares the first two full pages (32 tokens) with the first entry
    let a2 = fake_state(&*b, &t[..50], 64);
    let copied2 = s.insert(&*b, &a2, 50, None, &[16, 32, 50]).unwrap();
    let page = (2 * 2 * 4 + 2 * 8) * 4; // bytes per token row over the attention slots
    assert_eq!(copied2 as usize, 18 * page + 96, "only the pages past 32 tokens and the recurrent blob are new");
    // re-inserting is a no-op
    assert_eq!(s.insert(&*b, &a2, 50, None, &[16, 32, 50]).unwrap(), 0);

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
    s.insert(&*b, &fake_state(&*b, &p1, 64), 32, Some("system"), &[]).unwrap();
    s.insert(&*b, &fake_state(&*b, &p2, 64), 32, None, &[]).unwrap();
    s.insert(&*b, &fake_state(&*b, &p3, 64), 32, None, &[]).unwrap();
    // p2 is the LRU unpinned entry: evicted; the pinned p1 stays although it is older
    assert!(s.lookup(&p1, 32).is_some());
    assert!(s.lookup(&p2, 32).is_none());
    assert!(s.lookup(&p3, 32).is_some());
    let snap = s.snapshot();
    assert!(snap.pinned_entries >= 1, "{snap:?}");
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
    s.insert(&*b, &a1, 32, None, &[]).unwrap();
    s.insert(&*b, &fake_state(&*b, &t2, 64), 32, None, &[]).unwrap();
    let snap = s.snapshot();
    assert!(snap.device.bytes <= 7000 && snap.host.bytes > 0, "{snap:?}");
    let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
    let h = s.lookup(&t1, 32).unwrap();
    s.restore(&*b, &mut st, h).unwrap();
    assert_eq!(dump(&*b, &st), dump(&*b, &a1));
}

/// Captures made while deferring stay on the device (past the device budget) until `settle`; an entry restored before
/// then still comes back bit-exact, and the settled ones reach disk with a checksum the reloading store accepts (the
/// header is built by the writer thread).
#[test]
fn store_deferred_captures_restore_unsettled_then_settle_to_host_and_disk() {
    let b = cpu();
    let dir = tmpdir("defer");
    let mut c = cfg(16);
    c.device_bytes = 1; // everything leaves the device at settle
    c.disk_dir = Some(dir.clone());
    c.disk_bytes = 1 << 30;
    let t1: Vec<u32> = (0..32).map(|i| i as u32).collect();
    let t2: Vec<u32> = (0..32).map(|i| 500 + i as u32).collect();
    let a1 = fake_state(&*b, &t1, 64);
    let a2 = fake_state(&*b, &t2, 64);
    {
        let mut s = PrefixStore::new(c.clone(), &specs(), "defer").unwrap();
        s.set_defer(true);
        s.insert(&*b, &a1, 32, None, &[]).unwrap();
        s.insert(&*b, &a2, 32, None, &[]).unwrap();
        let snap = s.snapshot();
        assert!(s.deferring());
        assert!(snap.device.bytes > 1 && snap.host.bytes == 0, "{snap:?}");
        // an entry still on the device restores
        let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
        let h = s.lookup(&t1, 32).unwrap();
        s.restore(&*b, &mut st, h).unwrap();
        assert_eq!(dump(&*b, &st), dump(&*b, &a1));
        s.settle(&*b, "test");
        assert!(!s.deferring());
        let snap = s.snapshot();
        assert!(snap.device.bytes == 0 && snap.host.bytes > 0, "{snap:?}");
        let h = s.lookup(&t2, 32).unwrap();
        s.restore(&*b, &mut st, h).unwrap();
        assert_eq!(dump(&*b, &st), dump(&*b, &a2));
        s.flush();
    }
    // the demoted object was written with a valid header: a new store on the same directory reads it back
    let mut s = PrefixStore::new(c, &specs(), "defer").unwrap();
    let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
    let mut from_disk = 0;
    for (t, a) in [(&t1, &a1), (&t2, &a2)] {
        if let Some(h) = s.lookup(t, 32) {
            let r = s.restore(&*b, &mut st, h).unwrap();
            assert_eq!(dump(&*b, &st), dump(&*b, a));
            from_disk += (r.tier == Some(intarsia_core::prefix_store::Tier::Disk)) as usize;
        }
    }
    assert!(from_disk > 0, "nothing came back from disk");
    assert_eq!(s.snapshot().rejected, 0);
}

/// Many requests through a host tier that holds about two entries: every settle evicts, so the demotes take the evicted
/// buffers over (stale bytes in them). Whatever the store still finds must come back bit-exact, with and without the disk
/// tier (where an evicted object's buffer may still be held by the writer), and the host tier stays within its budget.
#[test]
fn store_recycled_host_buffers_restore_exact_bytes_under_eviction() {
    let b = cpu();
    for disk in [false, true] {
        let dir = tmpdir(if disk { "recycle-disk" } else { "recycle" });
        let mut c = cfg(16);
        c.device_bytes = 1;
        c.host_bytes = 9_000;
        if disk {
            c.disk_dir = Some(dir.clone());
            c.disk_bytes = 1 << 30;
        }
        let mut s = PrefixStore::new(c, &specs(), "recycle").unwrap();
        let mut kept = Vec::new();
        for i in 0..8u32 {
            let t: Vec<u32> = (0..32).map(|j| 1000 * i + j).collect();
            let a = fake_state(&*b, &t, 64);
            s.set_defer(true);
            s.insert(&*b, &a, 32, None, &[]).unwrap();
            s.settle(&*b, "test");
            // with the disk tier an object whose write is still queued cannot go yet (its buffer is the writer's): the
            // tier is over budget until the writer confirms it, and within it at the next settle after that
            if !disk {
                assert!(s.snapshot().host.bytes <= 9_000, "{:?}", s.snapshot());
            }
            kept.push((t, a));
        }
        s.flush();
        s.settle(&*b, "test");
        assert!(s.snapshot().host.bytes <= 9_000, "disk {disk}: {:?}", s.snapshot());
        let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
        let mut found = 0;
        for (t, a) in &kept {
            if let Some(h) = s.lookup(t, 32) {
                s.restore(&*b, &mut st, h).unwrap();
                assert_eq!(dump(&*b, &st), dump(&*b, a), "disk {disk}");
                found += 1;
            }
        }
        assert!(found >= 2, "disk {disk}: only {found} entries found");
        assert!(s.snapshot().host.evictions >= 1);
    }
}

/// The captures of one request share a device slab that later requests reuse: restoring every capture before and after the
/// settle, over several requests, gives each one's own bytes.
#[test]
fn store_deferred_captures_share_a_reused_slab_without_mixing_bytes() {
    let b = cpu();
    let mut c = cfg(16);
    c.device_bytes = 0; // no device tier: the captures are carved from slabs
    let mut s = PrefixStore::new(c, &specs(), "slab").unwrap();
    let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
    for round in 0..3u32 {
        let mut made = Vec::new();
        s.set_defer(true);
        for i in 0..3u32 {
            let t: Vec<u32> = (0..32).map(|j| 10_000 * round + 1000 * i + j).collect();
            let a = fake_state(&*b, &t, 64);
            s.insert(&*b, &a, 32, None, &[]).unwrap();
            made.push((t, a));
        }
        for phase in ["on the device", "settled"] {
            for (t, a) in &made {
                let h = s.lookup(t, 32).unwrap();
                s.restore(&*b, &mut st, h).unwrap();
                assert_eq!(dump(&*b, &st), dump(&*b, a), "round {round}, {phase}");
            }
            s.settle(&*b, "test");
        }
    }
}

/// The batched engine's pattern (two requests in flight share one deferral): A and B capture interleaved, A's `Done` settles
/// while B is still prefilling (which also settles B's captures so far and recycles their slabs), then B re-arms the deferral
/// and captures more, carved from a recycled slab. Nothing B captured before the settle may be overwritten by what it captures
/// after: every entry restores exactly, before and after the final settle.
#[test]
fn store_two_requests_share_the_deferral_and_recycled_slabs() {
    let b = cpu();
    let mut c = cfg(16);
    c.device_bytes = 0; // no device tier: the captures are carved from slabs
    let mut s = PrefixStore::new(c, &specs(), "two").unwrap();
    let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
    let mut made = Vec::new();
    let mut capture = |s: &mut PrefixStore, tag: u32| {
        s.set_defer(true);
        let t: Vec<u32> = (0..32).map(|j| tag * 1000 + j).collect();
        let a = fake_state(&*b, &t, 64);
        s.insert(&*b, &a, 32, None, &[]).unwrap();
        (t, a)
    };
    for round in 0..3u32 {
        let base = 100 * round;
        made.push(capture(&mut s, base + 1)); // A
        made.push(capture(&mut s, base + 2)); // B
        made.push(capture(&mut s, base + 3)); // A
        s.settle(&*b, "after A's done"); // B still in flight
        made.push(capture(&mut s, base + 4)); // B, from a recycled slab
        made.push(capture(&mut s, base + 5)); // B
        for phase in ["B's captures on the device", "settled"] {
            for (i, (t, a)) in made.iter().enumerate() {
                let h = s.lookup(t, 32).unwrap_or_else(|| panic!("round {round}: entry {i} lost"));
                s.restore(&*b, &mut st, h).unwrap();
                assert_eq!(dump(&*b, &st), dump(&*b, a), "round {round}, entry {i}, {phase}");
            }
            if phase.starts_with("B") {
                s.settle(&*b, "after B's done");
            }
        }
    }
}

/// Entries that share their first page, captured in one request and so tight a host tier that demoting one object evicts
/// another entry (one holding the shared page too) while the shared page is still being demoted: the references stay
/// consistent whatever the budget (no object outlives the entries that need it, none is dropped while one does), and what
/// is found restores exactly.
#[test]
fn store_shared_pages_keep_their_references_when_demotes_evict_entries() {
    let b = cpu();
    for budget in (2_000..9_000).step_by(100) {
        let mut c = cfg(16);
        c.device_bytes = 0;
        c.host_bytes = budget;
        let mut s = PrefixStore::new(c, &specs(), "refs").unwrap();
        let mut made = Vec::new();
        s.set_defer(true);
        for i in 0..3u32 {
            // the first 16 tokens (page 1) are the same for every entry, the second page and the recurrent state are not
            let t: Vec<u32> = (0..32).map(|j| if j < 16 { 100 + j } else { 5000 + 1000 * i + j }).collect();
            let a = fake_state(&*b, &t, 64);
            s.insert(&*b, &a, 32, None, &[16, 32]).unwrap();
            made.push((t, a));
        }
        s.settle(&*b, "test");
        let snap = s.snapshot();
        let (entries, objects) = (snap.entries as u64, snap.host.objects as u64);
        // per entry its own page and recurrent blob, plus the shared page while any entry lives
        assert_eq!(objects, 2 * entries + (entries > 0) as u64, "budget {budget}: {snap:?}");
        assert!(snap.host.bytes <= budget as u64, "budget {budget}: {snap:?}");
        let mut st = SequenceState::new(&*b, &specs(), 64).unwrap();
        for (t, a) in &made {
            if let Some(h) = s.lookup(t, 32) {
                s.restore(&*b, &mut st, h).unwrap();
                assert_eq!(dump(&*b, &st), dump(&*b, a), "budget {budget}");
            }
        }
    }
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
        s.insert(&*b, &a, 40, None, &[]).unwrap();
        s.insert(&*b, &fake_state(&*b, &u, 64), 40, None, &[]).unwrap();
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
    assert!(s.insert(&*b, &st, 3, None, &[]).is_err());
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
            "assistant" => vec![20],
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
#[derive(Clone)]
struct Chat {
    kind: Kind,
    toks: Vec<u32>,
    /// the generation prompt opens a think block (Qwen3.6's `<think>\n`); false: it ends at the assistant header
    /// (Qwen3's template with thinking on), so the prompt's last header is its end
    open_think: bool,
}

impl Chat {
    fn new(kind: Kind) -> Chat {
        Chat { kind, toks: Vec::new(), open_think: true }
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
        if self.open_think {
            p.push(c.think);
            p.push(21);
        }
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
    multi_turn_identity_cfg(kind, chunk, store_cfg());
}

fn multi_turn_identity_cfg(kind: Kind, chunk: usize, cfg: PrefixStoreConfig) {
    multi_turn_identity_with(kind, chunk, cfg, true)
}

fn multi_turn_identity_with(kind: Kind, chunk: usize, cfg: PrefixStoreConfig, open_think: bool) {
    let store_cfg = || cfg.clone();
    let warm = engine(kind, chunk, Some(store_cfg()));
    let mut a = Chat::new(kind);
    a.open_think = open_think;
    a.msg(1, 23).msg(2, 9);
    let t1 = run(&warm, &a.prompt(), 6);
    let mut b = Chat::new(kind);
    b.open_think = open_think;
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

/// `roles=assistant`: cuts only after assistant headers (and message 2's); turn 2 still restores turn 1's prompt.
#[test]
fn hybrid_multi_turn_assistant_cuts_only() {
    let mut c = store_cfg();
    c.cut_roles = vec!["assistant".into()];
    multi_turn_identity_cfg(Kind::Hybrid, 8, c.clone());
    multi_turn_identity_cfg(Kind::Dense, 64, c);
}

#[test]
fn dense_multi_turn_restore_is_bitwise_cold() {
    multi_turn_identity(Kind::Dense, 8);
    multi_turn_identity(Kind::Dense, 64);
}

/// Qwen3's template (thinking on) ends the prompt at the assistant header: that header is the prompt's end, which the
/// final (logits) pass computes, and the next turn must still restore it (Metal 8B gate, 2026-10-23: it did not).
#[test]
fn multi_turn_prompt_ending_at_the_header() {
    for (kind, chunk) in [(Kind::Dense, 8), (Kind::Dense, 64), (Kind::Hybrid, 8), (Kind::Hybrid, 64)] {
        multi_turn_identity_with(kind, chunk, store_cfg(), false);
    }
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
        let mut c = Chat { kind, toks: pinned.clone(), open_think: true };
        c.msg(100 + q, 5 + q as usize * 3);
        let p = c.prompt();
        let r = run(&warm, &p, 6);
        assert!(r.cached >= pinned.len(), "question {q} reused {} of the {}-token prefix", r.cached, pinned.len());
        same_bits(&r, &run(&engine(kind, chunk, Some(store_cfg())), &p, 6), "shared prefix restored vs cold (store on)");
        off_check(kind, &r, &run(&engine(kind, chunk, None), &p, 6), "shared prefix restored vs cold (store off)");
    }
    let snap = warm.telemetry().prefix_store.unwrap();
    assert!(snap.pinned_entries >= 1, "{snap:?}");
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

/// A prewarm of a half-typed message, the finished turn, another chat, then the next turn: the next turn restores the
/// finished turn's entry, whose KV pages overlap rows the prewarm wrote in a chunk that ended elsewhere (mid-message). On
/// the chunk-sensitive hybrid those rows differ, so a page must not be shared across the two (page keys carry the chunk
/// end). Restored must equal cold (store on).
fn partial_prewarm_then_turns(kind: Kind) {
    let chunk = 16;
    let mut c = store_cfg();
    c.page_tokens = 16;
    let warm = engine(kind, chunk, Some(c.clone()));
    let mut a = Chat::new(kind);
    a.msg(71, 37).msg(72, 9).answer(&[30, 31, 32, 33, 34, 35]);
    let mut typing = a.clone();
    typing.msg(73, 40);
    let n = typing.toks.len();
    typing.toks.truncate(n - 19); // half typed: the chunk ends mid-message
    prewarm(&warm, &typing.toks, None);
    a.msg(73, 40);
    let p2 = a.prompt();
    let r2 = run(&warm, &p2, 5);
    assert!(r2.cached > 50, "turn 2 reused {}", r2.cached);
    let mut b = Chat::new(kind);
    b.msg(9, 33);
    let _ = run(&warm, &b.prompt(), 3);
    a.answer(&r2.ids).msg(74, 12);
    let p3 = a.prompt();
    let r3 = run(&warm, &p3, 6);
    assert!(r3.cached >= p2.len() - 2, "turn 3 reused {} of {}", r3.cached, p2.len());
    same_bits(&r3, &run(&engine(kind, chunk, Some(c)), &p3, 6), "turn after a partial prewarm: restored vs cold (store on)");
}

#[test]
fn hybrid_partial_prewarm_then_turns_is_bitwise() {
    partial_prewarm_then_turns(Kind::Hybrid);
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
    s.insert(b, &st, 24, None, &[]).unwrap();
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

// ------------------------------------------------------------------------------------------- batched engine (slice 1)

/// The test model for the batched engine (batching enabled for `rows` slots; the hybrid with the cpu opt-in) and its
/// single-sequence twin (`rows` 1): both planned the same way.
fn planned(kind: Kind, chunk: usize, rows: usize) -> Model {
    let mut m = match kind {
        Kind::Hybrid => hybrid(chunk),
        Kind::Dense => dense(chunk),
    };
    if rows > 1 {
        if kind == Kind::Hybrid {
            m.allow_recurrent_batch().unwrap();
        }
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

fn store_meta(cfg: Option<PrefixStoreConfig>) -> EngineMeta {
    let store = cfg.map(|mut c| {
        c.boundary_tokens = vec![BOUND_TEXT.into()];
        c.header_end_tokens = vec!["\n".into()];
        c
    });
    EngineMeta { id: "tiny".into(), prefix_store: store, ..Default::default() }
}

/// A batched engine of 2 slots with the store; `busy_chunk` below the model's chunk, so a store request that took the
/// scheduler's busy chunk instead of its own schedule would change its bits.
fn batched_store(kind: Kind, chunk: usize, cfg: PrefixStoreConfig, reuse: bool) -> BatchedEngine {
    batched_store_opts(kind, chunk, Some(cfg), |o| o.prefix_reuse = reuse).unwrap()
}

fn batched_store_opts(
    kind: Kind,
    chunk: usize,
    cfg: Option<PrefixStoreConfig>,
    f: impl FnOnce(&mut BatchedOptions),
) -> intarsia_core::Result<BatchedEngine> {
    let mut o = BatchedOptions::new(2, 64);
    o.busy_chunk = (chunk / 2).max(1);
    f(&mut o);
    BatchedEngine::new(planned(kind, chunk, 2), Arc::new(kind.codec()), store_meta(cfg), Arc::new(Telemetry::new()), o)
}

fn traced(prompt: &[u32], n: usize) -> (GenRequest, Arc<Mutex<Vec<Vec<f32>>>>) {
    let trace = Arc::new(Mutex::new(Vec::new()));
    let req = GenRequest {
        prompt: prompt.to_vec(),
        max_tokens: n,
        ignore_eos: true,
        trace_logits: Some(trace.clone()),
        ..Default::default()
    };
    (req, trace)
}

/// Submit every request at once (several in flight on the slots, the rest queued), then collect each reply.
fn run_batched(e: &BatchedEngine, reqs: &[(Vec<u32>, usize)]) -> Vec<Out> {
    let pending: Vec<_> = reqs
        .iter()
        .map(|(p, n)| {
            let (req, trace) = traced(p, *n);
            (e.submit(req).unwrap(), trace)
        })
        .collect();
    pending
        .into_iter()
        .map(|(rx, trace)| {
            let (mut ids, mut cached) = (vec![], usize::MAX);
            for ev in rx.iter() {
                match ev {
                    GenEvent::Token { id, .. } => ids.push(id),
                    GenEvent::Done { cached_tokens, .. } => {
                        cached = cached_tokens;
                        break;
                    }
                    GenEvent::Error { message } => panic!("{message}"),
                }
            }
            assert_ne!(cached, usize::MAX, "no Done");
            let logits = trace.lock().unwrap().clone();
            Out { ids, logits, cached }
        })
        .collect()
}

/// Three multi-turn conversations, three turns each, every round submitted together to a 2-slot batched engine (so two run
/// beside each other and one waits; a prompt prefills while the other slot decodes, slots change conversations, turns restore
/// from the store). Every reply is bitwise the single-sequence engine's with the store on (the same requests one at a time).
fn batched_multi_turn(kind: Kind, chunk: usize, reuse: bool) {
    batched_multi_turn_opts(kind, chunk, |o| o.prefix_reuse = reuse);
}

/// [`batched_multi_turn`] with the batched engine's options set by `f`; returns the engine (idle) for its metrics.
fn batched_multi_turn_opts(kind: Kind, chunk: usize, f: impl FnOnce(&mut BatchedOptions)) -> BatchedEngine {
    let cfg = store_cfg();
    let solo = LocalEngine::new(planned(kind, chunk, 1), Arc::new(kind.codec()), store_meta(Some(cfg.clone())), Arc::new(Telemetry::new()))
        .unwrap();
    let eng = batched_store_opts(kind, chunk, Some(cfg), f).unwrap();
    let mut chats: Vec<Chat> = (0..3u32)
        .map(|c| {
            let mut ch = Chat::new(kind);
            ch.msg(10 + c, 19 + 4 * c as usize).msg(20 + c, 7 + c as usize);
            ch
        })
        .collect();
    let gen = [9usize, 5, 7];
    let mut cached_later = 0;
    for round in 0..3u32 {
        let reqs: Vec<(Vec<u32>, usize)> = chats.iter().zip(gen).map(|(c, n)| (c.prompt(), n)).collect();
        let got = run_batched(&eng, &reqs);
        for (i, ((p, n), b)) in reqs.iter().zip(&got).enumerate() {
            let s = run(&solo, p, *n);
            same_bits(b, &s, &format!("round {round} conversation {i}: batched vs single-sequence (store on)"));
            if round > 0 {
                cached_later += b.cached;
            }
        }
        for (c, (out, extra)) in chats.iter_mut().zip(got.iter().zip([5usize, 9, 3])) {
            c.answer(&out.ids).msg(30 + round, extra);
        }
    }
    assert!(cached_later > 0, "later turns reused nothing");
    let snap = eng.telemetry().prefix_store.expect("the batched engine reports its store");
    assert!(snap.inserts > 0, "nothing was captured: {snap:?}");
    assert!(snap.hits > 0 && snap.tokens_saved > 0, "no restore from the store: {snap:?}");
    let info = eng.prefix_store();
    assert!(info["entries"].as_array().is_some_and(|e| !e.is_empty()), "{info}");
    eng
}

#[test]
fn batched_hybrid_multi_turn_is_the_single_engine_store_on() {
    batched_multi_turn(Kind::Hybrid, 8, true);
}

#[test]
fn batched_hybrid_multi_turn_without_slot_reuse() {
    // every request starts from an empty slot (the CUDA case: no batched snapshots), so every later turn restores
    batched_multi_turn(Kind::Hybrid, 8, false);
}

#[test]
fn batched_dense_multi_turn_is_the_single_engine_store_on() {
    batched_multi_turn(Kind::Dense, 8, true);
}

/// Lane interleave: chunks that park between layers while the other slot decodes (2 rows per segment of an 8-row chunk), and a
/// settle in 1 MiB slices (one object each on the tiny models): every reply is still bitwise the single-sequence engine's with the store on, decode steps ran
/// inside parked chunks, and once idle nothing deferred is left on the device.
fn batched_multi_turn_paused(kind: Kind) {
    let eng = batched_multi_turn_opts(kind, 8, |o| {
        o.pause_rows = 2;
        o.settle_slice_mb = 1;
    });
    let m = eng.metrics();
    assert!(m.paused_steps > 0, "no decode step ran inside a parked chunk: {m:?}");
    let t0 = std::time::Instant::now();
    loop {
        // the scheduler settles what is left once it is idle, right after the last request's `Done`
        let snap = eng.telemetry().prefix_store.unwrap();
        if snap.device.bytes == 0 {
            break;
        }
        assert!(t0.elapsed().as_secs() < 10, "deferred captures left on the device: {snap:?}");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn batched_hybrid_multi_turn_paused_chunks_are_bitwise() {
    batched_multi_turn_paused(Kind::Hybrid);
}

#[test]
fn batched_dense_multi_turn_paused_chunks_are_bitwise() {
    batched_multi_turn_paused(Kind::Dense);
}

/// Fable #359 (1): a request that fails after its prefill (a bad decode row: `fail_active`, never `finish`) leaves its deferred
/// captures on the device; once the engine is idle they settle (the idle housekeeping), with no further admission. Both with the
/// default whole settle and with sliced settles.
#[test]
fn batched_store_failed_request_settles_when_idle() {
    for slice in [0usize, 1] {
        let kind = Kind::Hybrid;
        let mut a = Chat::new(kind);
        a.msg(1, 23).msg(2, 9);
        let prompt = a.prompt();
        let plen = prompt.len();
        // slot 0 is fed an out-of-vocab id at its second decode row: the request fails there, its prefill captures deferred
        let eng = batched_store_opts(kind, 8, Some(store_cfg()), |o| {
            o.row_fault = Some((0, plen + 1, 99_999));
            o.settle_slice_mb = slice;
        })
        .unwrap();
        let (req, _) = traced(&prompt, 6);
        let rx = eng.submit(req).unwrap();
        let err = rx.iter().find_map(|ev| match ev {
            GenEvent::Error { message } => Some(message),
            GenEvent::Done { .. } => panic!("the faulted request finished"),
            _ => None,
        });
        assert!(err.is_some_and(|m| m.contains("vocab")), "slice {slice}: the request must fail on its bad row");
        let t0 = std::time::Instant::now();
        loop {
            let snap = eng.telemetry().prefix_store.unwrap();
            if snap.inserts > 0 && snap.device.bytes == 0 {
                break;
            }
            assert!(t0.elapsed().as_secs() < 10, "slice {slice}: the failed request's captures stayed on the device: {snap:?}");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(eng.failed().is_none());
    }
}

#[test]
fn batched_store_refuses_overlap_pack_and_prewarm() {
    let err = |r: intarsia_core::Result<BatchedEngine>| r.err().map(|e| e.to_string()).unwrap_or_default();
    let e = err(batched_store_opts(Kind::Hybrid, 8, Some(store_cfg()), |o| o.overlap = true));
    assert!(e.contains("INTARSIA_OVERLAP"), "{e}");
    let e = err(batched_store_opts(Kind::Hybrid, 8, Some(store_cfg()), |o| o.pack_prefill = Some(true)));
    assert!(e.contains("INTARSIA_PACK_PREFILL"), "{e}");
    // without the store both stay accepted
    assert!(batched_store_opts(Kind::Hybrid, 8, None, |o| o.pack_prefill = Some(true)).is_ok());
    let eng = batched_store(Kind::Hybrid, 8, store_cfg(), true);
    let req = GenRequest { prompt: vec![3, 25, 5, 40, 41], max_tokens: 1, prewarm: Some(Prewarm { pin: None }), ..Default::default() };
    let e = eng.submit(req).err().map(|e| e.to_string()).unwrap_or_default();
    assert!(e.contains("prewarm"), "{e}");
}


#[test]
fn batched_store_refuses_speculation() {
    use intarsia_core::spec::{SpecConfig, SpecMode};
    let mut meta = store_meta(Some(store_cfg()));
    meta.spec = SpecConfig { mode: SpecMode::Mtp, k: 2, k_min: None };
    // `INTARSIA_MTP_STORE=0`; on CUDA by default `--spec mtp` runs with the store (lane mtp-store, `tests/mtp.rs`)
    meta.mtp_store = Some(false);
    let mut o = BatchedOptions::new(2, 64);
    o.mtp_max_b = 1;
    let e = BatchedEngine::new(planned(Kind::Hybrid, 8, 2), Arc::new(Kind::Hybrid.codec()), meta, Arc::new(Telemetry::new()), o)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("prefix store") && e.contains("speculative decoding"), "{e}");
}

/// A prompt that does not fit a slot is rejected before the store is looked up: no lookup, no restore, and the slot keeps what
/// it held (the next turn of the conversation still finds its history).
#[test]
fn batched_store_rejects_an_over_length_prompt_before_the_lookup() {
    let kind = Kind::Hybrid;
    let eng = batched_store(kind, 8, store_cfg(), true);
    let mut a = Chat::new(kind);
    a.msg(1, 23).msg(2, 9);
    let t1 = run_batched(&eng, &[(a.prompt(), 4)]).remove(0);
    let lookups = eng.telemetry().prefix_store.unwrap().lookups;
    let mut long = a.prompt();
    long.extend(std::iter::repeat(40).take(eng.info().context_len));
    let (req, _) = traced(&long, 4);
    let rx = eng.submit(req).unwrap();
    let err = rx.iter().find_map(|ev| match ev {
        GenEvent::Error { message } => Some(message),
        _ => None,
    });
    assert!(err.is_some_and(|m| m.contains("does not fit")), "an over-length prompt must be rejected");
    assert_eq!(eng.telemetry().prefix_store.unwrap().lookups, lookups, "the rejected prompt was looked up");
    a.answer(&t1.ids).msg(3, 5);
    let t2 = run_batched(&eng, &[(a.prompt(), 4)]).remove(0);
    assert!(t2.cached > 30, "turn 2 reused only {} tokens", t2.cached);
}

/// A structured-output constraint that cannot start rejects the request before the store is looked up: no lookup, no restore into
/// the slot, which keeps what it held. (A token id outside the vocabulary is refused by `submit` itself.)
#[test]
fn batched_store_rejects_a_bad_constraint_before_the_lookup() {
    #[derive(Debug)]
    struct Broken;
    impl intarsia_core::constrain::ConstraintSpec for Broken {
        fn start(&self) -> intarsia_core::Result<Box<dyn intarsia_core::constrain::TokenConstraint>> {
            Err(intarsia_core::Error::BadRequest("no such schema".into()))
        }
    }
    let kind = Kind::Hybrid;
    let eng = batched_store(kind, 8, store_cfg(), true);
    let mut a = Chat::new(kind);
    a.msg(1, 23).msg(2, 9);
    let t1 = run_batched(&eng, &[(a.prompt(), 4)]).remove(0);
    let lookups = eng.telemetry().prefix_store.unwrap().lookups;
    // the conversation's own history, which the store could restore
    let (mut req, _) = traced(&a.prompt(), 4);
    req.constraint = Some(Arc::new(Broken));
    let rx = eng.submit(req).unwrap();
    let err = rx.iter().find_map(|ev| match ev {
        GenEvent::Error { message } => Some(message),
        _ => None,
    });
    assert!(err.is_some_and(|m| m.contains("structured output")), "a broken constraint must be rejected");
    assert_eq!(eng.telemetry().prefix_store.unwrap().lookups, lookups, "the rejected prompt was looked up");
    a.answer(&t1.ids).msg(3, 5);
    let t2 = run_batched(&eng, &[(a.prompt(), 4)]).remove(0);
    assert!(t2.cached > 30, "turn 2 reused only {} tokens", t2.cached);
}
