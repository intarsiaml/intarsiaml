//! Content-addressed prefix store: prefill that was already done is restored instead of recomputed.
//!
//! A sequence's state after `len` prompt tokens (every attention slot's K/V rows `0..len` and every recurrent slot:
//! GatedDeltaNet / Mamba conv and delta-rule state) is a pure function of the token ids `0..len`, the build that
//! computed it and the chunk schedule it was computed with. The store keeps such states keyed by a hash of exactly
//! those inputs, so any request whose prompt starts with a stored prefix (another conversation, a regenerated answer,
//! the same system prompt and tool schema, the same RAG documents) restores it and prefills only the rest.
//!
//! Layout (docs/results/2026-10-22-pc-prefix-store.md):
//! - An **entry** is one restorable prefix: its token ids (compared in full on every hit, so a hash collision can never
//!   restore a wrong state), one recurrent blob (all recurrent slots, slot order; none for attention-only models) and
//!   the list of KV **pages** covering `0..len`. With an MTP head (`--spec mtp`, lane mhkv) an entry a drafting request
//!   stored also has the head's pages (its cache rows, shifted by one position against the main pages) and its queued
//!   hidden row, so a restored sequence drafts as after a cold prefill ([`HeadLayout`]).
//! - A **page** holds `page_tokens` rows of every attention slot (K then V, per head). Pages are content-addressed by
//!   the hash of the whole prefix up to the page end, so the entries of one conversation (turn 1, turn 2, ...) and of
//!   every conversation sharing a system prompt share their KV pages; only the recurrent blob is per entry.
//! - Objects (pages, blobs) live in tiers: device memory, host memory, disk. LRU objects move down a tier when a tier
//!   is over budget and are dropped from the last one (an entry missing an object is dropped with it). Pinned entries
//!   (named system prompts, tool schemas) never lose their objects.
//! - Every key is salted with the build fingerprint (model, recipe, dtypes, resolved kernel plan, backend fingerprint,
//!   chunk configuration, store format version, the executable): an entry of another build is never restored, and disk
//!   files of another build or format are rejected on load.
//!
//! The store only holds prefill-written state at points of the chunk schedule (the engine's rule, see
//! `engine::Seq::prefill_step`), so a restore gives the bits a cold prefill of the same prompt gives.

use crate::backend::{Backend, Buffer, MemKind};
use crate::error::{Error, Result};
use crate::state::{MtpState, SequenceState, SlotSpec, StateSlot};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Version of the key derivation and of the on-disk format. Bumped on any change of either.
pub const FORMAT_VERSION: u32 = 1;
const OBJ_MAGIC: &[u8; 4] = b"IPSO";
const ENTRY_MAGIC: &[u8; 4] = b"IPSE";
const MIB: u64 = 1 << 20;

/// Store configuration (`intarsia serve --prefix-store ...` or `INTARSIA_PREFIX_STORE`).
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixStoreConfig {
    /// Budget of the device tier in bytes (0 = no device tier; the default: device memory is planned for the model).
    pub device_bytes: u64,
    /// Budget of the host tier in bytes.
    pub host_bytes: u64,
    /// Budget of the disk tier in bytes (0 = no disk tier).
    pub disk_bytes: u64,
    /// Directory of the disk tier (files go under `<dir>/<build fingerprint>/`).
    pub disk_dir: Option<PathBuf>,
    /// Tokens per KV page.
    pub page_tokens: usize,
    /// Shortest prefix worth an entry.
    pub min_tokens: usize,
    /// Smallest distance in tokens between two entries of one prompt that are not message boundaries (chunk-grid
    /// points inside a long message, e.g. a pasted document).
    pub grid_tokens: usize,
    /// Token texts that start a message (each must be one token): a prefill chunk ends right before each, and an entry
    /// is stored there.
    pub boundary_tokens: Vec<String>,
    /// Token texts that end a message header (`<|im_start|>assistant\n`: the newline): a chunk also ends right after
    /// the first one following a boundary token. That cut is where the next turn's prompt leaves this one (the Qwen3.6
    /// template drops earlier turns' `<think>`, so turn N+1 shares turn N's prompt up to `<|im_start|>assistant\n`).
    pub header_end_tokens: Vec<String>,
    /// Restore at any prefill-written length, not only at points of the prompt's own chunk schedule. Only for a backend
    /// and model whose prefill is bitwise chunk-invariant (measured: dense models on cpu; NOT the hybrid on cpu, whose
    /// GatedDeltaNet bits depend on the chunking). Default off.
    pub any_cut: bool,
    /// Only cut after the headers of these roles (texts as the template writes them, e.g. `assistant`), plus the second
    /// message's. Empty (the default): every header. Fewer cuts mean fewer forward passes (on a MoE each pass reads every
    /// expert), but a prewarm or a divergence inside another role's message restores less.
    pub cut_roles: Vec<String>,
}

impl Default for PrefixStoreConfig {
    fn default() -> Self {
        PrefixStoreConfig {
            device_bytes: 0,
            host_bytes: 8192 * MIB,
            disk_bytes: 0,
            disk_dir: None,
            page_tokens: 256,
            min_tokens: 32,
            grid_tokens: 2048,
            boundary_tokens: vec!["<|im_start|>".into()],
            header_end_tokens: vec!["\n".into()],
            any_cut: false,
            cut_roles: Vec::new(),
        }
    }
}

/// Hard limits on the knobs (a typo cannot ask for a petabyte or a 1-token page).
const MAX_TIER_MB: u64 = 4 << 20; // 4 TiB
const PAGE_RANGE: (usize, usize) = (16, 8192);

impl PrefixStoreConfig {
    /// Parse `key=value` pairs separated by commas: `on` (or `1`) alone takes the defaults; keys `host_mb`, `device_mb`,
    /// `disk_mb`, `dir`, `page`, `min`, `grid`, `boundary` (`|`-separated token texts). `off`, `0` or empty: `None`.
    pub fn parse(spec: &str) -> Result<Option<PrefixStoreConfig>> {
        let spec = spec.trim();
        if spec.is_empty() || spec == "0" || spec.eq_ignore_ascii_case("off") {
            return Ok(None);
        }
        let mut c = PrefixStoreConfig::default();
        let mut disk_set = false;
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if part == "1" || part.eq_ignore_ascii_case("on") {
                continue;
            }
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| Error::invalid(format!("prefix store: '{part}' is not key=value (spec '{spec}')")))?;
            let num = |v: &str| -> Result<u64> {
                v.trim().parse::<u64>().map_err(|_| Error::invalid(format!("prefix store: {k}={v} is not a number")))
            };
            let mb = |v: &str| -> Result<u64> {
                let n = num(v)?;
                if n > MAX_TIER_MB {
                    return Err(Error::invalid(format!("prefix store: {k}={n} MiB is above the {MAX_TIER_MB} MiB limit")));
                }
                Ok(n * MIB)
            };
            match k.trim() {
                "host_mb" => c.host_bytes = mb(v)?,
                "device_mb" => c.device_bytes = mb(v)?,
                "disk_mb" => {
                    c.disk_bytes = mb(v)?;
                    disk_set = true;
                }
                "dir" => c.disk_dir = Some(PathBuf::from(v.trim())),
                "page" => {
                    let p = num(v)? as usize;
                    if !(PAGE_RANGE.0..=PAGE_RANGE.1).contains(&p) {
                        return Err(Error::invalid(format!(
                            "prefix store: page={p} outside {}..={}",
                            PAGE_RANGE.0, PAGE_RANGE.1
                        )));
                    }
                    c.page_tokens = p;
                }
                "min" => c.min_tokens = (num(v)? as usize).max(1),
                "grid" => c.grid_tokens = (num(v)? as usize).max(1),
                "boundary" => {
                    c.boundary_tokens = v.split('|').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect()
                }
                "header" => {
                    // `\n` spelled as two characters in a flag
                    c.header_end_tokens =
                        v.split('|').filter(|s| !s.is_empty()).map(|s| s.replace("\\n", "\n")).collect()
                }
                "any_cut" => c.any_cut = num(v)? != 0,
                "roles" => c.cut_roles = v.split('|').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect(),
                other => return Err(Error::invalid(format!("prefix store: unknown key '{other}' (spec '{spec}')"))),
            }
        }
        if c.disk_dir.is_some() && !disk_set {
            c.disk_bytes = 64 * 1024 * MIB;
        }
        if c.disk_bytes > 0 && c.disk_dir.is_none() {
            return Err(Error::invalid("prefix store: disk_mb needs dir=PATH"));
        }
        Ok(Some(c))
    }

    /// `INTARSIA_PREFIX_STORE` (unset: `None`).
    pub fn from_env() -> Result<Option<PrefixStoreConfig>> {
        match std::env::var("INTARSIA_PREFIX_STORE") {
            Ok(v) => Self::parse(&v),
            Err(_) => Ok(None),
        }
    }
}

// ------------------------------------------------------------------------------------------------------------- hashing

/// 128-bit key (two independent 64-bit lanes).
pub type Key = u128;

fn mix(mut x: u64) -> u64 {
    // splitmix64 finaliser
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Incremental hash of a token sequence in one domain (entries, pages), salted with the build.
#[derive(Clone, Copy)]
struct TokHash {
    a: u64,
    b: u64,
}

const DOMAIN_ENTRY: u64 = 0x454e_5452_5900_0001;
const DOMAIN_PAGE: u64 = 0x5041_4745_0000_0002;

impl TokHash {
    fn new(build: u64, domain: u64) -> TokHash {
        TokHash { a: mix(build ^ domain), b: mix(build.rotate_left(29) ^ domain ^ 0xC2B2_AE3D_27D4_EB4F) }
    }
    fn push(&mut self, t: u32) {
        self.a = mix(self.a ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        self.b = mix(self.b.wrapping_add(t as u64 ^ 0xD6E8_FEB8_6659_FD93).rotate_left(17));
    }
    fn key(&self, len: usize) -> Key {
        let a = mix(self.a ^ (len as u64));
        let b = mix(self.b ^ (len as u64).rotate_left(32));
        ((a as u128) << 64) | b as u128
    }
}

/// Key of the entry for exactly `tokens`.
fn entry_key(build: u64, tokens: &[u32]) -> Key {
    let mut h = TokHash::new(build, DOMAIN_ENTRY);
    for &t in tokens {
        h.push(t);
    }
    h.key(tokens.len())
}

/// Keys of the pages covering `tokens` (`page` tokens each, the last one partial). `ends` are the ends of the prefill
/// chunks that wrote the rows (sorted; the last one is `tokens.len()`). A page's rows were written by chunks that may
/// extend past the page end, and on a chunk-sensitive kernel a row's bits depend on its chunk; so a page is keyed by the
/// tokens up to the end of the chunk that wrote its last row, and two entries share a page only when they share that
/// chunk.
fn page_keys(build: u64, tokens: &[u32], page: usize, ends: &[usize]) -> Vec<(Key, usize, usize)> {
    let n = tokens.len();
    let mut h = TokHash::new(build, DOMAIN_PAGE);
    let mut at: HashMap<usize, TokHash> = HashMap::new();
    for (i, &t) in tokens.iter().enumerate() {
        h.push(t);
        at.insert(i + 1, h);
    }
    let mut out = Vec::with_capacity(n.div_ceil(page));
    let mut start = 0;
    while start < n {
        let end = (start + page).min(n);
        let e = ends.iter().copied().find(|&e| e >= end).unwrap_or(n).min(n);
        let mut k = at[&e];
        k.push(u32::MAX);
        k.push(start as u32);
        k.push(end as u32);
        out.push((k.key(e), start, end));
        start = end;
    }
    out
}

/// Word-at-a-time checksum of a payload (not cryptographic: catches torn or flipped files).
pub fn checksum(data: &[u8]) -> u64 {
    let mut h = 0x8422_2325_CBF2_9CE4u64 ^ data.len() as u64;
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(29);
    }
    for &b in chunks.remainder() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    mix(h)
}

/// The build fingerprint every key is salted with: hash of `desc` (see [`PrefixStore::new`]) and the format version.
pub fn build_fingerprint(desc: &str) -> u64 {
    let mut h = TokHash::new(FORMAT_VERSION as u64, 0x4255_494c_4400_0003);
    for c in desc.as_bytes().chunks(4) {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        h.push(u32::from_le_bytes(w));
    }
    (h.key(desc.len()) >> 64) as u64 ^ h.key(desc.len()) as u64
}

// ------------------------------------------------------------------------------------------------------------- layout

#[derive(Clone, Debug, PartialEq)]
struct AttnLayout {
    slot: usize,
    heads: usize,
    head_dim: usize,
    elem: usize,
}

/// What a state looks like to the store: the attention slots' row geometry and the recurrent slots' sizes.
#[derive(Clone, Debug, PartialEq)]
struct Layout {
    attn: Vec<AttnLayout>,
    rec: Vec<(usize, usize)>,
    rec_bytes: usize,
}

impl Layout {
    fn from_specs(specs: &[SlotSpec]) -> Layout {
        let mut attn = Vec::new();
        let mut rec = Vec::new();
        for (i, s) in specs.iter().enumerate() {
            match s {
                SlotSpec::AttnKv { n_kv_heads, head_dim, dtype } => {
                    attn.push(AttnLayout { slot: i, heads: *n_kv_heads, head_dim: *head_dim, elem: dtype.size_bytes() })
                }
                SlotSpec::Recurrent { bytes } => rec.push((i, *bytes)),
                // DeepSeek-V4.1's ring and shared slots are not stored (refused in `check`)
                SlotSpec::Ring { .. } | SlotSpec::Shared { .. } => {}
            }
        }
        let rec_bytes = rec.iter().map(|r| r.1).sum();
        Layout { attn, rec, rec_bytes }
    }

    fn page_bytes(&self, rows: usize) -> usize {
        self.attn.iter().map(|a| 2 * a.heads * rows * a.head_dim * a.elem).sum()
    }

    fn check(&self, st: &SequenceState) -> Result<()> {
        if st.slots.iter().any(|s| matches!(s, StateSlot::Ring(_) | StateSlot::Shared(_))) {
            return Err(Error::Unsupported("prefix store of DeepSeek-V4.1 ring / shared state slots".into()));
        }
        let ok = self.attn.iter().all(|a| match st.slots.get(a.slot) {
            Some(StateSlot::AttnKv(s)) => s.n_kv_heads == a.heads && s.head_dim == a.head_dim && s.dtype.size_bytes() == a.elem,
            _ => false,
        }) && self.rec.iter().all(|&(i, b)| matches!(st.slots.get(i), Some(StateSlot::Recurrent(r)) if r.bytes == b));
        if !ok || st.slots.len() != self.attn.len() + self.rec.len() {
            return Err(Error::invalid("prefix store: the sequence state does not match the store's slot layout"));
        }
        if !st.peers.is_empty() || st.remote.is_some() {
            return Err(Error::Unsupported("prefix store with tensor parallelism".into()));
        }
        Ok(())
    }

    /// `(state buffer, byte offset, bytes)` of every segment of rows `start..end`, in page order: per attention slot K then
    /// V, per head.
    fn segments<'s>(&self, st: &'s SequenceState, start: usize, end: usize) -> Vec<(&'s Buffer, usize, usize)> {
        let mut out = Vec::with_capacity(self.attn.len() * 4);
        for a in &self.attn {
            let StateSlot::AttnKv(s) = &st.slots[a.slot] else { unreachable!("checked layout") };
            let row = a.head_dim * a.elem;
            for buf in [&s.k, &s.v] {
                for h in 0..a.heads {
                    out.push((buf, (h * s.cap + start) * row, (end - start) * row));
                }
            }
        }
        out
    }

    fn rec_segments<'s>(&self, st: &'s SequenceState) -> Vec<(&'s Buffer, usize, usize)> {
        self.rec
            .iter()
            .map(|&(i, b)| {
                let StateSlot::Recurrent(r) = &st.slots[i] else { unreachable!("checked layout") };
                (&r.buf, 0, b)
            })
            .collect()
    }
}

/// The MTP head's part of an entry (lane mhkv; `--spec mtp`). The head trails the main model by one position: its cache row
/// `p` reads the main model's hidden row of `p` and the token of `p + 1`, so after a prefill chunk ending at `len` it holds
/// rows `0..len - 1` and queues the hidden row of `len - 1`. Head pages are shifted by one against the main pages: the head
/// page of the main page `[start, end)` holds head rows `[start - 1, end - 1)` (`[0, end - 1)` for the first), which depend on
/// the tokens up to `end` and on the chunks that wrote main rows up to `end - 1`, exactly what the main page's key covers, so
/// a head page shares its main page's key (salted) and is shared between entries alike. The queued row is per entry (the
/// head tail). An entry holds no head part when the sequence's head did not follow it from position 0 (a request that did
/// not draft, a head context restarted by an earlier restore), and then a restore restarts the head's context as before.
#[derive(Clone, Debug, PartialEq)]
struct HeadLayout {
    kv: AttnLayout,
    /// Bytes of one queued hidden row.
    row_bytes: usize,
}

/// Salts of the head objects' keys (a head page: its main page's key; the tail: the entry's key). The recurrent blob uses
/// `u128::MAX >> 1`.
const HEAD_PAGE_SALT: u128 = 0x4d54_5048_4541_4450_4147_4500_0000_0001;
const HEAD_TAIL_SALT: u128 = 0x4d54_5048_4541_4454_4149_4c00_0000_0002;

impl HeadLayout {
    fn from_spec(spec: &SlotSpec, row_bytes: usize) -> Option<HeadLayout> {
        match spec {
            SlotSpec::AttnKv { n_kv_heads, head_dim, dtype } => Some(HeadLayout {
                kv: AttnLayout { slot: 0, heads: *n_kv_heads, head_dim: *head_dim, elem: dtype.size_bytes() },
                row_bytes,
            }),
            _ => None,
        }
    }

    /// Head rows stored with the main page `[start, end)`.
    fn rows(start: usize, end: usize) -> (usize, usize) {
        (start.saturating_sub(1), end - 1)
    }

    fn page_bytes(&self, start: usize, end: usize) -> usize {
        let (a, b) = Self::rows(start, end);
        2 * self.kv.heads * (b - a) * self.kv.head_dim * self.kv.elem
    }

    /// The head state matches this layout.
    fn fits(&self, m: &MtpState) -> bool {
        matches!(m.kv.slots.as_slice(), [StateSlot::AttnKv(s)] if s.n_kv_heads == self.kv.heads && s.head_dim == self.kv.head_dim && s.dtype.size_bytes() == self.kv.elem)
            && m.row_bytes == self.row_bytes
            && m.cap_rows >= 1
    }

    /// The head of `st` can be stored with an entry of `len` tokens: it followed the sequence from position 0 through
    /// prefill passes that caught it up (cache `0..len - 1`, one queued row).
    fn capturable(&self, st: &SequenceState, len: usize) -> bool {
        st.mtp.as_ref().is_some_and(|m| {
            m.active && m.synced && m.base == 0 && len >= 2 && st.len == len && m.kv.len + 1 == len && m.pending_rows == 1 && self.fits(m)
        })
    }

    /// Segments of the head page for the main page `[start, end)` (per K / V, per head).
    fn segments<'s>(&self, m: &'s MtpState, start: usize, end: usize) -> Vec<(&'s Buffer, usize, usize)> {
        let (a, b) = Self::rows(start, end);
        let StateSlot::AttnKv(s) = &m.kv.slots[0] else { unreachable!("checked layout") };
        let row = self.kv.head_dim * self.kv.elem;
        let mut out = Vec::with_capacity(2 * self.kv.heads);
        for buf in [&s.k, &s.v] {
            for h in 0..self.kv.heads {
                out.push((buf, (h * s.cap + a) * row, (b - a) * row));
            }
        }
        out
    }
}

/// An entry's head objects: one page per main page, then the tail.
#[derive(Clone, Debug)]
struct HeadRef {
    pages: Vec<Key>,
    tail: Key,
}

// ------------------------------------------------------------------------------------------------------------- objects

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Device,
    Host,
    Disk,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Device => "device",
            Tier::Host => "host",
            Tier::Disk => "disk",
        }
    }
}

enum Mem {
    /// A range of a device allocation (one per insert, shared by that insert's objects) starting at the offset.
    Device(Buffer, usize),
    Host(Arc<HostBuf>),
}

struct Obj {
    bytes: usize,
    mem: Option<Mem>,
    /// On disk (or queued for the writer, whose staged copy `staged` holds until it confirms).
    disk: Option<PathBuf>,
    staged: Option<Arc<HostBuf>>,
    last_use: u64,
    /// Entries referencing it, and how many of those are pinned.
    refs: u32,
    pins: u32,
}

impl Obj {
    fn tier(&self) -> Tier {
        match &self.mem {
            Some(Mem::Device(..)) => Tier::Device,
            Some(Mem::Host(_)) => Tier::Host,
            None => Tier::Disk,
        }
    }
}

struct Entry {
    tokens: Arc<Vec<u32>>,
    /// The chunk ends its pages are keyed by ([`page_keys`]).
    ends: Vec<usize>,
    rec: Option<Key>,
    pages: Vec<Key>,
    /// The MTP head's pages and tail ([`HeadLayout`]); `None`: the entry restores the main model only.
    head: Option<HeadRef>,
    pinned: Option<String>,
    last_use: u64,
    hits: u64,
    created: Instant,
}

impl Entry {
    /// Every object the entry references (an entry missing one is not restorable and is dropped with it).
    fn objs(&self) -> impl Iterator<Item = &Key> + '_ {
        self.rec.iter().chain(self.pages.iter()).chain(self.head.iter().flat_map(|h| h.pages.iter().chain(std::iter::once(&h.tail))))
    }
}

/// A restorable prefix found for a prompt ([`PrefixStore::lookup`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub key: Key,
    pub len: usize,
}

/// What a restore did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Restored {
    pub len: usize,
    pub bytes: u64,
    pub elapsed: Duration,
    /// The slowest tier an object came from.
    pub tier: Option<Tier>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TierStats {
    pub objects: u64,
    pub bytes: u64,
    pub budget_bytes: u64,
    /// Restores whose slowest object came from this tier.
    pub hits: u64,
    /// Objects moved out of this tier (down a tier or dropped).
    pub evictions: u64,
}

/// Counters for `/metrics` and the dashboard (`TelemetrySnapshot::prefix_store`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct PrefixStoreSnapshot {
    pub build: String,
    pub entries: u64,
    pub pinned_entries: u64,
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    /// Hits that lost to a longer prefix the sequence already held (nothing restored).
    pub local_wins: u64,
    pub tokens_saved: u64,
    pub bytes_restored: u64,
    pub restore_seconds_total: f64,
    pub last_restore_ms: f64,
    pub inserts: u64,
    pub insert_bytes: u64,
    pub insert_seconds_total: f64,
    /// Part of `insert_seconds_total` spent in `settle`, moving deferred captures off the device (the engine calls it after
    /// the request's `Done` went out).
    pub settle_seconds_total: f64,
    /// Restores rejected because an object failed its checks (corrupt or foreign file); the request ran cold.
    pub rejected: u64,
    /// Estimated time to first token saved: tokens restored at the measured cold prefill rate, minus restore time.
    pub ttft_saved_seconds_total: f64,
    /// Cold prefill rate (tok/s, moving average) the estimate uses.
    pub prefill_tok_s: f64,
    pub device: TierStats,
    pub host: TierStats,
    pub disk: TierStats,
    /// Host buffers queued for the disk writer and not yet confirmed (the writer holds them, whatever tier the object is in now).
    pub disk_pending_bytes: u64,
    /// Objects not written to disk because the pending writes had reached `host_mb` (a disk slower than the inserts): they
    /// stay host-only and are dropped, not demoted, when the host tier evicts them.
    pub disk_write_skips: u64,
    /// Entries stored with the MTP head's state (lane mhkv), and restores that also restored it (the head then drafts as
    /// after a cold prefill instead of restarting at the restore point). Absent without an MTP head.
    #[serde(skip_serializing_if = "is_zero")]
    pub mtp_head_inserts: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub mtp_head_restores: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// One entry, for `GET /v1/intarsia/prefix-store`.
#[derive(Clone, Debug, Serialize)]
pub struct EntryInfo {
    pub key: String,
    pub tokens: usize,
    pub tier: Tier,
    pub bytes: u64,
    pub pinned: Option<String>,
    pub hits: u64,
    pub age_s: f64,
}

// ------------------------------------------------------------------------------------------------------------- store

/// The store. Owned by the engine worker (behind a mutex that is only held for a lookup, insert or restore).
pub struct PrefixStore {
    cfg: PrefixStoreConfig,
    build: u64,
    layout: Layout,
    /// The MTP head's layout when entries keep the head's state ([`PrefixStore::new_with_head`]).
    head: Option<HeadLayout>,
    entries: HashMap<Key, Entry>,
    /// Distinct entry lengths (a lookup tries them longest first).
    lens: BTreeSet<usize>,
    objs: HashMap<Key, Obj>,
    clock: u64,
    used: [u64; 3],
    stats: PrefixStoreSnapshot,
    writer: Option<DiskWriter>,
    /// Inserts go to the device whatever its budget, and the device tier is not brought within budget, until
    /// [`PrefixStore::settle`] (the engine defers the device-to-host copies until the request's `Done` went out: a download
    /// synchronises the device, which would stall the prefill that queued the next chunk, and inside decode it stalled
    /// the stream; ab35c).
    defer: bool,
    /// Device bytes of the per-insert allocations made while deferring (only with a device tier, where no slab is used;
    /// capped at [`DEFER_CAP`], like the slabs' size: past it inserts copy to host directly).
    deferred: u64,
    /// Engine-thread time spent in `queue_write` since the last `settle` (reported under `INTARSIA_PREFIX_STORE_DEBUG`).
    queue_write_seconds: f64,
    /// Bytes sent to the disk writer and not yet confirmed. The queue holds those buffers even after the object left the host
    /// tier (the disk budget clears `staged`; an evicted or dropped object leaves its buffer to the queue), so this, not
    /// `used[1]`, is what a slow disk adds to resident memory: `queue_write` stops queueing past `host_bytes`.
    pending_write_bytes: u64,
    /// Pre-faulted host buffers freed by host-tier eviction, kept for the next demote (or host capture) of the same size
    /// (objects come in a few sizes; a fresh buffer costs a page fault per 4 KiB, a recycled one nothing). Counted with
    /// `used[1]` against the host budget: `enforce` drops pooled buffers first when the two together exceed it.
    host_free: Vec<HostBuf>,
    host_free_bytes: u64,
    /// Device slab the deferred captures of the current request are carved from (`bytes used` of it taken), and the
    /// slabs of this request it has outgrown; see [`PrefixStore::slab`].
    slab: Option<(Buffer, usize)>,
    slabs_full: Vec<Buffer>,
    /// Idle slabs kept for the next request (at most [`SLAB_POOL_CAP`] bytes): a device allocation costs 3-10 ms each.
    slab_pool: Vec<Buffer>,
}

/// `INTARSIA_PREFIX_STORE_DEBUG` is set: report the store's timings on stderr.
fn debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("INTARSIA_PREFIX_STORE_DEBUG").is_some())
}

/// Where one insert's capture went (reported under `INTARSIA_PREFIX_STORE_DEBUG`; `=sync` first drains the stream (`drain`:
/// the prefill queued before the capture), then waits for the capture's own copies, so `sync` is how long they take to run
/// on the device rather than to queue. It changes the timing it measures: use it to diagnose, not to benchmark).
#[derive(Default)]
struct CaptureTimes {
    segments: usize,
    alloc_s: f64,
    copy_s: f64,
    sync_s: f64,
    drain_s: f64,
    how: &'static str,
}

impl CaptureTimes {
    fn sync_mode() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("INTARSIA_PREFIX_STORE_DEBUG").is_ok_and(|v| v == "sync"))
    }

    fn drain(&mut self, backend: &dyn Backend) -> Result<()> {
        if Self::sync_mode() {
            let t = Instant::now();
            backend.synchronize(0)?;
            self.drain_s = t.elapsed().as_secs_f64();
        }
        Ok(())
    }

    fn sync(&mut self, backend: &dyn Backend) -> Result<()> {
        if Self::sync_mode() {
            let t = Instant::now();
            backend.synchronize(0)?;
            self.sync_s = t.elapsed().as_secs_f64();
        }
        Ok(())
    }
}

/// Most device memory the deferred captures may hold between two settles (with slabs: the slabs' size): one request's on the
/// single-sequence engine; on the batched engine every request in flight shares it (a settle after any request's `Done`
/// resets it), and past it inserts download to host synchronously, inside the prefill.
pub const DEFER_CAP: u64 = 1 << 30;

/// `INTARSIA_PREFIX_STORE_MACRO=1` (lane storettft, opt-in, default off): with `--prefix-store` and `--max-batch` > 1 the recipe's macro prefill
/// stays on: a request that prefills alone runs up to a macro pass of consecutive chunks of ITS OWN store schedule (the model's chunk, cut
/// at message boundaries) as one layer-by-layer pass, which is bitwise those chunks one by one, and the store captures at the pass's end
/// and at message cuts instead of after every chunk (`SeqStore::macro_len`).
pub fn store_macro() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_PREFIX_STORE_MACRO").is_ok_and(|v| v.trim() == "1"))
}

/// Device slabs the deferred captures are carved from: a new slab is at least this big (rounded up to the next multiple
/// of it), and the idle ones the store keeps between requests add up to at most `SLAB_POOL_CAP`.
const SLAB_MIN: usize = 128 << 20;
const SLAB_POOL_CAP: usize = 256 << 20;

struct DiskWriter {
    dir: PathBuf,
    tx: Sender<(Key, PathBuf, Arc<HostBuf>)>,
    /// `(key, written, bytes)` per queued write, in queue order.
    done: Receiver<(Key, bool, usize)>,
    _thread: std::thread::JoinHandle<()>,
}

fn tier_idx(t: Tier) -> usize {
    t as usize
}

/// Dropping the store (an engine shut down, a server stopping) waits for the disk writer, so an entry whose objects were
/// queued for disk before the drop is on disk afterwards and survives a restart (lane mstore). Bounded like [`PrefixStore::flush`].
impl Drop for PrefixStore {
    fn drop(&mut self) {
        self.flush();
    }
}

impl PrefixStore {
    /// A store for states laid out as `specs`. `build_desc` describes everything that can change the bits a prefill
    /// produces (the caller's [`describe_build`]); its fingerprint salts every key and names the disk directory.
    pub fn new(cfg: PrefixStoreConfig, specs: &[SlotSpec], build_desc: &str) -> Result<PrefixStore> {
        Self::new_with_head(cfg, specs, None, build_desc)
    }

    /// [`PrefixStore::new`] for a model with an MTP head (lane mhkv): `head` is the head's cache slot and the bytes of one
    /// queued hidden row (`Model::mtp_store_layout`); entries then also keep the head's cache and queue, so a restored
    /// sequence drafts as a cold one does. `build_desc` must say so ([`describe_build`] does).
    pub fn new_with_head(cfg: PrefixStoreConfig, specs: &[SlotSpec], head: Option<(SlotSpec, usize)>, build_desc: &str) -> Result<PrefixStore> {
        let build = build_fingerprint(build_desc);
        let layout = Layout::from_specs(specs);
        let head = head.and_then(|(spec, row)| HeadLayout::from_spec(&spec, row));
        let mut stats = PrefixStoreSnapshot { build: format!("{build:016x}"), ..Default::default() };
        stats.device.budget_bytes = cfg.device_bytes;
        stats.host.budget_bytes = cfg.host_bytes;
        stats.disk.budget_bytes = cfg.disk_bytes;
        let mut s = PrefixStore {
            cfg,
            build,
            layout,
            head,
            entries: HashMap::new(),
            lens: BTreeSet::new(),
            objs: HashMap::new(),
            clock: 0,
            used: [0; 3],
            stats,
            writer: None,
            defer: false,
            deferred: 0,
            queue_write_seconds: 0.0,
            pending_write_bytes: 0,
            host_free: Vec::new(),
            host_free_bytes: 0,
            slab: None,
            slabs_full: Vec::new(),
            slab_pool: Vec::new(),
        };
        if let (Some(dir), true) = (s.cfg.disk_dir.clone(), s.cfg.disk_bytes > 0) {
            let dir = dir.join(format!("{build:016x}"));
            std::fs::create_dir_all(&dir)?;
            s.writer = Some(DiskWriter::spawn(dir, build)?);
            s.load_disk();
        }
        Ok(s)
    }

    pub fn config(&self) -> &PrefixStoreConfig {
        &self.cfg
    }

    pub fn build(&self) -> u64 {
        self.build
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The longest stored prefix of `prompt` of at most `max_len` tokens whose objects are all present.
    pub fn lookup(&mut self, prompt: &[u32], max_len: usize) -> Option<Hit> {
        self.lookup_filtered(prompt, max_len, |_| true)
    }

    /// [`PrefixStore::lookup`] among the lengths `accept` allows (the engine's restore points for this prompt).
    pub fn lookup_filtered(&mut self, prompt: &[u32], max_len: usize, accept: impl Fn(usize) -> bool) -> Option<Hit> {
        self.poll_writer();
        self.stats.lookups += 1;
        let max_len = max_len.min(prompt.len());
        let cands: Vec<usize> = self.lens.range(..=max_len).rev().copied().filter(|&l| accept(l)).collect();
        if cands.is_empty() {
            self.stats.misses += 1;
            return None;
        }
        // prefix hash states at every candidate length, one pass
        let mut h = TokHash::new(self.build, DOMAIN_ENTRY);
        let mut keys = HashMap::with_capacity(cands.len());
        let mut want: BTreeSet<usize> = cands.iter().copied().collect();
        if want.remove(&0) {
            keys.insert(0, h.key(0));
        }
        for (i, &t) in prompt[..max_len].iter().enumerate() {
            h.push(t);
            if want.remove(&(i + 1)) {
                keys.insert(i + 1, h.key(i + 1));
            }
            if want.is_empty() {
                break;
            }
        }
        for len in cands {
            let key = keys[&len];
            let Some(e) = self.entries.get(&key) else { continue };
            if e.tokens.len() != len || e.tokens[..] != prompt[..len] {
                continue; // hash collision: never restore it
            }
            if e.objs().all(|k| self.objs.contains_key(k)) {
                return Some(Hit { key, len });
            }
        }
        self.stats.misses += 1;
        None
    }

    /// Whether an entry for exactly `tokens` exists.
    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.entries.contains_key(&entry_key(self.build, tokens))
    }

    /// Pin the entry for exactly `tokens` under `name`, if there is one (a prewarm pins its last header entry too).
    pub fn pin(&mut self, tokens: &[u32], name: &str) -> bool {
        let key = entry_key(self.build, tokens);
        let Some(e) = self.entries.get_mut(&key) else { return false };
        if e.pinned.is_some() {
            return true;
        }
        e.pinned = Some(name.to_string());
        let ks: Vec<Key> = e.objs().copied().collect();
        for k in ks {
            if let Some(o) = self.objs.get_mut(&k) {
                o.pins += 1;
            }
        }
        self.write_manifest(key);
        self.refresh_counts();
        true
    }

    /// Count a hit that a longer prefix already held by the sequence beat (nothing is restored).
    pub fn note_local_win(&mut self) {
        self.stats.local_wins += 1;
    }

    /// Store the state `st` holds for its first `len` tokens. Recurrent slots must hold exactly `len` tokens
    /// (`st.len == len`); attention rows `0..len` must be prefill-written (the caller's watermark). An existing entry
    /// is only touched (and pinned when `pin` is set). Returns the bytes copied.
    /// `ends`: the ends of the prefill chunks that wrote rows `0..len` (sorted, the last one `len`; see [`page_keys`]).
    pub fn insert(
        &mut self,
        backend: &dyn Backend,
        st: &SequenceState,
        len: usize,
        pin: Option<&str>,
        ends: &[usize],
    ) -> Result<u64> {
        self.poll_writer();
        self.layout.check(st)?;
        if len == 0 || len > st.len || st.tokens.len() < len {
            return Err(Error::invalid(format!("prefix store: insert of {len} tokens from a state of {}", st.len)));
        }
        if self.layout.rec_bytes > 0 && st.len != len {
            return Err(Error::invalid("prefix store: recurrent state holds a different length than the insert"));
        }
        let t0 = Instant::now();
        let tokens = &st.tokens[..len];
        let key = entry_key(self.build, tokens);
        let now = self.tick();
        let head_ok = self.head.as_ref().is_some_and(|h| h.capturable(st, len));
        if let Some(e) = self.entries.get_mut(&key) {
            e.last_use = now;
            let newly_pinned = pin.is_some() && e.pinned.is_none();
            if newly_pinned {
                e.pinned = pin.map(String::from);
            }
            let ks: Vec<Key> = e.objs().copied().collect();
            for k in ks {
                if let Some(o) = self.objs.get_mut(&k) {
                    o.last_use = now;
                    if newly_pinned {
                        o.pins += 1;
                    }
                }
            }
            if newly_pinned {
                self.write_manifest(key);
                self.refresh_counts();
            }
            return Ok(0);
        }
        let mut copied = 0u64;
        let pinned = pin.is_some();
        // pages: only the ones no other entry already holds; then the recurrent blob (this entry's only)
        let mut ends: Vec<usize> = ends.iter().copied().filter(|&e| e > 0 && e <= len).collect();
        if ends.last() != Some(&len) {
            ends.push(len);
        }
        let mut pages = Vec::new();
        let mut shared = Vec::new();
        let mut todo: Vec<(Key, Vec<(&Buffer, usize, usize)>, usize)> = Vec::new();
        for (pk, start, end) in page_keys(self.build, tokens, self.cfg.page_tokens, &ends) {
            pages.push(pk);
            if self.objs.contains_key(&pk) {
                shared.push(pk);
            } else {
                todo.push((pk, self.layout.segments(st, start, end), self.layout.page_bytes(end - start)));
            }
        }
        let rec = (self.layout.rec_bytes > 0).then(|| key ^ (u128::MAX >> 1));
        if let Some(rk) = rec {
            todo.push((rk, self.layout.rec_segments(st), self.layout.rec_bytes));
        }
        let head = if head_ok { Some(self.head_todo(st, key, &pages, len, &mut shared, &mut todo)) } else { None };
        let t_keys = t0.elapsed();
        let (mems, cap) = self.capture_all(backend, &todo)?;
        let t_cap = t0.elapsed();
        // only now (the copies worked) does the entry take its references
        for pk in shared {
            let o = self.objs.get_mut(&pk).unwrap();
            o.refs += 1;
            o.pins += pinned as u32;
            o.last_use = now;
        }
        for ((k, _, bytes), mem) in todo.iter().zip(mems) {
            copied += *bytes as u64;
            self.add_obj(*k, *bytes, mem, now, pinned);
        }
        self.entries.insert(
            key,
            Entry {
                tokens: Arc::new(tokens.to_vec()),
                ends,
                rec,
                pages,
                head,
                pinned: pin.map(String::from),
                last_use: now,
                hits: 0,
                created: Instant::now(),
            },
        );
        self.lens.insert(len);
        let t_add = t0.elapsed();
        self.write_manifest(key);
        let t_man = t0.elapsed();
        self.enforce(backend);
        self.stats.inserts += 1;
        self.stats.mtp_head_inserts += head_ok as u64;
        self.stats.insert_bytes += copied;
        self.stats.insert_seconds_total += t0.elapsed().as_secs_f64();
        if debug() {
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            eprintln!(
                "prefix store: insert {len} tok: {} objs, {} segments, {:.1} MiB in {:.2} ms (keys {:.2}, capture {:.2} [drain {:.2}, alloc {:.2}, copies {:.2}, sync {:.2}], add {:.2}, manifest {:.2}, enforce {:.2}); {}",
                todo.len(),
                cap.segments,
                copied as f64 / (1u64 << 20) as f64,
                ms(t0.elapsed()),
                ms(t_keys),
                ms(t_cap - t_keys),
                cap.drain_s * 1e3,
                cap.alloc_s * 1e3,
                cap.copy_s * 1e3,
                cap.sync_s * 1e3,
                ms(t_add - t_cap),
                ms(t_man - t_add),
                ms(t0.elapsed() - t_man),
                cap.how
            );
        }
        self.refresh_counts();
        Ok(copied)
    }

    /// The head objects of the entry `key` whose main pages are `pages` (`len` tokens): their keys, and the capture of each
    /// one no other entry holds yet (pushed to `todo`; the others to `shared`).
    fn head_todo<'s>(
        &self,
        st: &'s SequenceState,
        key: Key,
        pages: &[Key],
        len: usize,
        shared: &mut Vec<Key>,
        todo: &mut Vec<(Key, Vec<(&'s Buffer, usize, usize)>, usize)>,
    ) -> HeadRef {
        let h = self.head.as_ref().expect("checked by the caller");
        let m = st.mtp.as_ref().expect("checked by the caller");
        let page = self.cfg.page_tokens;
        let mut keys = Vec::with_capacity(pages.len());
        for (i, pk) in pages.iter().enumerate() {
            let (start, end) = (i * page, ((i + 1) * page).min(len));
            let hk = pk ^ HEAD_PAGE_SALT;
            keys.push(hk);
            if self.objs.contains_key(&hk) {
                shared.push(hk);
            } else {
                todo.push((hk, h.segments(m, start, end), h.page_bytes(start, end)));
            }
        }
        let tail = key ^ HEAD_TAIL_SALT;
        todo.push((tail, vec![(&m.pending, 0, h.row_bytes)], h.row_bytes));
        HeadRef { pages: keys, tail }
    }

    /// Copy each object's segments out of the state into new objects in the highest tier with room: on the device one
    /// allocation for all of them (device-to-device copies, nothing waits), else one host buffer each.
    fn capture_all(&mut self, backend: &dyn Backend, todo: &[(Key, Vec<(&Buffer, usize, usize)>, usize)]) -> Result<(Vec<Mem>, CaptureTimes)> {
        let mut cap = CaptureTimes::default();
        let total: usize = todo.iter().map(|t| t.2).sum();
        if total == 0 {
            return Ok((Vec::new(), cap));
        }
        cap.segments = todo.iter().map(|t| t.1.len()).sum();
        cap.drain(backend)?;
        let defer = self.defer && self.defer_fits(total);
        if defer && !self.use_slab() {
            self.deferred += total as u64;
        }
        if !defer && self.cfg.device_bytes >= total as u64 {
            // a real device tier: one allocation per object, so demoting one frees its memory
            cap.how = "device tier";
            let mut out = Vec::with_capacity(todo.len());
            for (_, segs, bytes) in todo {
                let t = Instant::now();
                let buf = backend.alloc((*bytes).max(16), MemKind::Device)?;
                cap.alloc_s += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let mut off = 0;
                for &(src, so, n) in segs {
                    backend.copy(&buf, off, src, so, n, 0)?;
                    off += n;
                }
                cap.copy_s += t.elapsed().as_secs_f64();
                out.push(Mem::Device(buf, 0));
            }
            cap.sync(backend)?;
            return Ok((out, cap));
        }
        if defer {
            // deferred: carved from a slab (no device tier) or one allocation per insert; all of it leaves the device at `settle`
            cap.how = "deferred";
            let t = Instant::now();
            let (buf, base) = if self.use_slab() { self.slab(backend, total)? } else { (backend.alloc(total.max(16), MemKind::Device)?, 0) };
            cap.alloc_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut out = Vec::with_capacity(todo.len());
            let mut off = base;
            for (_, segs, _) in todo {
                out.push(Mem::Device(buf.clone(), off));
                for &(src, so, n) in segs {
                    backend.copy(&buf, off, src, so, n, 0)?;
                    off += n;
                }
            }
            cap.copy_s += t.elapsed().as_secs_f64();
            cap.sync(backend)?;
            return Ok((out, cap));
        }
        cap.how = "host";
        let mut out = Vec::with_capacity(todo.len());
        for (_, segs, bytes) in todo {
            let t = Instant::now();
            let mut v = self.take_host_buf(*bytes);
            cap.alloc_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut off = 0;
            for &(src, so, n) in segs {
                backend.download(src, so, &mut v[off..off + n], 0)?;
                off += n;
            }
            cap.copy_s += t.elapsed().as_secs_f64();
            out.push(Mem::Host(Arc::new(v)));
        }
        Ok((out, cap))
    }

    /// Deferred captures are carved from slabs only without a device tier (`device_mb` 0: all of them leave the device at
    /// `settle`, so a slab is idle again right after). With one, survivors on the device would pin whole slabs, so each
    /// insert gets its own allocation (freed with its last object).
    fn use_slab(&self) -> bool {
        self.cfg.device_bytes == 0
    }

    /// Device bytes the deferred captures of this request hold (slabs allocated, not just carved, or per-insert allocations).
    fn deferred_footprint(&self) -> u64 {
        if self.use_slab() {
            self.slabs_full.iter().map(|b| b.bytes() as u64).sum::<u64>() + self.slab.as_ref().map_or(0, |(b, _)| b.bytes() as u64)
        } else {
            self.deferred
        }
    }

    /// Size of a new slab for a capture of `bytes` when `held` bytes of slabs are in use: [`SLAB_MIN`] (less when the cap is
    /// near), or exactly the capture when it is larger.
    fn new_slab_size(bytes: usize, held: u64) -> usize {
        let room = DEFER_CAP.saturating_sub(held) as usize;
        bytes.max(SLAB_MIN.min(room)).next_multiple_of(256)
    }

    /// Whether a deferred capture of `bytes` keeps the request's device memory within [`DEFER_CAP`].
    fn defer_fits(&self, bytes: usize) -> bool {
        if !self.use_slab() {
            return self.deferred + bytes as u64 <= DEFER_CAP;
        }
        if self.slab.as_ref().is_some_and(|(b, used)| *used + bytes <= b.bytes()) {
            return true;
        }
        let held = self.deferred_footprint();
        let fresh = match self.pooled_slab(bytes) {
            Some(i) => self.slab_pool[i].bytes(),
            None => Self::new_slab_size(bytes, held),
        };
        held + fresh as u64 <= DEFER_CAP
    }

    /// The idle slab a capture of `bytes` takes: the smallest that fits (what [`PrefixStore::defer_fits`] budgets with, so
    /// a larger idle slab cannot push the request past [`DEFER_CAP`]).
    fn pooled_slab(&self, bytes: usize) -> Option<usize> {
        self.slab_pool.iter().enumerate().filter(|(_, b)| b.bytes() >= bytes).min_by_key(|(_, b)| b.bytes()).map(|(i, _)| i)
    }

    /// `bytes` of device memory for a deferred capture: `(slab, offset)`. The current slab while it has room, else an idle
    /// one from the pool, else a new allocation. Everything carved from a slab is demoted at `settle`, after which the
    /// slab goes back to the pool ([`PrefixStore::recycle_slabs`]); its stale bytes are never read (every capture writes
    /// its whole range, and a restore reads only what a capture wrote).
    fn slab(&mut self, backend: &dyn Backend, bytes: usize) -> Result<(Buffer, usize)> {
        if let Some((buf, used)) = &mut self.slab {
            if *used + bytes <= buf.bytes() {
                let at = *used;
                *used = (at + bytes).next_multiple_of(256);
                return Ok((buf.clone(), at));
            }
        }
        let buf = match self.pooled_slab(bytes) {
            Some(i) => self.slab_pool.swap_remove(i),
            None => backend.alloc(Self::new_slab_size(bytes, self.deferred_footprint()), MemKind::Device)?,
        };
        if let Some((old, _)) = self.slab.take() {
            self.slabs_full.push(old);
        }
        self.slab = Some((buf.clone(), bytes.next_multiple_of(256)));
        Ok((buf, 0))
    }

    /// After a settle: the slabs nothing but the store holds any more become idle (the first ones, up to the pool cap, are
    /// kept; the rest are freed). A slab that a device-resident object still points into is left to it.
    fn recycle_slabs(&mut self) {
        let mut all: Vec<Buffer> = std::mem::take(&mut self.slabs_full);
        all.extend(self.slab.take().map(|(b, _)| b));
        let mut kept: usize = self.slab_pool.iter().map(Buffer::bytes).sum();
        for b in all {
            if b.strong_count() == 1 && kept + b.bytes() <= SLAB_POOL_CAP {
                kept += b.bytes();
                self.slab_pool.push(b);
            }
        }
    }

    fn add_obj(&mut self, k: Key, bytes: usize, mem: Mem, now: u64, pinned: bool) {
        let tier = match &mem {
            Mem::Device(..) => Tier::Device,
            Mem::Host(_) => Tier::Host,
        };
        self.used[tier_idx(tier)] += bytes as u64;
        let host = match &mem {
            Mem::Host(v) => Some(v.clone()),
            Mem::Device(..) => None,
        };
        let mut o = Obj { bytes, mem: Some(mem), disk: None, staged: None, last_use: now, refs: 1, pins: pinned as u32 };
        if let Some(v) = host {
            // write-through: host entries survive a restart
            self.queue_write(k, &mut o, v);
        }
        self.objs.insert(k, o);
    }

    /// Restore entry `hit` into `st` (which is reset first; on an error it is left reset). The entry's objects are
    /// checked before the state is touched where they can be (disk files: header, build, checksum).
    pub fn restore(&mut self, backend: &dyn Backend, st: &mut SequenceState, hit: Hit) -> Result<Restored> {
        self.poll_writer();
        self.layout.check(st)?;
        let t0 = Instant::now();
        let Some(e) = self.entries.get(&hit.key) else {
            return Err(Error::invalid("prefix store: entry gone"));
        };
        if e.tokens.len() != hit.len || hit.len > st.cap {
            return Err(Error::invalid("prefix store: entry does not fit the state"));
        }
        let tokens = e.tokens.clone();
        let pages = e.pages.clone();
        let rec = e.rec;
        // the head's part only into a head of this layout (a state without the head restores the main model alone)
        let head = e.head.clone().filter(|_| self.head.as_ref().is_some_and(|h| st.mtp.as_ref().is_some_and(|m| h.fits(m))));
        let head_keys: Vec<Key> = head.iter().flat_map(|h| h.pages.iter().chain(std::iter::once(&h.tail))).copied().collect();
        // load what is on disk only, verified, before anything is written
        let mut loaded: HashMap<Key, Arc<HostBuf>> = HashMap::new();
        for k in rec.iter().chain(pages.iter()).chain(head_keys.iter()) {
            let o = self.objs.get(k).ok_or_else(|| Error::invalid("prefix store: object gone"))?;
            if o.mem.is_none() {
                let data = match &o.staged {
                    Some(s) => Ok(s.clone()),
                    None => match &o.disk {
                        Some(p) => read_obj(p, self.build, *k, o.bytes).map(|v| Arc::new(v.into())),
                        None => Err(Error::invalid("prefix store: object without a copy")),
                    },
                };
                match data {
                    Ok(d) => {
                        loaded.insert(*k, d);
                    }
                    Err(err) => {
                        self.stats.rejected += 1;
                        self.drop_obj(*k);
                        self.refresh_counts();
                        return Err(err);
                    }
                }
            }
        }
        st.truncate(0, backend)?;
        let mut bytes = 0u64;
        let mut slowest: Option<Tier> = None;
        let page = self.cfg.page_tokens;
        let now = self.tick();
        for (i, k) in pages.iter().enumerate() {
            let start = i * page;
            let end = ((i + 1) * page).min(hit.len);
            let segs = self.layout.segments(st, start, end);
            let o = self.objs.get_mut(k).unwrap();
            o.last_use = now;
            let tier = o.tier();
            slowest = slowest.max(Some(tier));
            put(backend, o, loaded.get(k), &segs)?;
            bytes += o.bytes as u64;
        }
        if let Some(k) = rec {
            let segs = self.layout.rec_segments(st);
            let o = self.objs.get_mut(&k).unwrap();
            o.last_use = now;
            slowest = slowest.max(Some(o.tier()));
            put(backend, o, loaded.get(&k), &segs)?;
            bytes += o.bytes as u64;
        }
        if let (Some(hr), Some(h), Some(m)) = (&head, &self.head, st.mtp.as_ref()) {
            let mut segs: Vec<(Vec<(&Buffer, usize, usize)>, Key)> = Vec::with_capacity(hr.pages.len() + 1);
            for (i, k) in hr.pages.iter().enumerate() {
                segs.push((h.segments(m, i * page, ((i + 1) * page).min(hit.len)), *k));
            }
            segs.push((vec![(&m.pending, 0, h.row_bytes)], hr.tail));
            for (sg, k) in segs {
                let o = self.objs.get_mut(&k).unwrap();
                o.last_use = now;
                slowest = slowest.max(Some(o.tier()));
                put(backend, o, loaded.get(&k), &sg)?;
                bytes += o.bytes as u64;
            }
        }
        backend.synchronize(0)?;
        st.load_prefix(&tokens);
        if head.is_some() {
            st.mtp_load_head();
            self.stats.mtp_head_restores += 1;
        }
        // what came from disk goes back to host memory (it is hot again)
        for (k, d) in loaded {
            if let Some(o) = self.objs.get_mut(&k) {
                if o.mem.is_none() {
                    self.used[tier_idx(Tier::Host)] += o.bytes as u64;
                    o.mem = Some(Mem::Host(d));
                }
            }
        }
        if let Some(e) = self.entries.get_mut(&hit.key) {
            e.hits += 1;
            e.last_use = now;
        }
        self.enforce(backend);
        let elapsed = t0.elapsed();
        self.stats.hits += 1;
        self.stats.bytes_restored += bytes;
        self.stats.tokens_saved += hit.len as u64;
        self.stats.restore_seconds_total += elapsed.as_secs_f64();
        self.stats.last_restore_ms = elapsed.as_secs_f64() * 1e3;
        if self.stats.prefill_tok_s > 0.0 {
            self.stats.ttft_saved_seconds_total += hit.len as f64 / self.stats.prefill_tok_s - elapsed.as_secs_f64();
        }
        match slowest {
            Some(Tier::Device) => self.stats.device.hits += 1,
            Some(Tier::Host) => self.stats.host.hits += 1,
            Some(Tier::Disk) => self.stats.disk.hits += 1,
            None => {}
        }
        self.refresh_counts();
        Ok(Restored { len: hit.len, bytes, elapsed, tier: slowest })
    }

    /// Start or stop deferring device-to-host copies (see `defer`).
    pub fn set_defer(&mut self, on: bool) {
        self.defer = on;
    }

    pub fn deferring(&self) -> bool {
        self.defer
    }

    /// Stop deferring and bring every tier within budget (device captures move to host, and to disk when it is on).
    /// `site` says where the engine called it (only for the `INTARSIA_PREFIX_STORE_DEBUG` line).
    pub fn settle(&mut self, backend: &dyn Backend, site: &str) {
        self.defer = false;
        self.deferred = 0;
        self.queue_write_seconds = 0.0;
        self.poll_writer();
        let t0 = Instant::now();
        let before = self.used[0];
        self.enforce(backend);
        self.recycle_slabs();
        let took = t0.elapsed().as_secs_f64();
        self.stats.insert_seconds_total += took;
        self.stats.settle_seconds_total += took;
        if debug() {
            eprintln!(
                "prefix store: settle ({site}) moved {} MiB off the device in {:.1} ms (queue_write {:.2} ms)",
                (before - self.used[0]) >> 20,
                took * 1e3,
                self.queue_write_seconds * 1e3
            );
        }
        self.refresh_counts();
    }

    /// [`PrefixStore::settle`] in slices (lane interleave, the batched engine between decode steps): demote device objects, whatever
    /// `defer` says, until at least `budget` bytes moved or none is left over the device budget. Returns true once nothing is
    /// left, after a full [`PrefixStore::settle`] (the same end state); false: call again. Full slabs whose captures have all
    /// moved go back to the pool on the way, so the deferred footprint ([`PrefixStore::deferred_bytes`]) shrinks with each slice.
    pub fn settle_some(&mut self, backend: &dyn Backend, budget: u64, site: &str) -> bool {
        self.poll_writer();
        let t0 = Instant::now();
        let mut moved = 0u64;
        while moved < budget.max(1) && self.used[0] > self.cfg.device_bytes {
            match self.demote_one(backend) {
                Some(b) => moved += b,
                None => break,
            }
        }
        if self.used[0] <= self.cfg.device_bytes || moved == 0 {
            self.settle(backend, site);
            return true;
        }
        self.deferred = self.deferred.saturating_sub(moved);
        let full = std::mem::take(&mut self.slabs_full);
        let mut kept: usize = self.slab_pool.iter().map(Buffer::bytes).sum();
        for b in full {
            if b.strong_count() > 1 {
                self.slabs_full.push(b);
            } else if kept + b.bytes() <= SLAB_POOL_CAP {
                kept += b.bytes();
                self.slab_pool.push(b);
            }
        }
        let took = t0.elapsed().as_secs_f64();
        self.stats.insert_seconds_total += took;
        self.stats.settle_seconds_total += took;
        if debug() {
            eprintln!("prefix store: settle slice ({site}) moved {} MiB off the device in {:.1} ms", moved >> 20, took * 1e3);
        }
        self.refresh_counts();
        false
    }

    /// Device bytes the deferred captures hold (what [`DEFER_CAP`] bounds).
    pub fn deferred_bytes(&self) -> u64 {
        self.deferred_footprint()
    }

    /// Feed the cold prefill rate the TTFT-saved estimate uses (tokens computed, time they took).
    pub fn note_prefill(&mut self, tokens: usize, elapsed: Duration) {
        let s = elapsed.as_secs_f64();
        if tokens < 64 || s <= 0.0 {
            return;
        }
        let r = tokens as f64 / s;
        let p = &mut self.stats.prefill_tok_s;
        *p = if *p == 0.0 { r } else { 0.8 * *p + 0.2 * r };
    }

    pub fn snapshot(&mut self) -> PrefixStoreSnapshot {
        self.poll_writer();
        self.refresh_counts();
        self.stats.clone()
    }

    pub fn entries(&self) -> Vec<EntryInfo> {
        let mut v: Vec<(u64, EntryInfo)> = self
            .entries
            .iter()
            .map(|(k, e)| {
                let objs = e.objs().filter_map(|k| self.objs.get(k));
                let (mut tier, mut bytes) = (Tier::Device, 0u64);
                for o in objs {
                    tier = tier.max(o.tier());
                    bytes += o.bytes as u64;
                }
                (
                    e.last_use,
                    EntryInfo {
                        key: format!("{k:032x}"),
                        tokens: e.tokens.len(),
                        tier,
                        bytes,
                        pinned: e.pinned.clone(),
                        hits: e.hits,
                        age_s: e.created.elapsed().as_secs_f64(),
                    },
                )
            })
            .collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v.into_iter().map(|x| x.1).collect()
    }

    fn refresh_counts(&mut self) {
        let mut n = [0u64; 3];
        for o in self.objs.values() {
            if o.mem.is_some() {
                n[tier_idx(o.tier())] += 1;
            }
            if o.disk.is_some() {
                n[2] += 1;
            }
        }
        self.stats.entries = self.entries.len() as u64;
        self.stats.disk_pending_bytes = self.pending_write_bytes;
        self.stats.pinned_entries = self.entries.values().filter(|e| e.pinned.is_some()).count() as u64;
        self.stats.device.objects = n[0];
        self.stats.host.objects = n[1];
        self.stats.disk.objects = n[2];
        self.stats.device.bytes = self.used[0];
        self.stats.host.bytes = self.used[1];
        self.stats.disk.bytes = self.used[2];
    }

    // ------------------------------------------------------------------------------------------------- eviction

    /// Bring every tier within its budget: device objects move to host, host objects to disk (or are dropped), disk
    /// objects are deleted. Pinned objects never leave memory.
    fn enforce(&mut self, backend: &dyn Backend) {
        // confirmed writes first: they lower the pending bytes the demotes below are queued against
        self.poll_writer();
        while self.used[0] > self.cfg.device_bytes && !self.defer {
            if self.demote_one(backend).is_none() {
                break;
            }
        }
        // idle pooled buffers go first; then host objects, oldest first. One whose write the disk writer has not confirmed
        // stays (the writer holds its buffer, so evicting it would free nothing): the tier can be over budget until the writer
        // confirms it (polled at the next insert / settle) and a later enforce evicts it
        while self.used[1] + self.host_free_bytes > self.cfg.host_bytes {
            let Some(v) = self.host_free.pop() else { break };
            self.host_free_bytes -= v.len() as u64;
        }
        while self.used[1] > self.cfg.host_bytes {
            let Some(k) = self.lru(|o| matches!(o.mem, Some(Mem::Host(_))) && o.pins == 0 && o.staged.is_none()) else { break };
            self.evict_host(k);
        }
        self.enforce_disk();
    }

    /// Bring the disk tier within budget, oldest first. An object whose write is still queued has no file yet: the writer
    /// creates it later, and [`PrefixStore::poll_writer`] deletes it then (the object no longer claims it).
    fn enforce_disk(&mut self) {
        while self.used[2] > self.cfg.disk_bytes {
            let Some(k) = self.lru(|o| o.disk.is_some() && o.pins == 0) else { break };
            self.stats.disk.evictions += 1;
            let o = self.objs.get_mut(&k).unwrap();
            if let Some(p) = o.disk.take() {
                let _ = std::fs::remove_file(p);
            }
            o.staged = None;
            self.used[2] -= o.bytes as u64;
            if o.mem.is_none() {
                self.drop_obj(k);
            }
        }
    }

    /// Move the least recently used device object to host (and queue its disk write): the bytes it moved, or `None` when no
    /// device object is left.
    fn demote_one(&mut self, backend: &dyn Backend) -> Option<u64> {
        let k = self.lru(|o| matches!(o.mem, Some(Mem::Device(..))))?;
        let ta = Instant::now();
        // room first, while `k` is still registered: an entry the evictions drop gives its references back to it
        let (bytes, last_use) = self.objs.get(&k).map(|o| (o.bytes, o.last_use)).unwrap();
        self.make_host_room(bytes, last_use);
        let Some(mut o) = self.objs.remove(&k) else { return Some(0) }; // went with the entry that needed it
        let Some(Mem::Device(buf, boff)) = o.mem.take() else { unreachable!() };
        self.used[0] -= o.bytes as u64;
        self.stats.device.evictions += 1;
        let mut v = self.take_host_buf(o.bytes);
        let tb = Instant::now();
        let ok = backend.download(&buf, boff, &mut v, 0).is_ok();
        if debug() {
            eprintln!(
                "prefix store: demote {} KiB: alloc {:.2} ms, download {:.2} ms",
                o.bytes >> 10,
                (tb - ta).as_secs_f64() * 1e3,
                tb.elapsed().as_secs_f64() * 1e3
            );
        }
        if !ok {
            self.objs.insert(k, o);
            self.drop_obj(k);
            return Some(0);
        }
        self.used[1] += o.bytes as u64;
        let v = Arc::new(v);
        o.mem = Some(Mem::Host(v.clone()));
        if o.disk.is_none() {
            self.queue_write(k, &mut o, v);
        }
        let moved = o.bytes as u64;
        self.objs.insert(k, o);
        Some(moved)
    }

    /// Take `k`'s host copy away: its disk copy stays (when it has one), else the object goes; the buffer is kept for reuse.
    fn evict_host(&mut self, k: Key) {
        self.stats.host.evictions += 1;
        let o = self.objs.get_mut(&k).unwrap();
        if o.disk.is_some() {
            let bytes = o.bytes as u64;
            let mem = o.mem.take();
            self.used[1] -= bytes;
            if let Some(Mem::Host(v)) = mem {
                self.release_host(v);
            }
        } else {
            self.drop_obj(k);
        }
    }

    /// Before a demote of `bytes` (an object last used at `last_use`): evict the host objects that are older than it, oldest
    /// first, until it fits the host budget, so its buffer can be the evicted one. What does not fit this way is left to the
    /// loop after the demotes, which evicts in the same order.
    fn make_host_room(&mut self, bytes: usize, last_use: u64) {
        while self.used[1] + bytes as u64 > self.cfg.host_bytes {
            let Some(k) = self.lru(|o| matches!(o.mem, Some(Mem::Host(_))) && o.pins == 0 && o.staged.is_none() && o.last_use < last_use)
            else {
                break;
            };
            self.evict_host(k);
        }
    }

    /// Keep a host buffer nobody else holds for [`PrefixStore::take_host_buf`], if it fits the host budget.
    fn release_host(&mut self, v: Arc<HostBuf>) {
        if let Ok(v) = Arc::try_unwrap(v) {
            if self.used[1] + self.host_free_bytes + v.len() as u64 <= self.cfg.host_bytes {
                self.host_free_bytes += v.len() as u64;
                self.host_free.push(v);
            }
        }
    }

    /// A host buffer of `n` bytes, every page touched: a recycled one (stale bytes: the caller overwrites all of it), else
    /// a new one ([`host_buf`]) after giving back pooled buffers the budget needs.
    fn take_host_buf(&mut self, n: usize) -> HostBuf {
        if let Some(i) = self.host_free.iter().position(|v| v.len() == n) {
            self.host_free_bytes -= n as u64;
            return self.host_free.swap_remove(i);
        }
        while self.used[1] + self.host_free_bytes + n as u64 > self.cfg.host_bytes {
            let Some(v) = self.host_free.pop() else { break };
            self.host_free_bytes -= v.len() as u64;
        }
        new_host_buf(n, host_prefault())
    }

    fn lru(&self, f: impl Fn(&Obj) -> bool) -> Option<Key> {
        self.objs.iter().filter(|(_, o)| f(o)).min_by_key(|(_, o)| o.last_use).map(|(k, _)| *k)
    }

    /// Drop an object from every tier, and every entry that needs it (and their objects nothing else needs).
    fn drop_obj(&mut self, k: Key) {
        let dead: Vec<Key> = self
            .entries
            .iter()
            .filter(|(_, e)| e.objs().any(|x| *x == k))
            .map(|(ek, _)| *ek)
            .collect();
        for ek in dead {
            self.remove_entry(ek);
        }
        if let Some(o) = self.objs.remove(&k) {
            self.release_obj(o);
        }
    }

    fn release_obj(&mut self, o: Obj) {
        match o.mem {
            Some(Mem::Device(..)) => self.used[0] -= o.bytes as u64,
            Some(Mem::Host(v)) => {
                self.used[1] -= o.bytes as u64;
                self.release_host(v);
            }
            None => {}
        }
        if let Some(p) = &o.disk {
            self.used[2] -= o.bytes as u64;
            let _ = std::fs::remove_file(p);
        }
    }

    fn remove_entry(&mut self, ek: Key) {
        let Some(e) = self.entries.remove(&ek) else { return };
        let len = e.tokens.len();
        if !self.entries.values().any(|x| x.tokens.len() == len) {
            self.lens.remove(&len);
        }
        if let Some(w) = &self.writer {
            let _ = std::fs::remove_file(w.dir.join(format!("e-{ek:032x}.entry")));
        }
        for k in e.objs() {
            let gone = match self.objs.get_mut(k) {
                Some(o) => {
                    o.refs = o.refs.saturating_sub(1);
                    if e.pinned.is_some() {
                        o.pins = o.pins.saturating_sub(1);
                    }
                    o.refs == 0
                }
                None => false,
            };
            if gone {
                let o = self.objs.remove(k).unwrap();
                self.release_obj(o);
            }
        }
    }

    // ------------------------------------------------------------------------------------------------- disk

    fn queue_write(&mut self, k: Key, o: &mut Obj, data: Arc<HostBuf>) {
        let Some(w) = &self.writer else { return };
        if self.cfg.disk_bytes < o.bytes as u64 {
            return;
        }
        // a disk slower than the inserts: past `host_bytes` of unconfirmed writes the object stays host-only (one write is
        // always let through, so an object larger than the limit is still written when the queue is empty)
        if self.pending_write_bytes > 0 && self.pending_write_bytes + o.bytes as u64 > self.cfg.host_bytes {
            self.stats.disk_write_skips += 1;
            return;
        }
        let t0 = Instant::now();
        let path = w.dir.join(format!("o-{k:032x}.bin"));
        // the header (its checksum reads every byte of `data`) is built by the writer thread, off the engine thread
        if w.tx.send((k, path.clone(), data.clone())).is_ok() {
            o.disk = Some(path);
            o.staged = Some(data);
            self.used[2] += o.bytes as u64;
            self.pending_write_bytes += o.bytes as u64;
        }
        let took = t0.elapsed().as_secs_f64();
        self.queue_write_seconds += took;
        if debug() {
            eprintln!("prefix store: queue_write {} KiB: {:.3} ms", o.bytes >> 10, took * 1e3);
        }
    }

    fn poll_writer(&mut self) {
        let Some(w) = &self.writer else { return };
        let done: Vec<(Key, bool, usize)> = w.done.try_iter().collect();
        let dir = w.dir.clone();
        for (k, ok, bytes) in done {
            self.pending_write_bytes = self.pending_write_bytes.saturating_sub(bytes as u64);
            let mut lost = false;
            match self.objs.get_mut(&k) {
                Some(o) if o.disk.is_some() => {
                    o.staged = None;
                    if !ok {
                        o.disk = None;
                        self.used[2] -= o.bytes as u64;
                        lost = o.mem.is_none();
                    }
                }
                // evicted from disk or dropped while queued: nothing claims the file the writer just made
                _ => {
                    if ok {
                        let _ = std::fs::remove_file(dir.join(format!("o-{k:032x}.bin")));
                    }
                }
            }
            if lost {
                self.drop_obj(k);
            }
        }
    }

    /// Wait for the disk writer to finish what is queued (tests, shutdown).
    pub fn flush(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(60);
        self.poll_writer();
        while self.pending_write_bytes > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
            self.poll_writer();
        }
    }

    fn write_manifest(&self, ek: Key) {
        let (Some(w), Some(e)) = (&self.writer, self.entries.get(&ek)) else { return };
        let mut body = serde_json::json!({
            "key": format!("{ek:032x}"),
            "rec": e.rec.map(|k| format!("{k:032x}")),
            "pages": e.pages.iter().map(|k| format!("{k:032x}")).collect::<Vec<_>>(),
            "pinned": e.pinned,
            "tokens": &*e.tokens,
            "ends": e.ends,
        });
        // only with the head's part: a manifest without it is what it was before lane mhkv
        if let Some(h) = &e.head {
            body["head"] = serde_json::json!({
                "pages": h.pages.iter().map(|k| format!("{k:032x}")).collect::<Vec<_>>(),
                "tail": format!("{:032x}", h.tail),
            });
        }
        let body = serde_json::to_vec(&body).unwrap_or_default();
        let mut f = Vec::with_capacity(body.len() + 32);
        f.extend_from_slice(ENTRY_MAGIC);
        f.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        f.extend_from_slice(&self.build.to_le_bytes());
        f.extend_from_slice(&checksum(&body).to_le_bytes());
        f.extend_from_slice(&body);
        let _ = write_atomic(&w.dir.join(format!("e-{ek:032x}.entry")), &[&f]);
    }

    /// Register the entries a previous run of this build left on disk (their objects stay on disk until used).
    fn load_disk(&mut self) {
        let Some(dir) = self.writer.as_ref().map(|w| w.dir.clone()) else { return };
        let Ok(rd) = std::fs::read_dir(&dir) else { return };
        let mut manifests: Vec<(std::time::SystemTime, PathBuf)> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "entry"))
            .map(|p| (std::fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), p))
            .collect();
        manifests.sort();
        for (_, p) in manifests {
            match self.load_manifest(&dir, &p) {
                Ok(true) => {}
                // an entry whose objects never reached the disk (they were in device memory, or dropped first)
                Ok(false) => {
                    let _ = std::fs::remove_file(&p);
                }
                Err(_) => {
                    self.stats.rejected += 1;
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
        // drop objects no manifest references (an interrupted run)
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for p in rd.filter_map(|e| e.ok()).map(|e| e.path()) {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if let Some(hex) = name.strip_prefix("o-").and_then(|n| n.strip_suffix(".bin")) {
                    if u128::from_str_radix(hex, 16).map(|k| !self.objs.contains_key(&k)).unwrap_or(true) {
                        let _ = std::fs::remove_file(&p);
                    }
                } else if name.ends_with(".tmp") {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
        // disk budget: the oldest entries go first
        while self.used[2] > self.cfg.disk_bytes {
            let Some(k) = self.lru(|o| o.disk.is_some()) else { break };
            self.drop_obj(k);
        }
        self.refresh_counts();
    }

    /// `Ok(false)`: a valid manifest whose object files are not all there.
    fn load_manifest(&mut self, dir: &Path, p: &Path) -> Result<bool> {
        let f = std::fs::read(p)?;
        let bad = || Error::invalid(format!("prefix store: bad manifest {}", p.display()));
        if f.len() < 24 || &f[..4] != ENTRY_MAGIC {
            return Err(bad());
        }
        let version = u32::from_le_bytes(f[4..8].try_into().unwrap());
        let build = u64::from_le_bytes(f[8..16].try_into().unwrap());
        let sum = u64::from_le_bytes(f[16..24].try_into().unwrap());
        let body = &f[24..];
        if version != FORMAT_VERSION || build != self.build || checksum(body) != sum {
            return Err(bad());
        }
        let v: serde_json::Value = serde_json::from_slice(body).map_err(|_| bad())?;
        let hex = |x: &serde_json::Value| x.as_str().and_then(|s| u128::from_str_radix(s, 16).ok());
        let key = hex(&v["key"]).ok_or_else(bad)?;
        let tokens: Vec<u32> = serde_json::from_value(v["tokens"].clone()).map_err(|_| bad())?;
        if entry_key(self.build, &tokens) != key {
            return Err(bad());
        }
        let rec = if v["rec"].is_null() { None } else { Some(hex(&v["rec"]).ok_or_else(bad)?) };
        let pages: Vec<Key> = v["pages"].as_array().ok_or_else(bad)?.iter().map(|x| hex(x).ok_or_else(bad)).collect::<Result<_>>()?;
        let ends: Vec<usize> = serde_json::from_value(v["ends"].clone()).map_err(|_| bad())?;
        let expect: Vec<(Key, usize, usize)> = page_keys(self.build, &tokens, self.cfg.page_tokens, &ends);
        if expect.len() != pages.len() || expect.iter().zip(&pages).any(|(a, b)| a.0 != *b) || rec.is_some() != (self.layout.rec_bytes > 0) {
            return Err(bad());
        }
        let pinned = v["pinned"].as_str().map(String::from);
        // the head's part: its keys follow from the entry's (a store without a head layout has no manifest with one: the
        // build key differs)
        let head = if v.get("head").is_none() {
            None
        } else {
            let h = self.head.as_ref().ok_or_else(bad)?;
            let hp: Vec<Key> = v["head"]["pages"].as_array().ok_or_else(bad)?.iter().map(|x| hex(x).ok_or_else(bad)).collect::<Result<_>>()?;
            let tail = hex(&v["head"]["tail"]).ok_or_else(bad)?;
            if hp.len() != pages.len() || hp.iter().zip(&pages).any(|(a, b)| *a != b ^ HEAD_PAGE_SALT) || tail != key ^ HEAD_TAIL_SALT || tokens.len() < 2 {
                return Err(bad());
            }
            let sizes: Vec<usize> = expect.iter().map(|x| h.page_bytes(x.1, x.2)).chain(std::iter::once(h.row_bytes)).collect();
            Some((HeadRef { pages: hp, tail }, sizes))
        };
        // every object file must exist with the right size (contents are checked when restored)
        let mut sizes = Vec::new();
        let head_objs: Vec<(Key, usize)> = match &head {
            Some((hr, hs)) => hr.pages.iter().chain(std::iter::once(&hr.tail)).copied().zip(hs.iter().copied()).collect(),
            None => Vec::new(),
        };
        let main = rec.iter().chain(pages.iter()).enumerate().map(|(i, k)| {
            let want = if rec.is_some() && i == 0 {
                self.layout.rec_bytes
            } else {
                let pi = if rec.is_some() { i - 1 } else { i };
                self.layout.page_bytes(expect[pi].2 - expect[pi].1)
            };
            (*k, want)
        });
        let all: Vec<(Key, usize)> = main.chain(head_objs).collect();
        for (k, want) in all {
            let k = &k;
            if !self.objs.contains_key(k) {
                let op = dir.join(format!("o-{k:032x}.bin"));
                let Ok(m) = std::fs::metadata(&op) else { return Ok(false) };
                if m.len() != (OBJ_HEADER + want) as u64 {
                    return Err(bad());
                }
            }
            sizes.push((*k, want));
        }
        let now = self.tick();
        for (k, bytes) in sizes {
            match self.objs.get_mut(&k) {
                Some(o) => {
                    o.refs += 1;
                    o.pins += pinned.is_some() as u32;
                }
                None => {
                    self.used[2] += bytes as u64;
                    self.objs.insert(
                        k,
                        Obj {
                            bytes,
                            mem: None,
                            disk: Some(dir.join(format!("o-{k:032x}.bin"))),
                            staged: None,
                            last_use: now,
                            refs: 1,
                            pins: pinned.is_some() as u32,
                        },
                    );
                }
            }
        }
        self.lens.insert(tokens.len());
        self.entries.insert(
            key,
            Entry { tokens: Arc::new(tokens), ends, rec, pages, head: head.map(|h| h.0), pinned, last_use: now, hits: 0, created: Instant::now() },
        );
        Ok(true)
    }
}

/// Write the object's bytes into the state segments: a device copy, an upload from host memory, or the bytes loaded from
/// disk.
fn put(backend: &dyn Backend, o: &Obj, loaded: Option<&Arc<HostBuf>>, segs: &[(&Buffer, usize, usize)]) -> Result<()> {
    let total: usize = segs.iter().map(|s| s.2).sum();
    if total != o.bytes {
        return Err(Error::invalid("prefix store: object size does not match the state layout"));
    }
    let mut off = 0;
    for &(dst, doff, n) in segs {
        match (&o.mem, loaded) {
            (Some(Mem::Device(b, boff)), _) => backend.copy(dst, doff, b, boff + off, n, 0)?,
            (Some(Mem::Host(v)), _) => backend.upload(dst, doff, &v[off..off + n], 0)?,
            (None, Some(v)) => backend.upload(dst, doff, &v[off..off + n], 0)?,
            (None, None) => return Err(Error::invalid("prefix store: object not loaded")),
        }
        off += n;
    }
    Ok(())
}

const OBJ_HEADER: usize = 4 + 4 + 8 + 16 + 8 + 8;

fn obj_header(build: u64, k: Key, data: &[u8]) -> Vec<u8> {
    let mut h = Vec::with_capacity(OBJ_HEADER);
    h.extend_from_slice(OBJ_MAGIC);
    h.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    h.extend_from_slice(&build.to_le_bytes());
    h.extend_from_slice(&k.to_le_bytes());
    h.extend_from_slice(&(data.len() as u64).to_le_bytes());
    h.extend_from_slice(&checksum(data).to_le_bytes());
    h
}

/// Read and check an object file: magic, format version, build, key, length and checksum must all match.
fn read_obj(p: &Path, build: u64, k: Key, bytes: usize) -> Result<Vec<u8>> {
    let f = std::fs::read(p)?;
    let bad = |why: &str| Error::invalid(format!("prefix store: rejected {} ({why})", p.display()));
    if f.len() != OBJ_HEADER + bytes || &f[..4] != OBJ_MAGIC {
        return Err(bad("size or magic"));
    }
    if u32::from_le_bytes(f[4..8].try_into().unwrap()) != FORMAT_VERSION {
        return Err(bad("format version"));
    }
    if u64::from_le_bytes(f[8..16].try_into().unwrap()) != build {
        return Err(bad("another build"));
    }
    if u128::from_le_bytes(f[16..32].try_into().unwrap()) != k {
        return Err(bad("key"));
    }
    if u64::from_le_bytes(f[32..40].try_into().unwrap()) != bytes as u64 {
        return Err(bad("length"));
    }
    let data = f[OBJ_HEADER..].to_vec();
    if checksum(&data) != u64::from_le_bytes(f[40..48].try_into().unwrap()) {
        return Err(bad("checksum"));
    }
    Ok(data)
}

fn write_atomic(path: &Path, parts: &[&[u8]]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        for p in parts {
            f.write_all(p)?;
        }
    }
    std::fs::rename(&tmp, path)
}

impl DiskWriter {
    fn spawn(dir: PathBuf, build: u64) -> Result<DiskWriter> {
        let (tx, rx) = channel::<(Key, PathBuf, Arc<HostBuf>)>();
        let (dtx, done) = channel();
        let thread = std::thread::Builder::new()
            .name("intarsia-prefix-store-writer".into())
            .spawn(move || run_writer(rx, dtx, build))
            .map_err(|e| Error::Other(format!("spawn prefix store writer: {e}")))?;
        Ok(DiskWriter { dir, tx, done, _thread: thread })
    }
}

/// The disk writer thread: each queued object written (header with its checksum, then the bytes) and confirmed in order.
fn run_writer(rx: Receiver<(Key, PathBuf, Arc<HostBuf>)>, dtx: Sender<(Key, bool, usize)>, build: u64) {
    let dbg = debug();
    for (k, path, data) in rx.iter() {
        let t0 = Instant::now();
        let header = obj_header(build, k, &data);
        let t1 = Instant::now();
        let ok = write_atomic(&path, &[&header, &data]).is_ok();
        if dbg {
            eprintln!(
                "prefix store: writer {} KiB: checksum {:.2} ms, write {:.2} ms{}",
                data.len() >> 10,
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3,
                if ok { "" } else { " (failed)" }
            );
        }
        if dtx.send((k, ok, data.len())).is_err() {
            break;
        }
    }
}

/// Everything that can change the bits a prefill computes, for [`PrefixStore::new`]: model id, recipe, dims, dtypes,
/// the resolved plan (kernels, fusions, knobs that reach it), backend fingerprint, chunk configuration, slot layout,
/// boundary policy, and the running executable (size and modification time: a rebuild never reads an older build's
/// entries).
pub fn describe_build(model: &crate::model::Model, model_id: &str, cfg: &PrefixStoreConfig) -> String {
    // the resolved kernels (which op runs on which kernel, fused or not): what the plan says about numerics, without its
    // memory numbers (free memory differs from run to run)
    let plan = crate::plan::plan_json(model, model_id);
    let plan = format!("{}{}", plan["ops"], plan["layers"]);
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| {
            let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
            format!("{}:{t}", m.len())
        })
        .unwrap_or_default();
    format!(
        "v{FORMAT_VERSION}|{model_id}|{:?}|{:?}|{}|{}|{}|{}|{}|{}|{}|{:?}|{exe}|w{}",
        model.recipe,
        model.dims,
        model.act.name(),
        model.kv.name(),
        model.backend.fingerprint(),
        model.prefill_chunk(),
        model.tail_merge(),
        cfg.page_tokens,
        plan,
        (&cfg.boundary_tokens, &cfg.header_end_tokens, &cfg.cut_roles),
        model.weights_bytes,
    ) + &(if model.gdn_state_dt != crate::DType::F32 { format!("|gdnstate{}", model.gdn_state_dt.name()) } else { String::new() })
        + &if model.prefill_peel() > 0 {
        // the period only when not the default 256, so the key of every model with a 256-period peel is unchanged
        match model.prefill_peel_period() {
            crate::recipe::DEFAULT_PEEL_PERIOD => format!("|peel{}", model.prefill_peel()),
            period => format!("|peel{}|period{period}", model.prefill_peel()),
        }
    } else {
        String::new()
    }
        // lane mhkv: entries of a model with an MTP head keep the head's state (another layout of objects and manifests); a
        // model without one keeps its key
        + if model.mtp_store_layout().is_some() { "|mtphead1" } else { "" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_spec() {
        assert_eq!(PrefixStoreConfig::parse("").unwrap(), None);
        assert_eq!(PrefixStoreConfig::parse("off").unwrap(), None);
        assert_eq!(PrefixStoreConfig::parse("on").unwrap(), Some(PrefixStoreConfig::default()));
        let c = PrefixStoreConfig::parse("host_mb=64,page=128,dir=/tmp/x").unwrap().unwrap();
        assert_eq!(c.host_bytes, 64 * MIB);
        assert_eq!(c.page_tokens, 128);
        assert_eq!(c.disk_bytes, 64 * 1024 * MIB);
        assert!(PrefixStoreConfig::parse("page=3").is_err());
        assert!(PrefixStoreConfig::parse("host_mb=99999999999").is_err());
        assert!(PrefixStoreConfig::parse("disk_mb=5").is_err());
        assert!(PrefixStoreConfig::parse("bogus=1").is_err());
        let b = PrefixStoreConfig::parse("boundary=<a>|<b>").unwrap().unwrap();
        assert_eq!(b.boundary_tokens, vec!["<a>".to_string(), "<b>".to_string()]);
        let r = PrefixStoreConfig::parse("on,roles=assistant|tool").unwrap().unwrap();
        assert_eq!(r.cut_roles, vec!["assistant".to_string(), "tool".to_string()]);
    }

    #[test]
    fn keys_depend_on_tokens_length_and_build() {
        let t: Vec<u32> = (0..100).collect();
        assert_eq!(entry_key(1, &t), entry_key(1, &t));
        assert_ne!(entry_key(1, &t), entry_key(2, &t));
        assert_ne!(entry_key(1, &t[..99]), entry_key(1, &t));
        let mut u = t.clone();
        u[50] = 7;
        assert_ne!(entry_key(1, &t), entry_key(1, &u));
        // pages: shared up to the first differing page, then not
        let g: Vec<usize> = (1..=6).map(|i| i * 16).chain([100]).collect();
        let pa = page_keys(1, &t, 16, &g);
        let pb = page_keys(1, &u, 16, &g);
        assert_eq!(pa.len(), 7);
        assert_eq!(pa[..3], pb[..3]);
        assert_ne!(pa[3].0, pb[3].0);
        assert_eq!(pa[6].1, 96);
        assert_eq!(pa[6].2, 100);
        // a prefix's pages are the longer prompt's pages (except a partial last page)
        let pc = page_keys(1, &t[..64], 16, &[16, 32, 48, 64]);
        assert_eq!(pc[..], pa[..4]);
        // the same tokens written by a chunk that ran further: not the same page
        let pd = page_keys(1, &t[..64], 16, &[16, 40, 64]);
        assert_eq!(pd[0], pa[0]);
        assert_ne!(pd[1].0, pa[1].0);
        assert_ne!(build_fingerprint("a"), build_fingerprint("b"));
        assert_ne!(checksum(b"abcdefgh1"), checksum(b"abcdefgh2"));
    }

    pub(super) fn test_store(cfg: PrefixStoreConfig) -> PrefixStore {
        PrefixStore::new(cfg, &[crate::state::SlotSpec::Recurrent { bytes: 64 }], "unit").unwrap()
    }

    /// A store with a disk tier whose writer writes one queued object per token sent on the returned gate (a stalled disk).
    fn stalled_store(name: &str, host_bytes: u64, disk_bytes: u64) -> (PrefixStore, Sender<()>, PathBuf) {
        let root = std::env::temp_dir().join(format!("intarsia-ps-unit-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let cfg = PrefixStoreConfig { host_bytes, disk_bytes, disk_dir: Some(root), ..Default::default() };
        let mut s = test_store(cfg);
        let dir = s.writer.as_ref().unwrap().dir.clone();
        let (tx, rx) = channel::<(Key, PathBuf, Arc<HostBuf>)>();
        let (gated_tx, gated) = channel();
        let (gate_tx, gate) = channel::<()>();
        let (dtx, done) = channel();
        std::thread::spawn(move || {
            for job in rx.iter() {
                if gate.recv().is_err() || gated_tx.send(job).is_err() {
                    break;
                }
            }
        });
        let build = s.build;
        let thread = std::thread::spawn(move || run_writer(gated, dtx, build));
        s.writer = Some(DiskWriter { dir: dir.clone(), tx, done, _thread: thread });
        (s, gate_tx, dir)
    }

    fn obj_path(dir: &Path, k: Key) -> PathBuf {
        dir.join(format!("o-{k:032x}.bin"))
    }

    #[test]
    fn disk_writes_stop_queueing_past_the_host_budget() {
        let mib = MIB as usize;
        let (mut s, gate, dir) = stalled_store("bound", 4 * MIB, 1 << 30);
        for k in 1..=6u128 {
            s.add_obj(k, mib, Mem::Host(Arc::new(vec![k as u8; mib].into())), k as u64, false);
        }
        // four 1 MiB writes fill the 4 MiB limit; the rest stay host-only
        assert_eq!(s.pending_write_bytes, 4 * MIB);
        assert_eq!(s.stats.disk_write_skips, 2);
        assert!((1..=4).all(|k| s.objs[&k].disk.is_some()));
        assert!((5..=6).all(|k| s.objs[&k].disk.is_none()));
        // the queued objects leave the store: their buffers are still the queue's, so the limit still holds
        for k in 1..=4u128 {
            s.drop_obj(k);
        }
        s.add_obj(7, mib, Mem::Host(Arc::new(vec![7; mib].into())), 7, false);
        assert!(s.objs[&7].disk.is_none());
        assert_eq!(s.pending_write_bytes, 4 * MIB);
        // the disk catches up: nothing pending, and no file for the objects dropped while queued
        for _ in 0..4 {
            gate.send(()).unwrap();
        }
        s.flush();
        assert_eq!(s.pending_write_bytes, 0);
        assert!((1..=4).all(|k| !obj_path(&dir, k).exists()));
        s.add_obj(8, mib, Mem::Host(Arc::new(vec![8; mib].into())), 8, false);
        assert!(s.objs[&8].disk.is_some());
        gate.send(()).unwrap();
        s.flush();
        assert!(obj_path(&dir, 8).exists());
        assert_eq!(s.snapshot().disk_pending_bytes, 0);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn disk_eviction_of_a_queued_write_leaves_no_file() {
        let mib = MIB as usize;
        let (mut s, gate, dir) = stalled_store("evict", 64 * MIB, 2 * MIB);
        for k in 1..=3u128 {
            s.add_obj(k, mib, Mem::Host(Arc::new(vec![k as u8; mib].into())), k as u64, false);
        }
        assert_eq!(s.used[2], 3 * MIB);
        // over the disk budget before anything was written: the oldest loses its (not yet existing) file, keeps its host copy
        s.enforce_disk();
        assert_eq!(s.used[2], 2 * MIB);
        assert!(s.objs[&1].disk.is_none() && s.objs[&1].mem.is_some());
        for _ in 0..3 {
            gate.send(()).unwrap();
        }
        s.flush();
        assert!(!obj_path(&dir, 1).exists(), "the writer's late file for an evicted object is removed");
        assert!(obj_path(&dir, 2).exists() && obj_path(&dir, 3).exists());
        assert!(s.objs[&2].disk.is_some() && s.objs[&2].staged.is_none());
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    struct FakeBuf(usize);

    impl crate::backend::BufferImpl for FakeBuf {
        fn bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn deferred_slabs_stay_within_the_cap() {
        let mib = MIB as usize;
        let mut s = test_store(PrefixStoreConfig { device_bytes: 0, ..Default::default() });
        assert!(s.use_slab());
        // near the cap a new slab shrinks to the room left, and a capture larger than the room does not defer
        s.slabs_full.push(Buffer::new(FakeBuf(DEFER_CAP as usize - 100 * mib)));
        let held = s.deferred_footprint();
        assert_eq!(PrefixStore::new_slab_size(mib, held), 100 * mib);
        assert!(s.defer_fits(100 * mib));
        assert!(!s.defer_fits(100 * mib + 256));
        // idle slabs: the first that fits (192 MiB) would pass the cap; the smallest (64 MiB) is what both the check and
        // the pick use
        s.slab_pool = vec![Buffer::new(FakeBuf(192 * mib)), Buffer::new(FakeBuf(64 * mib))];
        assert!(s.defer_fits(32 * mib));
        let i = s.pooled_slab(32 * mib).unwrap();
        assert_eq!(s.slab_pool[i].bytes(), 64 * mib);
        let buf = s.slab_pool.swap_remove(i);
        s.slab = Some((buf, 32 * mib));
        assert!(s.deferred_footprint() <= DEFER_CAP);
        assert!(!s.defer_fits(128 * mib), "only the 192 MiB slab is left, and it does not fit under the cap");
        // after the request: the near-cap slab is over the pool cap and freed, the 64 MiB one is idle again and reused
        s.recycle_slabs();
        let mut sizes: Vec<usize> = s.slab_pool.iter().map(Buffer::bytes).collect();
        sizes.sort();
        assert_eq!(sizes, vec![64 * mib, 192 * mib]);
        assert!(s.slabs_full.is_empty() && s.slab.is_none());
        assert_eq!(s.slab_pool[s.pooled_slab(mib).unwrap()].bytes(), 64 * mib);
    }
}

/// A host buffer for a device-to-host download, with every page already faulted in by the CPU. `vec![0u8; n]` maps
/// fresh zero pages lazily, and on the GB10 (unified memory) a download into untouched pages ran at about 170 MB/s
/// (ab35c: 11.8 GiB in 70.9 s; one 32 MiB object 227 ms) against GB/s into touched ones (the same object 1 ms after
/// a reused, already-written allocation). Writing one byte per page first moves the faults to the CPU, where they
/// are cheap.
fn host_buf(n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    let p = v.as_mut_ptr();
    for i in (0..n).step_by(4096) {
        // SAFETY: i < n = v.len(); a volatile write so the touch is not folded into the zeroed allocation.
        unsafe { std::ptr::write_volatile(p.add(i), 0) };
    }
    v
}

/// How the store allocates a NEW host buffer (`INTARSIA_PREFIX_STORE_PREFAULT`, lane storeprefault, opt-in, default off).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prefault {
    /// `vec![0u8; n]` plus a write per 4 KiB page ([`host_buf`]): the allocator puts a 16-byte header in front of a large block,
    /// so the data starts 16 bytes into a page.
    Off,
    /// A mapping of its own, starting on a page (`4k`).
    Aligned,
    /// As `Aligned`, starting on a 2 MiB boundary with `MADV_HUGEPAGE` over its whole 2 MiB blocks (`1`).
    Huge,
}

/// `INTARSIA_PREFIX_STORE_PREFAULT`: `1` = [`Prefault::Huge`], `4k` = [`Prefault::Aligned`], unset or anything else = off.
/// Measured on the GB10 (`scripts/storeprefault/d2h4.cu`): the first device-to-host copy into a new buffer that starts 16 bytes
/// into a page runs at about 1.3 GB/s (5 MiB in 3.9 ms, 32 MiB in 4.2 ms), into a page-aligned one at 54 GB/s (0.10 / 0.56 ms),
/// and later copies are fast either way, which is why a recycled host buffer was fast and a fresh server's settle slow
/// (lane storettft, cause C). Touching the pages costs about 3.5 GB/s with 4 KiB pages and 13 to 25 GB/s with huge ones.
fn host_prefault() -> Prefault {
    static P: std::sync::OnceLock<Prefault> = std::sync::OnceLock::new();
    *P.get_or_init(|| match std::env::var("INTARSIA_PREFIX_STORE_PREFAULT").as_deref().map(str::trim) {
        Ok("1") => Prefault::Huge,
        Ok("4k") => Prefault::Aligned,
        _ => Prefault::Off,
    })
}

/// Smallest new host buffer that gets a mapping of its own under [`host_prefault`]: every non-empty one. The first window (wd,
/// `INTARSIA_PREFIX_STORE_DEBUG=1`) had this at 1 MiB, and the first download into a new misaligned `Vec` was slow at every size
/// the store makes (4 KiB in 0.6 ms, 512 KiB in 1.4 ms: 406 such objects were half of a 885 MiB settle's 300 ms), not only the
/// large ones. A mapping is a VMA, two from 2 MiB up (the `madvise` splits off the 4 KiB tail; `vm.max_map_count` is 65530 by
/// default): the 8 GiB default host tier of the 35B holds about 3000 objects, about 4500 VMAs.
const MAPPED_MIN: usize = 1;
const HUGE: usize = 2 << 20;

/// A new host buffer of `n` bytes for a download, every page touched (see [`host_buf`] and [`Prefault`]).
fn new_host_buf(n: usize, mode: Prefault) -> HostBuf {
    if mode != Prefault::Off && n >= MAPPED_MIN {
        if let Some(b) = HostBuf::mapped(n, mode == Prefault::Huge) {
            return b;
        }
    }
    HostBuf::Vec(host_buf(n))
}

/// Bytes of an object in host memory: a `Vec` (disk loads, tests, small buffers, the default), or an anonymous mapping of its
/// own that starts on a page or a 2 MiB boundary ([`Prefault`]). Derefs to the `n` bytes.
pub(crate) enum HostBuf {
    Vec(Vec<u8>),
    /// `ptr` .. `ptr + len` (the object's bytes) inside the mapping `ptr` .. `ptr + map_len`.
    Mapped { ptr: std::ptr::NonNull<u8>, len: usize, map_len: usize },
}

// SAFETY: a `Mapped` buffer owns its mapping exclusively (nothing else holds the pointer), like a `Vec` owns its allocation.
unsafe impl Send for HostBuf {}
// SAFETY: shared access only reads through `&[u8]`.
unsafe impl Sync for HostBuf {}

impl HostBuf {
    /// Map `n` bytes (rounded up to the page) at a page boundary, or at a 2 MiB boundary with `MADV_HUGEPAGE` on its whole
    /// 2 MiB blocks when `huge`, and touch every page. `None` when the mapping fails (the caller falls back to a `Vec`).
    fn mapped(n: usize, huge: bool) -> Option<HostBuf> {
        // SAFETY: plain libc call.
        let page = (unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).max(4096) as usize;
        let len_map = n.next_multiple_of(page);
        let slack = if huge && n >= HUGE { HUGE } else { 0 };
        // SAFETY: an anonymous private mapping, no address hint; checked for MAP_FAILED.
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), len_map + slack, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        if p == libc::MAP_FAILED {
            return None;
        }
        let base = p as usize;
        let start = if slack > 0 { base.next_multiple_of(HUGE) } else { base };
        // SAFETY: both ranges lie inside the mapping just made (start - base < slack, start + len_map <= base + len_map + slack)
        // and are page multiples; nothing else uses them.
        unsafe {
            if start > base {
                libc::munmap(p, start - base);
            }
            let end = start + len_map;
            if base + len_map + slack > end {
                libc::munmap(end as *mut libc::c_void, base + len_map + slack - end);
            }
            #[cfg(target_os = "linux")]
            if slack > 0 {
                // best effort: without transparent huge pages (or with `never`) the pages are 4 KiB, which is only slower to touch
                libc::madvise(start as *mut libc::c_void, n / HUGE * HUGE, libc::MADV_HUGEPAGE);
            }
        }
        let ptr = std::ptr::NonNull::new(start as *mut u8)?;
        for i in (0..len_map).step_by(page) {
            // SAFETY: i < len_map, inside the mapping; a volatile write so the touch happens now, on this thread.
            unsafe { std::ptr::write_volatile(ptr.as_ptr().add(i), 0) };
        }
        Some(HostBuf::Mapped { ptr, len: n, map_len: len_map })
    }
}

impl From<Vec<u8>> for HostBuf {
    fn from(v: Vec<u8>) -> HostBuf {
        HostBuf::Vec(v)
    }
}

impl std::ops::Deref for HostBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            HostBuf::Vec(v) => v,
            // SAFETY: the mapping holds `len` initialised (zero-filled, then written) bytes for as long as `self` lives.
            HostBuf::Mapped { ptr, len, .. } => unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *len) },
        }
    }
}

impl std::ops::DerefMut for HostBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            HostBuf::Vec(v) => v,
            // SAFETY: as in `deref`, and `&mut self` makes this the only access.
            HostBuf::Mapped { ptr, len, .. } => unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *len) },
        }
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        if let HostBuf::Mapped { ptr, map_len, .. } = self {
            // SAFETY: the mapping `HostBuf::mapped` made (after its trims), unmapped once, here.
            unsafe { libc::munmap(ptr.as_ptr() as *mut libc::c_void, *map_len) };
        }
    }
}

#[cfg(test)]
mod host_buf_tests {
    use super::*;

    #[test]
    fn mapped_buffers_are_aligned_zeroed_and_writable() {
        for (n, mode) in [(5 << 20, Prefault::Huge), (32160 << 10, Prefault::Huge), (5 << 20, Prefault::Aligned), ((1 << 20) + 3, Prefault::Aligned), (512 << 10, Prefault::Huge), (6 << 10, Prefault::Huge)] {
            let mut b = new_host_buf(n, mode);
            assert!(matches!(b, HostBuf::Mapped { .. }), "{n} {mode:?}");
            assert_eq!(b.len(), n);
            let align = if mode == Prefault::Huge && n >= HUGE { HUGE } else { 4096 }; // (at least: the page size)
            assert_eq!(b.as_ptr() as usize % align, 0, "{n} {mode:?}");
            assert!(b.iter().all(|&x| x == 0));
            for (i, x) in b.iter_mut().enumerate() {
                *x = i as u8;
            }
            assert!(b.iter().enumerate().all(|(i, &x)| x == i as u8));
            let shared = Arc::new(b);
            let c = shared.clone();
            assert_eq!(std::thread::spawn(move || c[n - 1]).join().unwrap(), (n - 1) as u8);
        }
    }

    #[test]
    fn empty_or_off_buffers_stay_vecs() {
        assert!(matches!(new_host_buf(0, Prefault::Huge), HostBuf::Vec(_)));
        assert!(matches!(new_host_buf(4 << 10, Prefault::Aligned), HostBuf::Mapped { .. }));
        assert!(matches!(new_host_buf(5 << 20, Prefault::Off), HostBuf::Vec(_)));
        assert_eq!(new_host_buf(5 << 20, Prefault::Off).len(), 5 << 20);
    }

    /// With the knob on, the store's whole host path (capture to host, demote, recycle, restore) works on mapped buffers.
    #[test]
    fn take_host_buf_recycles_mapped_buffers_by_size() {
        let mut s = super::tests::test_store(PrefixStoreConfig { host_bytes: 64 << 20, ..Default::default() });
        let b = HostBuf::mapped(5 << 20, true).unwrap();
        let p = b.as_ptr();
        s.release_host(Arc::new(b));
        assert_eq!(s.host_free_bytes, 5 << 20);
        let again = s.take_host_buf(5 << 20);
        assert_eq!(again.as_ptr(), p);
        assert_eq!(s.host_free_bytes, 0);
    }
}
