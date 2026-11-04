//! DeepSeek-V4.1 serving: load one rank from its rank cache (`dsv41_cache`), build the Engram tier, and run the
//! single-sequence [`Engine`] on top of `intarsia_core::dsv41::runtime` (rank 0), or serve rank 0's passes (rank > 0,
//! `intarsia tp-worker`).
//!
//! Both ranks of a cross-process group are started with their own rank cache as `--model` (rank 0: `.../r0of2` on spark1,
//! rank 1: `.../r1of2` on spark2); the handshake fingerprint is the source checkpoint's fingerprint plus the group size,
//! layer limit, capacity and chunk, never the cache path.
//!
//! Environment (docs/tuning.md):
//! - `INTARSIA_DSV41_ENGRAM`: directory of this rank's Engram shards (`engram-l{1,14}-r{rank}of{world}.bin`, or the
//!   official safetensors); default `~/dsv41-engram`.
//! - `INTARSIA_DSV41_ENGRAM_HOT_MB`: hot-tier size (TinyLFU only, no pinned profile); default 0 (off).
//! - `INTARSIA_DSV41_CAP`: token capacity of the sequence (default 32768); `INTARSIA_DSV41_CHUNK`: prefill chunk (default
//!   the `--prefill-chunk`, else 512).

use crate::chat_template::ChatTemplateSource;
use crate::dsv41::{detect, HeaderShapes};
use crate::dsv41_cache::{load_rank_cache, open_manifest, Manifest, MANIFEST, MTP_DIR};
use crate::tokenizer::HfCodec;
use intarsia_core::comm::Comm;
use intarsia_core::dsv41::config::Dsv41Config;
use intarsia_core::dsv41::engram::EngramTier;
use intarsia_core::dsv41::graph::{GraphTp, VERIFY_MAX_ROWS};
use intarsia_core::dsv41::runtime::{fuse_shared_experts_at, Draft, Dsv41Runtime, EngramInputs, RuntimeOpts};
use intarsia_core::engine::{ChatOpts, Engine, FinishReason, GenEvent, GenRequest, IncrementalDecoder, ModelInfo, StopMatcher, TextCodec};
use intarsia_core::sampler::{Sampler, SamplingParams};
use intarsia_core::state::SequenceState;
use intarsia_core::telemetry::{Telemetry, TelemetrySnapshot};
use intarsia_core::tp_remote::{RemotePeer, RemoteSeq};
use intarsia_core::{Backend, DType, Error, Result};
use serde_json::Value;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn env_usize(k: &str) -> Option<usize> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

/// Load-time options of a DeepSeek-V4.1 rank.
#[derive(Clone, Debug)]
pub struct Dsv41Opts {
    pub cap: usize,
    pub chunk: usize,
    pub engram_dir: PathBuf,
    pub hot_mb: usize,
    /// Activation dtype (bf16; f32 for parity runs, `INTARSIA_DSV41_ACT=f32`).
    pub act: DType,
    /// Speculative decoding with the DSpark stages: on when the rank cache has its mtp part (`INTARSIA_DSV41_SPEC=0` turns it
    /// off, `=1` makes a missing mtp part an error).
    pub spec: bool,
}

impl Dsv41Opts {
    pub fn from_env(dir: &Path, prefill_chunk: Option<usize>) -> Dsv41Opts {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Dsv41Opts {
            cap: env_usize("INTARSIA_DSV41_CAP").unwrap_or(32768),
            chunk: env_usize("INTARSIA_DSV41_CHUNK").or(prefill_chunk).unwrap_or(512),
            engram_dir: std::env::var_os("INTARSIA_DSV41_ENGRAM").map(PathBuf::from).unwrap_or_else(|| Path::new(&home).join("dsv41-engram")),
            hot_mb: env_usize("INTARSIA_DSV41_ENGRAM_HOT_MB").unwrap_or(0),
            act: if std::env::var("INTARSIA_DSV41_ACT").as_deref() == Ok("f32") { DType::F32 } else { DType::Bf16 },
            spec: spec_on(std::env::var("INTARSIA_DSV41_SPEC").ok().as_deref(), dir.join(MTP_DIR).join(MANIFEST).exists()),
        }
    }
}

/// Unset: on when the cache has its DSpark sub-cache. Set: `1` is on, anything else off.
fn spec_on(setting: Option<&str>, has_mtp: bool) -> bool {
    setting.map_or(has_mtp, |v| v == "1")
}

/// Is `dir` a DeepSeek-V4.1 rank cache?
pub fn is_rank_cache(dir: &Path) -> bool {
    dir.join("manifest.json").exists() && detect(dir).ok().flatten().is_some()
}

/// What both ranks must agree on (handshake).
pub fn fingerprint(m: &Manifest, o: &Dsv41Opts) -> String {
    let mut base = format!("dsv41 source {} | world {} | layers {:?} | cap {} | chunk {} | act {}", m.source_fingerprint, m.world, m.layers, o.cap, o.chunk, o.act);
    if m.first_layer > 0 {
        base.push_str(&format!(" | first layer {}", m.first_layer));
    }
    // the spec-off fingerprint is the one before speculation existed, so old and new binaries still pair without it
    if o.spec {
        format!("{base} | dspark spec")
    } else {
        base
    }
}

fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    l.split_whitespace().nth(1)?.parse::<u64>().ok().map(|k| k * 1024)
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// `INTARSIA_DSV41_ENGRAM=off` (smoke tests without the 190 GB of tables): every row is zero, so the Engram layers add
/// nothing. The model's output is wrong in that mode; it only checks that everything else loads and runs.
struct ZeroTier {
    layout: intarsia_core::dsv41::engram::EngramLayout,
    shard: intarsia_core::dsv41::engram::EngramShard,
}

impl EngramTier for ZeroTier {
    fn layout(&self) -> &intarsia_core::dsv41::engram::EngramLayout {
        &self.layout
    }
    fn shard(&self) -> intarsia_core::dsv41::engram::EngramShard {
        self.shard
    }
    fn issue(&self, _layer: usize, _rows: &[u32]) -> Result<intarsia_core::dsv41::engram::EngramTicket> {
        Ok(intarsia_core::dsv41::engram::EngramTicket(0))
    }
    fn wait(&self, _t: intarsia_core::dsv41::engram::EngramTicket, out: &mut [u8]) -> Result<()> {
        let dh = self.layout.head_dim;
        let rb = intarsia_core::dsv41::engram::record_bytes(dh);
        for r in out.chunks_exact_mut(rb) {
            r[..dh].fill(0);
            r[dh..].fill(127);
        }
        Ok(())
    }
}

/// The Engram tier of rank `rank` of `world`: this rank's shards from `o.engram_dir`, O_DIRECT, io_uring.
fn engram_tier(cfg: &Dsv41Config, dir: &Path, o: &Dsv41Opts, rank: usize, world: usize) -> Result<EngramInputs> {
    use intarsia_engram::{EngramStore, HashLayout, NvmeEngramTier, ScaleMode, StoreConfig, TableShard, TokenMap};
    let hash = HashLayout::dsv41();
    let e = cfg.engram.as_ref().ok_or_else(|| Error::model("deepseek_v41: no engram config"))?;
    if hash.layer_ids() != e.layer_ids.as_slice() || hash.layout().rows != e.num_embeddings {
        return Err(Error::model(format!("deepseek_v41: the Engram tier's layout ({:?}, {:?}) is not the checkpoint's ({:?}, {:?})", hash.layer_ids(), hash.layout().rows, e.layer_ids, e.num_embeddings)));
    }
    if o.engram_dir.as_os_str() == "off" {
        eprintln!("dsv41: WARNING: INTARSIA_DSV41_ENGRAM=off: Engram rows are zero, the output is NOT the model's");
        let tm = TokenMap::load_json(&dir.join("dsv41_token_map.json")).map_err(|e| Error::model(format!("{}: {e}", dir.join("dsv41_token_map.json").display())))?;
        let tier = ZeroTier { layout: hash.layout().clone(), shard: intarsia_core::dsv41::engram::EngramShard { rank, world } };
        return Ok(EngramInputs { tier: Arc::new(tier), hasher: hash.hasher.clone(), token_map: tm.map });
    }
    let mut shards = Vec::new();
    for (li, &l) in hash.layer_ids().iter().enumerate() {
        let rows = hash.rank_rows(li, rank, world);
        let s = TableShard::open_dir(&o.engram_dir, l as u32, rank, world, rows, ScaleMode::Read, true)
            .map_err(|e| Error::model(format!("engram shard layer {l} rank {rank}/{world} in {}: {e}", o.engram_dir.display())))?;
        shards.push(s);
    }
    let mut sc = StoreConfig::default();
    if o.hot_mb > 0 {
        sc.hot.capacity_bytes = o.hot_mb << 20;
        sc.hot.dynamic_fraction = 1.0;
    }
    let store = EngramStore::new(shards, sc).map_err(|e| Error::model(format!("engram store: {e}")))?;
    let tier = NvmeEngramTier::new(store, &hash, rank, world)?;
    let tm = TokenMap::load_json(&dir.join("dsv41_token_map.json")).map_err(|e| Error::model(format!("{}: {e}", dir.join("dsv41_token_map.json").display())))?;
    eprintln!("dsv41: engram tier rank {rank}/{world} from {} (hot tier {} MiB), token map {} -> {}", o.engram_dir.display(), o.hot_mb, tm.map.len(), tm.compressed_vocab);
    Ok(EngramInputs { tier: Arc::new(tier) as Arc<dyn EngramTier>, hasher: hash.hasher.clone(), token_map: tm.map })
}

/// Load this rank: weights from the rank cache in `dir` (streamed, O_DIRECT), the shared experts fused, the Engram tier,
/// the runtime. Logs the per-rank memory breakdown.
pub fn load_runtime(dir: &Path, backend: Arc<dyn Backend>, comm: Option<Arc<dyn Comm>>, o: &Dsv41Opts) -> Result<(Dsv41Runtime, Manifest)> {
    load_runtime_with(dir, backend, comm, o, None)
}

/// The router's `gate.bias` is f32 in the graph (the official Gate holds it in an f32 parameter); the Mia checkpoint
/// stores the main layers' as F32 but the DSpark stages' (`mtp.{i}.ffn.gate.bias`) as F16. Upcast those at load
/// (a few hundred bytes each), as the official load_state_dict does.
fn upcast_f16_gate_biases(weights: &mut std::collections::HashMap<String, intarsia_core::backend::TensorRef>, shapes: &mut HeaderShapes, backend: &dyn Backend) -> Result<()> {
    let names: Vec<String> = weights.iter().filter(|(k, t)| k.ends_with(".gate.bias") && t.dtype == DType::F16).map(|(k, _)| k.clone()).collect();
    for n in names {
        let t = &weights[&n];
        let numel: usize = t.shape.iter().product();
        let mut h = vec![0u8; 2 * numel];
        backend.download(&t.buf, t.offset, &mut h, 0)?;
        backend.synchronize(0)?;
        let f: Vec<u8> = h.chunks_exact(2).flat_map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])).to_le_bytes()).collect();
        let buf = backend.alloc(f.len().max(16), intarsia_core::backend::MemKind::Device)?;
        backend.upload(&buf, 0, &f, 0)?;
        backend.synchronize(0)?;
        let shape = t.shape.clone();
        if let Some(e) = shapes.0.get_mut(&n) {
            e.0 = DType::F32;
        }
        weights.insert(n, intarsia_core::backend::TensorRef::new(buf, DType::F32, &shape));
    }
    Ok(())
}

fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, man) = (((h >> 15) as u32) << 31, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// [`load_runtime`] with the Engram inputs given (tests: a tier serving fixture rows) instead of the NVMe tier.
pub fn load_runtime_with(dir: &Path, backend: Arc<dyn Backend>, comm: Option<Arc<dyn Comm>>, o: &Dsv41Opts, engram_override: Option<EngramInputs>) -> Result<(Dsv41Runtime, Manifest)> {
    let m = open_manifest(dir)?;
    let world = comm.as_ref().map_or(1, |c| c.world());
    let rank = comm.as_ref().map_or(0, |c| c.rank());
    if (m.rank, m.world) != (rank, world) {
        return Err(Error::invalid(format!("deepseek_v41: {} is the cache of rank {} of {}, this process is rank {rank} of {world}", dir.display(), m.rank, m.world)));
    }
    let cfg = detect(dir)?.ok_or_else(|| Error::model(format!("{}: not a DeepSeek-V4.1 checkpoint", dir.display())))?;
    let avail0 = mem_available();
    let t0 = Instant::now();
    let lw = load_rank_cache(dir, &*backend)?;
    let mut weights = lw.tensors;
    let mut weight_bytes = lw.bytes;
    let mut shapes = HeaderShapes::open(dir)?;
    // the DSpark stages: the sub-cache `<dir>/mtp` (`intarsia dsv41-cache --mtp-only`), loaded only when speculating
    let mtp_dir = dir.join(MTP_DIR);
    if o.spec {
        let mm = open_manifest(&mtp_dir).map_err(|e| Error::model(format!("deepseek_v41: INTARSIA_DSV41_SPEC=1 needs the DSpark sub-cache ({e}); build it with `intarsia dsv41-cache --mtp-only`")))?;
        if mm.scope.as_deref() != Some("mtp") || (mm.rank, mm.world) != (m.rank, m.world) || mm.source_fingerprint != m.source_fingerprint {
            return Err(Error::model(format!("deepseek_v41: {} is not the DSpark sub-cache of this rank cache (scope {:?}, rank {} of {}, source {})", mtp_dir.display(), mm.scope, mm.rank, mm.world, mm.source_fingerprint)));
        }
        let mw = load_rank_cache(&mtp_dir, &*backend)?;
        weight_bytes += mw.bytes;
        weights.extend(mw.tensors);
        shapes.0.extend(HeaderShapes::open(&mtp_dir)?.0);
    }
    let n_layers = m.layers.unwrap_or(cfg.n_layers).min(cfg.n_layers);
    let mut moes: Vec<(String, usize)> = (m.first_layer..n_layers).map(|l| (format!("layers.{l}.ffn"), cfg.n_routed_experts)).collect();
    if let Some(m) = &cfg.mtp {
        moes.extend((0..m.n_layers).map(|i| (format!("mtp.{i}.ffn"), m.n_routed_experts)));
    }
    fuse_shared_experts_at(&*backend, &mut weights, &moes, cfg.dim)?;
    // vision / aligner tensors and anything else the text graph does not read stay on disk
    weights.retain(|k, _| !(k.starts_with("vision.") || k.starts_with("aligner.") || k.starts_with("image_")));
    upcast_f16_gate_biases(&mut weights, &mut shapes, &*backend)?;
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("dsv41: rank {rank}/{world}: loaded {:.2} GiB of weights from {} in {load_s:.1} s ({:.2} GB/s)", gib(weight_bytes), dir.display(), weight_bytes as f64 / 1e9 / load_s.max(1e-9));
    let first_engram = cfg.engram.as_ref().and_then(|e| e.layer_ids.iter().min().copied());
    let engram = match (engram_override, first_engram) {
        (Some(e), _) => Some(e),
        (None, Some(l)) if l < n_layers => Some(engram_tier(&cfg, dir, o, rank, world)?),
        _ => None,
    };
    let opts = RuntimeOpts { act: o.act, cap: o.cap, chunk: o.chunk, tp: GraphTp { world, layers: m.layers, first: m.first_layer, ..GraphTp::default() }, spec: o.spec };
    let rt = Dsv41Runtime::new(cfg, Box::new(shapes), backend, comm, weights, engram, opts)?;
    match (o.spec, rt.dspark) {
        (true, n) => eprintln!("dsv41: rank {rank}/{world}: speculative decoding on, {n} DSpark stages loaded from {}", mtp_dir.display()),
        (false, _) => eprintln!(
            "dsv41: rank {rank}/{world}: speculative decoding off ({})",
            if mtp_dir.join(MANIFEST).exists() { "INTARSIA_DSV41_SPEC is set and not 1" } else { "no DSpark sub-cache; build it with `intarsia dsv41-cache --mtp-only`" }
        ),
    }
    let mm = &rt.memory;
    let avail1 = mem_available();
    eprintln!(
        "dsv41: rank {rank}/{world} memory: weights {:.2} GiB, workspace {:.3} GiB (chunk {}), constants {:.3} GiB, state {:.3} GiB per sequence at {} tokens; MemAvailable {} -> {}",
        gib(mm.weights),
        gib(mm.workspace),
        o.chunk,
        gib(mm.constants),
        gib(mm.state_per_seq),
        o.cap,
        avail0.map_or("?".into(), |a| format!("{:.2} GiB", gib(a))),
        avail1.map_or("?".into(), |a| format!("{:.2} GiB", gib(a))),
    );
    Ok((rt, m))
}

/// Log the memory after warm-up (NCCL buffers, kernel scratch, CUDA context): the non-weight total this rank holds.
pub fn log_memory_after_warmup(rt: &Dsv41Runtime, avail_before_load: Option<u64>, rank: usize) {
    let avail = mem_available();
    let mm = &rt.memory;
    let used = match (avail_before_load, avail) {
        (Some(a), Some(b)) => Some(a.saturating_sub(b)),
        _ => None,
    };
    eprintln!(
        "dsv41: rank {rank} after warm-up: MemAvailable {}, used since start {}, of which weights {:.2} GiB, non-weight {} (workspace {:.3}, scratch {:.3}, state {:.3}, constants {:.3}, rest = CUDA context + NCCL + engram + host)",
        avail.map_or("?".into(), |a| format!("{:.2} GiB", gib(a))),
        used.map_or("?".into(), |u| format!("{:.2} GiB", gib(u))),
        gib(mm.weights),
        used.map_or("?".into(), |u| format!("{:.2} GiB", gib(u.saturating_sub(mm.weights)))),
        gib(mm.workspace),
        gib(rt.scratch_bytes()),
        gib(mm.state_per_seq),
        gib(mm.constants),
    );
}

pub fn mem_available_now() -> Option<u64> {
    mem_available()
}

/// Rank > 0 of a cross-process group (`intarsia tp-worker`): load, handshake, serve rank 0's passes.
pub fn serve_rank(dir: &Path, backend: Arc<dyn Backend>, comm: Arc<dyn Comm>, mut stream: std::net::TcpStream, prefill_chunk: Option<usize>) -> Result<u64> {
    let o = Dsv41Opts::from_env(dir, prefill_chunk);
    let avail0 = mem_available();
    let (rt, m) = load_runtime(dir, backend, Some(comm.clone()), &o)?;
    let cap = intarsia_core::tp_remote::handshake(&mut stream, &fingerprint(&m, &o), o.cap)?;
    if cap != o.cap {
        return Err(Error::invalid(format!("deepseek_v41: ranks disagree on the capacity ({cap} vs {})", o.cap)));
    }
    eprintln!("dsv41: rank {} serving rank 0's passes", comm.rank());
    let n = intarsia_core::tp_remote::serve_group(&rt, stream, comm.clone());
    log_memory_after_warmup(&rt, avail0, comm.rank());
    n
}

/// Speculative decoding settings (`INTARSIA_DSV41_SPEC_K`: most drafts a round, default 4, at most the block size - 1;
/// `INTARSIA_DSV41_SPEC_MIN_CONF`: verify the leading drafts whose confidence logit reaches it, at least one; default
/// off).
#[derive(Clone, Copy, Debug)]
struct SpecOpts {
    k: usize,
    min_conf: f32,
    /// `INTARSIA_DSV41_SPEC_TRACE=1`: one stderr line per round (what a different k or floor would have done offline).
    trace: bool,
    /// `INTARSIA_DSV41_SPEC_CHOOSE` (default on, `0` = fixed `k` and `min_conf`): k per round from the confidence head ([`choose_k`]).
    choose: bool,
    /// `INTARSIA_DSV41_SPEC_ROUND_MS`: a forward pass's ms by its rows (1, 2, ..; [`ROUND_MS`]).
    round_ms: [f64; ROUND_MS.len()],
    /// `INTARSIA_DSV41_SPEC_DRAFT_MS`: a round's ms outside its forward pass ([`DRAFT_MS`]).
    draft_ms: f64,
}

/// Measured at TP=2 on the shipped binary (docs/results/2026-11-02-dsv41-decode2.md, window 2, `INTARSIA_DSV41_SPEC_TRACE=1`): the
/// median whole-pass ms on rank 0 of a pass of 1 .. 4 rows over the 32 requests of the category set (1300 rounds or more each
/// but the 1-row pass, 60), 5 rows from the k = 4 arm. Window 6 of docs/results/2026-11-01-dsv41-graphs.md had 31.3, 38.1, 42.9,
/// 47.8, 52.7. The lane forks (`INTARSIA_DSV41_LANES`) take about 1 ms off each; the decisions depend on the steps between
/// entries and on [`DRAFT_MS`], and the shipped arm used this table.
const ROUND_MS: [f64; 5] = [31.1, 36.5, 41.2, 46.0, 50.8];
/// Measured the same way: a round's period minus its verify pass (drafting block, Markov loop, host work). 9.0 before the draft
/// round moved onto the device.
const DRAFT_MS: f64 = 5.5;

/// TensorFold's `chooseK` (`draft.zig`): the k in `0..=kmax` with the most expected tokens a millisecond. Expected tokens are
/// 1 plus the sum of the prefix survivals, a survival being the product of the sigmoids of the confidences so far; the cost
/// is `round_ms[k] + draft_ms` (`round_ms[k]`: a pass of `k + 1` rows). The first best wins.
fn choose_k(conf: &[f32], kmax: usize, round_ms: &[f64], draft_ms: f64) -> usize {
    let (mut best, mut best_v, mut surv, mut tokens) = (0, -1.0f64, 1.0f64, 1.0f64);
    for k in 0..=kmax.min(conf.len()).min(round_ms.len().saturating_sub(1)) {
        if k > 0 {
            surv /= 1.0 + (-(conf[k - 1] as f64)).exp();
            tokens += surv;
        }
        let v = tokens / (round_ms[k] + draft_ms);
        if v > best_v {
            (best, best_v) = (k, v);
        }
    }
    best
}

#[derive(Default)]
struct SpecStats {
    rounds: usize,
    drafted: usize,
    accepted: usize,
    /// Per draft position: rows checked (reached in the sequence), and accepted.
    reached: [usize; VERIFY_MAX_ROWS],
    ok: [usize; VERIFY_MAX_ROWS],
    /// This round: the next position to check, its drafts' confidences, drafts accepted.
    pos: usize,
    conf: Vec<f32>,
    round_ok: usize,
    /// A round is open (it may have chosen no draft, so `conf` can be empty).
    traced: bool,
    /// This round's ms (`INTARSIA_DSV41_SPEC_TRACE`): drafting, the forward pass (verify or one row), and the round's period
    /// (its start to the next round's start, which also holds the host work between rounds).
    draft_ms: f64,
    pass_ms: f64,
    round_start: Option<Instant>,
    period_ms: f64,
}

impl SpecStats {
    fn check(&mut self, accepted: bool) {
        self.reached[self.pos] += 1;
        if accepted {
            self.accepted += 1;
            self.ok[self.pos] += 1;
            self.round_ok += 1;
        }
        self.pos += 1;
    }

    /// `accepted/checked` per position, up to the last one reached.
    fn by_row(&self) -> String {
        (0..VERIFY_MAX_ROWS).filter(|&i| self.reached[i] > 0).map(|i| format!("{}/{}", self.ok[i], self.reached[i])).collect::<Vec<_>>().join(" ")
    }

    /// The round that just ended, as its trace line.
    fn end_round(&mut self, trace: bool) {
        if trace && self.traced {
            self.period_ms = self.round_start.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
            let c: Vec<String> = self.conf.iter().map(|v| format!("{v:.3}")).collect();
            eprintln!(
                "dsv41: spec round k={} conf={} ok={} draft_ms={:.2} pass_ms={:.2} period_ms={:.2}",
                self.conf.len(),
                c.join(","),
                self.round_ok,
                self.draft_ms,
                self.pass_ms,
                self.period_ms
            );
        }
        self.conf.clear();
        self.pos = 0;
        self.round_ok = 0;
        self.traced = false;
    }
}

/// One row of next-token scores: the logits, or (a speculative verify row whose logits nothing reads) the token the
/// device already took from them (first index of the maximum).
enum Row {
    Logits(Vec<f32>),
    Tok(u32),
}

struct Job {
    req: GenRequest,
    tx: Sender<GenEvent>,
}

struct Worker {
    rt: Dsv41Runtime,
    peer: Option<Arc<RemotePeer>>,
    st: SequenceState,
    remote: Option<RemoteSeq>,
    codec: Arc<HfCodec>,
    eos: Vec<u32>,
    telemetry: Arc<Telemetry>,
    spec: Option<SpecOpts>,
    /// Set when a tensor-parallel group failed for good (`Engine::failed`: /health turns 503, new requests are refused, and a
    /// tp-remote server exits for a restart of both ranks).
    failed: Arc<Mutex<Option<String>>>,
}

/// Collect the remote rank's answer to the pass just run here. A backend error on this rank fails the group (`group_failed`):
/// the other rank may sit in a collective this rank no longer joins and never answer, so do not wait for it (the peer is
/// marked unusable instead). Any other error still drains the answer, which keeps the two ranks in step.
fn settle<T>(remote: &mut Option<RemoteSeq>, out: &Result<T>) -> Result<()> {
    let Some(r) = remote else { return Ok(()) };
    if matches!(out, Err(Error::Backend(_))) {
        r.mark_unusable();
        return Ok(());
    }
    let rr = r.finish();
    if out.is_ok() {
        rr
    } else {
        Ok(())
    }
}

impl Worker {
    fn forward(&mut self, ids: &[u32], logits: bool) -> Result<Option<Vec<f32>>> {
        if let Some(r) = &self.remote {
            r.start_forward(ids, false)?;
        }
        let out = self.rt.forward(&mut self.st, ids, logits);
        settle(&mut self.remote, &out)?;
        out
    }

    /// One DSpark drafting round on both ranks after the sequence's last row, `first` the round's first token: the
    /// drafts rank 0 chose (`ids[1..]`) and their confidences.
    fn draft(&mut self, first: u32, steps: usize) -> Result<Draft> {
        if let Some(r) = &self.remote {
            r.start_draft(&[first])?;
        }
        let out = self.rt.draft_steps(&self.st, first, steps);
        settle(&mut self.remote, &out)?;
        out?.ok_or_else(|| Error::backend("deepseek_v41: rank 0 got no draft"))
    }

    /// A verify pass on both ranks (appended to the sequence): every row's logits, or (`logits` false) every row's
    /// argmax computed on the device (nothing else is read back).
    fn verify(&mut self, ids: &[u32], logits: bool) -> Result<Vec<Row>> {
        if let Some(r) = &self.remote {
            r.start_verify(ids)?;
        }
        let out = if logits {
            self.rt.forward_rows(&mut self.st, ids).map(|rows| rows.into_iter().map(Row::Logits).collect())
        } else {
            self.rt.forward_rows_ids(&mut self.st, ids).map(|t| t.into_iter().map(Row::Tok).collect())
        };
        settle(&mut self.remote, &out)?;
        out
    }

    /// The logits rows after `tok` (not yet in the sequence): with speculation a drafting round and a verify pass of
    /// `[tok, drafts..]` (each row tagged with the draft it checks; the last row checks none), else a plain step.
    fn next_rows(&mut self, tok: u32, spec: Option<SpecOpts>, stats: &mut SpecStats, logits: bool) -> Result<VecDeque<(Row, Option<u32>)>> {
        let room = self.rt.opts.cap.saturating_sub(self.st.len + 1);
        if let Some(sp) = spec.filter(|_| room >= 2) {
            stats.end_round(sp.trace);
            let t0 = Instant::now();
            stats.round_start = Some(t0);
            let d = self.draft(tok, sp.k)?;
            let t1 = Instant::now();
            stats.draft_ms = t1.duration_since(t0).as_secs_f64() * 1e3;
            let mut k = if sp.choose {
                choose_k(&d.confidence, sp.k, &sp.round_ms, sp.draft_ms)
            } else {
                d.confidence.iter().take(sp.k).take_while(|&&c| c >= sp.min_conf).count().max(1).min(sp.k)
            };
            k = k.min(room - 1).min(d.ids.len() - 1);
            stats.conf.extend_from_slice(&d.confidence[..k.min(d.confidence.len())]);
            stats.traced = true;
            if k == 0 {
                // no draft is worth a row: the plain one-row pass
                stats.rounds += 1;
                let l = self.forward(&[tok], true)?.expect("logits");
                stats.pass_ms = t1.elapsed().as_secs_f64() * 1e3;
                return Ok(VecDeque::from([(Row::Logits(l), None)]));
            }
            let mut ids = vec![tok];
            ids.extend_from_slice(&d.ids[1..=k]);
            let rows = self.verify(&ids, logits)?;
            stats.pass_ms = t1.elapsed().as_secs_f64() * 1e3;
            stats.rounds += 1;
            stats.drafted += k;
            return Ok(rows.into_iter().enumerate().map(|(i, r)| (r, ids.get(i + 1).copied())).collect());
        }
        let l = self.forward(&[tok], true)?.expect("logits");
        Ok(VecDeque::from([(Row::Logits(l), None)]))
    }

    /// Roll both ranks back to `len` tokens.
    fn truncate(&mut self, len: usize) -> Result<()> {
        if let Some(r) = &mut self.remote {
            r.truncate(len)?;
        }
        self.st.truncate(len, &*self.rt.backend)
    }

    /// A fresh sequence on both ranks (after a failure).
    fn reset(&mut self) -> Result<()> {
        self.remote = None;
        self.st = self.rt.new_state()?;
        if let Some(p) = &self.peer {
            self.remote = Some(p.new_state()?);
        }
        Ok(())
    }

    /// A device or group error at world > 1 leaves the group unusable (ADR-0006: the other rank may sit in a collective this
    /// rank never joins, the context may hold a sticky trap). Abort the communicator (off thread: it may not return on a
    /// trapped context), refuse the peer, mark the engine failed. No `reset`: it would talk to a rank that is gone.
    fn group_failed(&mut self, e: &Error) -> bool {
        let world = self.rt.comm.as_ref().map_or(1, |c| c.world());
        if world < 2 || !matches!(e, Error::Backend(_)) {
            return false;
        }
        if self.failed.lock().unwrap().is_some() {
            return true;
        }
        eprintln!("dsv41: the tensor-parallel group failed ({e}); aborting the communicator and marking the engine failed");
        if let Some(p) = &self.peer {
            p.mark_unusable();
        }
        if let Some(c) = self.rt.comm.clone() {
            let _ = std::thread::Builder::new().name("dsv41-abort".into()).spawn(move || c.abort());
        }
        *self.failed.lock().unwrap() = Some(format!("tensor-parallel group failed: {e}"));
        true
    }

    fn run(&mut self, job: Job) {
        let tx = job.tx.clone();
        if let Err(e) = self.generate(job) {
            let _ = tx.send(GenEvent::Error { message: e.to_string() });
            if self.group_failed(&e) {
                return;
            }
            if let Err(e2) = self.reset() {
                eprintln!("dsv41: reset after a failed request failed: {e2}");
            }
        }
    }

    fn generate(&mut self, job: Job) -> Result<()> {
        let Job { req, tx } = job;
        let t0 = Instant::now();
        self.telemetry.request_started();
        if req.prompt.is_empty() {
            return Err(Error::BadRequest("empty prompt".into()));
        }
        if req.prompt.len() + req.max_tokens.max(1) > self.rt.opts.cap {
            return Err(Error::BadRequest(format!("prompt {} + max_tokens {} exceed the context {}", req.prompt.len(), req.max_tokens, self.rt.opts.cap)));
        }
        // prefix reuse: the common prefix, if the state can roll back to it (ring margin), else from scratch
        let common = self.st.tokens.iter().zip(&req.prompt).take_while(|(a, b)| a == b).count().min(req.prompt.len() - 1);
        let keep = self.st.restorable_len(common);
        if keep != self.st.len {
            self.truncate(keep)?;
        }
        let cached = keep;
        // speculative decoding: greedy requests without a grammar or a logits trace (the accepted tokens are the target's
        // argmax, so the reply is the target model's own greedy reply, up to verify-row numerics)
        let spec = self.spec.filter(|_| req.temperature <= 0.0 && req.constraint.is_none() && req.trace_logits.is_none());
        let mut stats = SpecStats::default();
        let mut rows: VecDeque<(Row, Option<u32>)> = VecDeque::from([(Row::Logits(self.forward(&req.prompt[keep..].to_vec(), true)?.expect("logits")), None)]);
        let prefill = t0.elapsed();
        let params = SamplingParams { temperature: req.temperature, top_k: req.top_k, top_p: req.top_p };
        let mut sampler = Sampler::new(params, req.seed);
        let mut dec = IncrementalDecoder::new();
        let mut stops = StopMatcher::new(&req.stop);
        // structured output: the server's compiled constraint masks the host logits each step (computed inline: the
        // mask costs far less than this engine's decode step)
        let mut cons = match &req.constraint {
            Some(c) => Some(c.start()?),
            None => None,
        };
        let mut n = 0usize;
        let t1 = Instant::now();
        let finish;
        let mut stop_seq = None;
        loop {
            let (row, expect) = rows.pop_front().expect("a logits row");
            let tok = match row {
              // a verify row whose argmax the device computed (no trace, mask or constraint applies: `next_rows`)
              Row::Tok(t) => t,
              Row::Logits(mut logits) => {
            if let Some(tl) = &req.trace_logits {
                tl.lock().unwrap().push(logits.clone());
            }
            if n < req.min_tokens {
                // the stop set only, like the main engine (and vLLM): with ignore_eos the EOS ids are not stops and stay
                // samplable (a greedy golden with min_tokens + ignore_eos then matches TensorFold's, which emits EOS)
                let eos: &[u32] = if req.ignore_eos { &[] } else { &self.eos };
                for &e in eos.iter().chain(&req.stop_ids) {
                    if let Some(x) = logits.get_mut(e as usize) {
                        *x = f32::NEG_INFINITY;
                    }
                }
            }
            if let Some(c) = cons.as_mut().filter(|c| c.active()) {
                if !c.compute_mask()?.apply(&mut logits) {
                    return Err(Error::Other("structured output: the constraint allows no token here".into()));
                }
            }
            if req.temperature <= 0.0 { intarsia_core::sampler::argmax(&logits) as u32 } else { sampler.sample(&logits) as u32 }
              }
            };
            n += 1;
            // a verify row: its draft accepted (already in the sequence), or the first rejection (the rows after it and
            // the sequence past it are dropped)
            let in_seq = match expect {
                Some(d) if d == tok => {
                    stats.check(true);
                    true
                }
                Some(_) => {
                    stats.check(false);
                    let keep = self.st.len - rows.len();
                    rows.clear();
                    self.truncate(keep)?;
                    false
                }
                None => false,
            };
            let is_eos = !req.ignore_eos && (self.eos.contains(&tok) || req.stop_ids.contains(&tok));
            if is_eos {
                finish = FinishReason::Stop;
                break;
            }
            if let Some(c) = cons.as_mut() {
                c.commit(tok)?;
            }
            let text = dec.push(&*self.codec, tok)?;
            let (out, hit) = stops.push(&text);
            if tx.send(GenEvent::Token { id: tok, text: out }).is_err() {
                finish = FinishReason::Cancelled;
                break;
            }
            if hit {
                finish = FinishReason::Stop;
                stop_seq = req.stop.iter().find(|s| text.contains(s.as_str())).cloned();
                break;
            }
            if n >= req.max_tokens || self.st.len - rows.len() + 1 >= self.rt.opts.cap {
                finish = FinishReason::Length;
                break;
            }
            if req.cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                finish = FinishReason::Cancelled;
                break;
            }
            if !in_seq {
                // device argmax unless something reads the logits of the rows to come: the min_tokens mask (applies while `n` is
                // below it; a row consumed later sees a larger `n`), a constraint or a logits trace (neither speculates)
                rows = self.next_rows(tok, spec, &mut stats, n < req.min_tokens)?;
            }
        }
        // drafts past the reply's end stay out of the sequence (the next request's prefix match sees the reply only)
        if !rows.is_empty() {
            let keep = self.st.len - rows.len();
            self.truncate(keep)?;
        }
        stats.end_round(spec.is_some_and(|s| s.trace));
        if stats.rounds > 0 {
            eprintln!(
                "dsv41: spec {} tokens in {} rounds ({:.2} a round), {}/{} drafts accepted ({:.0} %), by position {}",
                n,
                stats.rounds,
                n as f64 / stats.rounds as f64,
                stats.accepted,
                stats.drafted,
                100.0 * stats.accepted as f64 / stats.drafted.max(1) as f64,
                stats.by_row()
            );
        }
        if finish != FinishReason::Stop || stop_seq.is_none() {
            let rest = dec.flush(&*self.codec)?;
            let (mut out, _) = stops.push(&rest);
            out.push_str(&stops.flush());
            if !out.is_empty() {
                let _ = tx.send(GenEvent::Token { id: 0, text: out });
            }
        }
        let decode = t1.elapsed();
        self.telemetry.request_finished(finish.as_str(), req.prompt.len(), n, cached, prefill, decode, prefill);
        let _ = tx.send(GenEvent::Done {
            finish_reason: finish,
            prompt_tokens: req.prompt.len(),
            completion_tokens: n,
            cached_tokens: cached,
            stop_sequence: stop_seq,
            prefill_ms: prefill.as_secs_f64() * 1e3,
            decode_ms: decode.as_secs_f64() * 1e3,
            prefill_chunk: None,
        });
        Ok(())
    }
}

/// The single-sequence DeepSeek-V4.1 engine (rank 0).
pub struct Dsv41Engine {
    tx: Mutex<Sender<Job>>,
    info: ModelInfo,
    codec: Arc<HfCodec>,
    telemetry: Arc<Telemetry>,
    failed: Arc<Mutex<Option<String>>>,
}

impl Dsv41Engine {
    /// Rank 0: load from the rank cache `dir`; `peer` is the remote rank (cross-process TP), `comm` the group.
    pub fn load(dir: &Path, backend: Arc<dyn Backend>, comm: Option<Arc<dyn Comm>>, remote: Option<std::net::TcpStream>, model_id: Option<String>, prefill_chunk: Option<usize>) -> Result<Dsv41Engine> {
        let o = Dsv41Opts::from_env(dir, prefill_chunk);
        let avail0 = mem_available();
        let (rt, m) = load_runtime(dir, backend.clone(), comm.clone(), &o)?;
        let peer = match remote {
            Some(mut stream) => {
                let cap = intarsia_core::tp_remote::handshake(&mut stream, &fingerprint(&m, &o), o.cap)?;
                if cap != o.cap {
                    return Err(Error::invalid(format!("deepseek_v41: ranks disagree on the capacity ({cap} vs {})", o.cap)));
                }
                Some(RemotePeer::new(stream, comm.clone().ok_or_else(|| Error::invalid("a remote rank needs a communicator"))?)?)
            }
            None => None,
        };
        let codec = Arc::new(HfCodec::load(dir, &ChatTemplateSource::Auto)?);
        let cfg_json: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error::model(e.to_string()))?).map_err(|e| Error::model(e.to_string()))?;
        let mut eos = crate::load::eos_ids(dir, &cfg_json);
        if !eos.contains(&rt.cfg.eos) {
            eos.push(rt.cfg.eos);
        }
        let info0 = backend.info();
        let c = rt.cfg.clone();
        let id = model_id.unwrap_or_else(|| Path::new(&m.source).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "deepseek-v4.1".into()));
        let info = ModelInfo {
            id,
            family: "deepseek_v41".into(),
            backend: backend.name().to_string(),
            device: info0.name.clone(),
            arch: "DeepseekV41ForCausalLM".into(),
            n_layers: m.layers.unwrap_or(c.n_layers),
            hidden: c.dim,
            vocab: c.vocab,
            n_heads: c.n_heads,
            n_kv_heads: 1,
            head_dim: c.head_dim,
            max_position_embeddings: 1 << 20,
            context_len: o.cap,
            act_dtype: "bf16".into(),
            kv_dtype: "bf16".into(),
            weights_bytes: rt.memory.weights,
            eos_ids: eos.clone(),
            has_chat_template: codec.has_chat_template(),
            // The template's thinking mode ends the prompt in `<think>`; the model closes it with the single `</think>`
            // token (128822), so a request with `enable_thinking: true` gets `reasoning_content` split off.
            reasoning_parser: Some("think-tags".into()),
            tool_parser: None,
            memory_plan: intarsia_core::memplan::MemoryPlan {
                device: info0.name.clone(),
                total_bytes: info0.total_mem,
                free_bytes: info0.free_mem,
                weights_bytes: rt.memory.weights,
                workspace_bytes: rt.memory.workspace,
                kv_bytes_per_token: rt.memory.state_per_seq / o.cap.max(1) as u64,
                kv_fraction: 0.0,
                kv_budget_bytes: rt.memory.state_per_seq,
                kv_cap_tokens: o.cap,
                kv_total_bytes: rt.memory.state_per_seq,
                notes: vec![format!("deepseek_v41 rank 0 of {}", m.world)],
            },
            // its own reuse path (`restorable_len` in `generate`), not `Model::reuse_len`
            reuse_cold_cuts_only: false,
        };
        let st = rt.new_state()?;
        let remote_seq = match &peer {
            Some(p) => Some(p.new_state()?),
            None => None,
        };
        let telemetry = Arc::new(Telemetry::new());
        let spec = rt.opts.spec.then(|| {
            let block = rt.cfg.mtp.as_ref().map_or(1, |m| m.block_size);
            let k = env_usize("INTARSIA_DSV41_SPEC_K").unwrap_or(4).clamp(1, block.saturating_sub(1).max(1));
            let min_conf = std::env::var("INTARSIA_DSV41_SPEC_MIN_CONF").ok().and_then(|v| v.parse().ok()).unwrap_or(f32::NEG_INFINITY);
            eprintln!("dsv41: speculative decoding: up to {k} drafts a round, confidence floor {min_conf}");
            let choose = std::env::var("INTARSIA_DSV41_SPEC_CHOOSE").map_or(true, |v| v != "0");
            let mut round_ms = ROUND_MS;
            if let Ok(v) = std::env::var("INTARSIA_DSV41_SPEC_ROUND_MS") {
                for (slot, x) in round_ms.iter_mut().zip(v.split(',')) {
                    *slot = x.trim().parse().unwrap_or(*slot);
                }
            }
            let draft_ms = std::env::var("INTARSIA_DSV41_SPEC_DRAFT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(DRAFT_MS);
            if choose {
                eprintln!("dsv41: speculative decoding: k per round from the confidence head, pass ms {round_ms:?}, round overhead {draft_ms} ms");
            }
            SpecOpts { k, min_conf, trace: std::env::var("INTARSIA_DSV41_SPEC_TRACE").is_ok_and(|v| v == "1"), choose, round_ms, draft_ms }
        });
        let failed = Arc::new(Mutex::new(None));
        if let Some(p) = &peer {
            // the engine thread may sit in a kernel that never returns once the other rank is gone: the peer's reader thread
            // marks the engine failed (health 503, then the server exits for a restart) without waiting for it
            let f = failed.clone();
            p.on_lost(move |why| {
                eprintln!("dsv41: {why}; marking the engine failed");
                f.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(format!("tensor-parallel group failed: {why}"));
            });
        }
        let mut w = Worker { rt, peer, st, remote: remote_seq, codec: codec.clone(), eos, telemetry: telemetry.clone(), spec, failed: failed.clone() };
        // warm-up: one short pass on both ranks (kernel choice, scratch, NCCL buffers), then roll back
        let wt = Instant::now();
        w.forward(&[c.bos], true)?;
        w.forward(&[c.bos], true)?;
        w.truncate(0)?;
        eprintln!("dsv41: warm-up passes in {:.2} s", wt.elapsed().as_secs_f64());
        log_memory_after_warmup(&w.rt, avail0, 0);
        let (tx, rx): (Sender<Job>, Receiver<Job>) = channel();
        let f2 = failed.clone();
        std::thread::Builder::new()
            .name("dsv41-engine".into())
            .spawn(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for job in rx.iter() {
                        w.run(job);
                    }
                }));
                if let Err(p) = r {
                    let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                    *f2.lock().unwrap() = Some(format!("dsv41 engine worker panicked: {msg}"));
                }
            })
            .map_err(|e| Error::backend(format!("spawning the engine worker: {e}")))?;
        Ok(Dsv41Engine { tx: Mutex::new(tx), info, codec, telemetry, failed })
    }
}

impl Engine for Dsv41Engine {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }
    fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.codec.encode(text, add_special_tokens)
    }
    fn detokenize(&self, ids: &[u32]) -> Result<String> {
        self.codec.decode(ids, true)
    }
    fn apply_chat_template(&self, messages: &[Value], opts: &ChatOpts) -> Result<String> {
        self.codec.apply_chat_template(messages, opts)
    }
    fn tokenizer_json(&self) -> Option<String> {
        self.codec.tokenizer_json()
    }
    fn submit(&self, req: GenRequest) -> Result<Receiver<GenEvent>> {
        if let Some(f) = self.failed() {
            return Err(Error::backend(f));
        }
        let (tx, rx) = channel();
        self.telemetry.request_enqueued();
        self.tx.lock().unwrap().send(Job { req, tx }).map_err(|_| Error::backend("the engine worker is gone"))?;
        Ok(rx)
    }
    fn forward_logits_each(&self, _ids: &[u32], _cb: &mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<()> {
        Err(Error::Unsupported("deepseek_v41: no all-position logits path (use generate)".into()))
    }
    fn telemetry(&self) -> TelemetrySnapshot {
        self.telemetry.snapshot()
    }
    fn failed(&self) -> Option<String> {
        self.failed.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod f16_tests {
    #[test]
    fn f16_to_f32_matches_ieee() {
        for (h, f) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x7bff, 65504.0), (0x0001, 2f32.powi(-24)), (0x0200, 2f32.powi(-15)), (0x8000, -0.0), (0x3555, 0.333_251_95)] {
            assert_eq!(super::f16_to_f32(h).to_bits(), f.to_bits(), "{h:#06x}");
        }
        assert!(super::f16_to_f32(0x7e00).is_nan());
        assert_eq!(super::f16_to_f32(0x7c00), f32::INFINITY);
    }
}

#[cfg(test)]
mod spec_default_tests {
    use super::spec_on;

    #[test]
    fn unset_follows_the_sub_cache_and_a_set_value_decides() {
        assert!(spec_on(None, true));
        assert!(!spec_on(None, false));
        assert!(spec_on(Some("1"), false));
        assert!(!spec_on(Some("0"), true));
        assert!(!spec_on(Some("off"), true));
    }
}

#[cfg(test)]
mod choose_k_tests {
    use super::{choose_k, DRAFT_MS, ROUND_MS};

    #[test]
    fn confident_drafts_take_every_row_and_hopeless_ones_none() {
        assert_eq!(choose_k(&[20.0, 20.0, 20.0, 20.0], 4, &ROUND_MS, DRAFT_MS), 4);
        assert_eq!(choose_k(&[20.0, 20.0, 20.0, 20.0], 3, &ROUND_MS, DRAFT_MS), 3);
        assert_eq!(choose_k(&[-20.0, 20.0, 20.0, 20.0], 4, &ROUND_MS, DRAFT_MS), 0);
    }

    #[test]
    fn a_row_pays_only_above_its_share_of_the_round() {
        // two rows: tokens 1 + s against 50 ms, one row 1 against 40.3 ms, so a draft needs s > 0.241 (logit -1.145)
        let (r, d) = (&[31.3, 41.0][..], 9.0);
        assert_eq!(choose_k(&[-1.2], 1, r, d), 0);
        assert_eq!(choose_k(&[-1.1], 1, r, d), 1);
    }

    #[test]
    fn a_weak_draft_ends_the_prefix_but_a_strong_later_one_can_carry_it() {
        // the survival is a product, so a later draft counts only through the earlier ones
        assert_eq!(choose_k(&[20.0, -20.0, 20.0, 20.0], 4, &ROUND_MS, DRAFT_MS), 1);
        // short lists and tables bound k
        assert_eq!(choose_k(&[20.0], 4, &ROUND_MS, DRAFT_MS), 1);
        assert_eq!(choose_k(&[], 4, &ROUND_MS, DRAFT_MS), 0);
        assert_eq!(choose_k(&[20.0, 20.0], 4, &[31.3], DRAFT_MS), 0);
    }

    #[test]
    fn the_first_best_wins_a_tie() {
        // equal value at k = 0 and k = 1: 1 / 10 = 1.5 / 15, with a survival of exactly 0.5
        assert_eq!(choose_k(&[0.0], 1, &[5.0, 10.0], 5.0), 0);
    }
}
