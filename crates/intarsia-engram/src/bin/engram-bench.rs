//! engram-bench: measure and prepare the Engram table tier.
//!
//! - `raw`     device behaviour: random aligned O_DIRECT reads, sustained queue depth sweep and token-shaped bursts
//! - `token`   the store end to end: per decode step, hash the step's tokens, fetch both Engram layers, wait
//! - `hit`     hot-tier hit rate of a frequency-pinned set (and the full tier) on a held-out token stream (no disk)
//! - `profile` top rows by frequency from a training stream, for `token --profile`
//! - `trace`   tokenize text files into compressed-id streams (feature `tokenizer`)
//! - `verify`  the same rows through two layouts must be byte-identical
//! - `repack`  write a paged (IENGRPG1) shard

use clap::{Parser, Subcommand};
use intarsia_engram::hot::key;
use intarsia_engram::io::{make_backend, AlignedBuf, Completion, DirectFile, ReadOp};
use intarsia_engram::{
    BackendKind, EngramStore, HashLayout, HotTier, HotTierConfig, NgramState, ReadGranule, ScaleMode, StoreConfig,
    TableShard, RECORD_BYTES,
};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(about = "Engram table tier benchmarks and tools")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Random aligned reads straight at a file: QD sweep and token-shaped bursts.
    Raw {
        #[arg(long)]
        file: PathBuf,
        /// Read sizes (bytes).
        #[arg(long, value_delimiter = ',', default_value = "512,4096")]
        bs: Vec<u32>,
        /// Sustained queue depths.
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,24,32,48,64")]
        qd: Vec<usize>,
        /// Seconds per sustained point.
        #[arg(long, default_value_t = 3.0)]
        secs: f64,
        /// Burst sizes (reads issued at once, then waited for): 12 = one layer at TP=2, 24 = one token at TP=2.
        #[arg(long, value_delimiter = ',', default_value = "12,24,48,144")]
        burst: Vec<usize>,
        /// Gaps between bursts (ms): 0 = back to back; decode steps are 16-75 ms apart.
        #[arg(long, value_delimiter = ',', default_value = "0,16,75")]
        gap_ms: Vec<f64>,
        /// Bursts per (size, gap) point.
        #[arg(long, default_value_t = 2000)]
        bursts: usize,
        #[arg(long, default_value = "uring")]
        backend: String,
        /// Buffered reads (page cache) instead of O_DIRECT; pages are dropped afterwards.
        #[arg(long)]
        buffered: bool,
        /// Only read below this many bytes of the file (default: all).
        #[arg(long)]
        span_bytes: Option<u64>,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Per decode step: hash, fetch both Engram layers through the store, wait. Reports step latency.
    Token {
        /// Directory with the shard files or the official safetensors.
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        rank: usize,
        #[arg(long, default_value_t = 2)]
        world: usize,
        /// Compressed-id stream (u32 LE, u32::MAX = document break). Default: uniform random ids.
        #[arg(long)]
        trace: Option<PathBuf>,
        #[arg(long, default_value_t = 3000)]
        steps: usize,
        /// With --trace: every N steps jump to a random place in the stream (and restart the n-gram state), so a
        /// short run samples many documents instead of the first few.
        #[arg(long, default_value_t = 50)]
        trace_hop: usize,
        /// Tokens per step (1 = plain decode, 6 = DSpark verify block of 5 drafts + 1).
        #[arg(long, default_value_t = 1)]
        tokens_per_step: usize,
        #[arg(long, value_delimiter = ',', default_value = "0,16,75")]
        gap_ms: Vec<f64>,
        /// Hot tier size in MiB (0: none).
        #[arg(long, default_value_t = 0)]
        hot_mb: usize,
        #[arg(long, default_value_t = 0.25)]
        dynamic_frac: f32,
        /// Pinned rows (from `profile`).
        #[arg(long)]
        profile: Option<PathBuf>,
        /// Read a 512 B sector first each step to wake the drive.
        #[arg(long)]
        touch: bool,
        /// With --touch: issue the touch this long before the token id is known (e.g. at lm_head), us.
        #[arg(long, default_value_t = 0)]
        touch_lead_us: u64,
        /// A background thread reads one 4 KiB block every N ms to keep the drive out of low-power states.
        #[arg(long, default_value_t = 0.0)]
        keepalive_ms: f64,
        /// Speculative prefetch of the rows of K candidate next tokens (K-1 random + the true one, i.e. assumes
        /// the sampled token is in the top-k) `spec_lead_us` before the token id is known.
        #[arg(long, default_value_t = 0)]
        spec_topk: usize,
        /// Time between the speculative prefetch and the token id being known (lm_head + sampling), us.
        #[arg(long, default_value_t = 500)]
        spec_lead_us: u64,
        #[arg(long, default_value = "auto")]
        backend: String,
        #[arg(long, default_value = "sector")]
        granule: String,
        /// Safetensors only: resident | read.
        #[arg(long, default_value = "read")]
        scales: String,
        #[arg(long)]
        buffered: bool,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Hot-tier hit rate on a held-out stream (static frequency-pinned set from `train`, and the full tier).
    Hit {
        #[arg(long)]
        train: PathBuf,
        #[arg(long)]
        test: PathBuf,
        #[arg(long, default_value_t = 1)]
        rank: usize,
        #[arg(long, default_value_t = 2)]
        world: usize,
        /// Hot set sizes (rows on this rank).
        #[arg(long, value_delimiter = ',', default_value = "10000,100000,1000000,4000000,8000000")]
        sizes: Vec<usize>,
        #[arg(long, default_value_t = 0.25)]
        dynamic_frac: f32,
    },
    /// Top rows (this rank) by frequency in a stream; writes u64 LE keys (layer << 56 | row), most frequent first.
    Profile {
        #[arg(long)]
        train: PathBuf,
        #[arg(long, default_value_t = 1)]
        rank: usize,
        #[arg(long, default_value_t = 2)]
        world: usize,
        #[arg(long, default_value_t = 4_000_000)]
        rows: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Tokenize text files into train/test compressed-id streams (needs --features tokenizer).
    Trace {
        #[arg(long)]
        tokenizer: PathBuf,
        /// Use this token map (JSON list) instead of building one.
        #[arg(long)]
        token_map: Option<PathBuf>,
        #[arg(long)]
        out_train: PathBuf,
        #[arg(long)]
        out_test: PathBuf,
        /// Percent of files (by path hash) in the test split.
        #[arg(long, default_value_t = 50)]
        test_pct: u64,
        #[arg(long, default_value_t = 4_000_000)]
        max_tokens: usize,
        /// Files or directories (recursed; text-like extensions).
        inputs: Vec<PathBuf>,
    },
    /// Rows read through two table directories must be identical.
    Verify {
        #[arg(long)]
        a: PathBuf,
        #[arg(long)]
        b: PathBuf,
        #[arg(long, default_value_t = 1)]
        rank: usize,
        #[arg(long, default_value_t = 2)]
        world: usize,
        #[arg(long, default_value_t = 20000)]
        rows: usize,
    },
    /// Write a paged (IENGRPG1) shard for one layer and rank. Writes ~52 GB per layer at TP=2.
    Repack {
        #[arg(long)]
        src: PathBuf,
        #[arg(long)]
        layer: u32,
        #[arg(long, default_value_t = 1)]
        rank: usize,
        #[arg(long, default_value_t = 2)]
        world: usize,
        #[arg(long)]
        out: PathBuf,
        /// Only the first N rows (a test).
        #[arg(long)]
        limit: Option<u64>,
        /// Required: confirms the (large) write.
        #[arg(long)]
        yes: bool,
    },
}

fn main() -> io::Result<()> {
    match Cli::parse().cmd {
        Cmd::Raw { file, bs, qd, secs, burst, gap_ms, bursts, backend, buffered, span_bytes, json } => {
            raw(&file, &bs, &qd, secs, &burst, &gap_ms, bursts, &backend, buffered, span_bytes, json.as_deref())
        }
        Cmd::Token {
            dir,
            rank,
            world,
            trace,
            steps,
            trace_hop,
            tokens_per_step,
            gap_ms,
            hot_mb,
            dynamic_frac,
            profile,
            touch,
            touch_lead_us,
            keepalive_ms,
            spec_topk,
            spec_lead_us,
            backend,
            granule,
            scales,
            buffered,
            json,
        } => token(TokenArgs {
            dir,
            rank,
            world,
            trace,
            steps,
            trace_hop,
            tokens_per_step,
            gap_ms,
            hot_mb,
            dynamic_frac,
            profile,
            touch,
            touch_lead_us,
            keepalive_ms,
            spec_topk,
            spec_lead_us,
            backend,
            granule,
            scales,
            buffered,
            json,
        }),
        Cmd::Hit { train, test, rank, world, sizes, dynamic_frac } => hit(&train, &test, rank, world, &sizes, dynamic_frac),
        Cmd::Profile { train, rank, world, rows, out } => profile(&train, rank, world, rows, &out),
        Cmd::Trace { tokenizer, token_map, out_train, out_test, test_pct, max_tokens, inputs } => {
            trace(&tokenizer, token_map.as_deref(), &out_train, &out_test, test_pct, max_tokens, &inputs)
        }
        Cmd::Verify { a, b, rank, world, rows } => verify(&a, &b, rank, world, rows),
        Cmd::Repack { src, layer, rank, world, out, limit, yes } => repack(&src, layer, rank, world, &out, limit, yes),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// helpers

fn backend_kind(s: &str) -> BackendKind {
    match s {
        "uring" => BackendKind::Uring { sqpoll_idle_ms: None },
        "sqpoll" => BackendKind::Uring { sqpoll_idle_ms: Some(50) },
        "inline" => BackendKind::Pool { threads: 0 },
        s if s.starts_with("pool") => BackendKind::Pool { threads: s[4..].parse().unwrap_or(16) },
        _ => BackendKind::Auto,
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Percentiles of durations, in microseconds.
#[derive(Clone, Copy, Debug)]
struct Pct {
    n: usize,
    p50: f64,
    p90: f64,
    p99: f64,
    p999: f64,
    max: f64,
    mean: f64,
}

fn pct(v: &mut [Duration]) -> Pct {
    v.sort_unstable();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    let at = |q: f64| us(v[((v.len() as f64 * q) as usize).min(v.len() - 1)]);
    let mean = v.iter().map(|&d| us(d)).sum::<f64>() / v.len().max(1) as f64;
    Pct { n: v.len(), p50: at(0.5), p90: at(0.9), p99: at(0.99), p999: at(0.999), max: us(*v.last().unwrap()), mean }
}

impl Pct {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({"n": self.n, "p50_us": self.p50, "p90_us": self.p90, "p99_us": self.p99,
            "p999_us": self.p999, "max_us": self.max, "mean_us": self.mean})
    }
    fn row(&self) -> String {
        format!("{:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>8.0}", self.p50, self.p90, self.p99, self.p999, self.max)
    }
}

/// MemAvailable in bytes (Linux), to refuse a run that would push a shared box under its floor.
fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    Some(l.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024)
}

fn guard_memory(need: u64) -> io::Result<()> {
    if let Some(avail) = mem_available() {
        let floor = 10u64 << 30;
        if avail < floor + need {
            return Err(io::Error::other(format!(
                "refusing: MemAvailable {:.1} GiB < 10 GiB floor + {:.1} GiB this run needs",
                avail as f64 / (1u64 << 30) as f64,
                need as f64 / (1u64 << 30) as f64
            )));
        }
        eprintln!("MemAvailable {:.1} GiB, this run needs ~{:.2} GiB", avail as f64 / (1u64 << 30) as f64, need as f64 / (1u64 << 30) as f64);
    }
    Ok(())
}

fn write_json(path: Option<&Path>, v: &serde_json::Value) -> io::Result<()> {
    if let Some(p) = path {
        std::fs::write(p, serde_json::to_string_pretty(v).unwrap())?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}

fn host_facts() -> serde_json::Value {
    let rd = |p: &str| std::fs::read_to_string(p).map(|s| s.trim().to_string()).unwrap_or_default();
    serde_json::json!({
        "kernel": rd("/proc/sys/kernel/osrelease"),
        "apst_max_latency_us": rd("/sys/module/nvme_core/parameters/default_ps_max_latency_us"),
        "nvme_model": rd("/sys/block/nvme0n1/device/model"),
        "logical_block": rd("/sys/block/nvme0n1/queue/logical_block_size"),
        "scheduler": rd("/sys/block/nvme0n1/queue/scheduler"),
    })
}

// ---------------------------------------------------------------------------------------------------------------
// raw

#[allow(clippy::too_many_arguments)]
fn raw(
    file: &Path,
    bss: &[u32],
    qds: &[usize],
    secs: f64,
    bursts_sizes: &[usize],
    gaps: &[f64],
    bursts: usize,
    backend: &str,
    buffered: bool,
    span_bytes: Option<u64>,
    json: Option<&Path>,
) -> io::Result<()> {
    let f = DirectFile::open(file, !buffered)?;
    let len = f.file.metadata()?.len();
    let span = span_bytes.unwrap_or(len).min(len);
    let maxq = qds.iter().copied().chain(bursts_sizes.iter().copied()).max().unwrap_or(64);
    let mut be = make_backend(backend_kind(backend), maxq.max(8))?;
    let slot = 8192usize;
    let mem = AlignedBuf::new(maxq * slot);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    eprintln!("raw: {} ({:.1} GB, reading below {:.1} GB) backend {} direct {}", file.display(), len as f64 / 1e9, span as f64 / 1e9, be.name(), f.direct);
    let mut out = serde_json::json!({"file": file, "backend": be.name(), "direct": f.direct, "host": host_facts(), "sustained": [], "bursts": []});
    let mut comps: Vec<Completion> = Vec::new();
    let spin = Duration::from_millis(5);
    let op_at = |rng: &mut Rng, bs: u32, s: usize, tag: u64| ReadOp {
        fd: f.fd(),
        offset: (rng.next() % (span / bs as u64 - 1)) * bs as u64,
        len: bs,
        // SAFETY: slot `s` belongs to this op until reaped.
        buf: unsafe { mem.as_ptr().add(s * slot) },
        tag,
    };
    println!("sustained random reads ({})\n{:>6} {:>4} {:>10} {:>9} | per-IO latency us: {:>6} {:>8} {:>8} {:>8} {:>8}", if f.direct { "O_DIRECT" } else { "buffered" }, "bs", "qd", "IOPS", "MB/s", "p50", "p90", "p99", "p99.9", "max");
    for &bs in bss {
        for &qd in qds {
            let mut start: Vec<Instant> = vec![Instant::now(); qd];
            let mut lats = Vec::new();
            let t0 = Instant::now();
            let mut ops = Vec::with_capacity(qd);
            for s in 0..qd {
                ops.push(op_at(&mut rng, bs, s, s as u64));
                start[s] = Instant::now();
            }
            be.submit(&ops)?;
            let mut done = 0u64;
            let mut errors = 0u64;
            while t0.elapsed().as_secs_f64() < secs {
                comps.clear();
                be.reap(&mut comps, true, spin)?;
                ops.clear();
                for c in &comps {
                    let s = c.tag as usize;
                    let now = Instant::now();
                    lats.push(now - start[s]);
                    if c.res < bs as i64 {
                        errors += 1;
                    }
                    done += 1;
                    ops.push(op_at(&mut rng, bs, s, s as u64));
                    start[s] = now;
                }
                be.submit(&ops)?;
            }
            while be.in_flight() > 0 {
                comps.clear();
                be.reap(&mut comps, true, spin)?;
            }
            let el = t0.elapsed().as_secs_f64();
            let p = pct(&mut lats);
            let iops = done as f64 / el;
            println!("{bs:>6} {qd:>4} {iops:>10.0} {:>9.1} | {}{}", iops * bs as f64 / 1e6, " ".repeat(19), p.row());
            if errors > 0 {
                println!("  {errors} short or failed reads");
            }
            out["sustained"].as_array_mut().unwrap().push(serde_json::json!({"bs": bs, "qd": qd, "iops": iops, "lat": p.json(), "errors": errors}));
        }
    }
    println!("\ntoken-shaped bursts: N reads at once, wait for all (latency of the whole burst)\n{:>6} {:>5} {:>7} | burst us: {:>6} {:>8} {:>8} {:>8} {:>8}", "bs", "burst", "gap ms", "p50", "p90", "p99", "p99.9", "max");
    for &bs in bss {
        for &b in bursts_sizes {
            for &gap in gaps {
                let n = if gap > 0.0 { bursts.min((20_000.0 / gap.max(1.0)) as usize).max(100) } else { bursts };
                let mut lats = Vec::with_capacity(n);
                let mut ops = Vec::with_capacity(b);
                for _ in 0..n {
                    if gap > 0.0 {
                        std::thread::sleep(Duration::from_secs_f64(gap / 1e3));
                    }
                    ops.clear();
                    for s in 0..b {
                        ops.push(op_at(&mut rng, bs, s, s as u64));
                    }
                    let t = Instant::now();
                    be.submit(&ops)?;
                    let mut got = 0;
                    while got < b {
                        comps.clear();
                        be.reap(&mut comps, true, spin)?;
                        got += comps.len();
                    }
                    lats.push(t.elapsed());
                }
                let p = pct(&mut lats);
                println!("{bs:>6} {b:>5} {gap:>7.0} | {}{}", " ".repeat(10), p.row());
                out["bursts"].as_array_mut().unwrap().push(serde_json::json!({"bs": bs, "burst": b, "gap_ms": gap, "lat": p.json()}));
            }
        }
    }
    if buffered {
        f.drop_cache();
    }
    write_json(json, &out)
}

// ---------------------------------------------------------------------------------------------------------------
// token streams

/// A compressed-id stream: u32 LE, u32::MAX = document break.
fn read_stream(p: &Path) -> io::Result<Vec<u32>> {
    let b = std::fs::read(p)?;
    Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Hash a stream; calls `f(position, rows[layer*cols + col])` per token.
fn hash_stream(layout: &HashLayout, ids: &[u32], mut f: impl FnMut(&[u64])) {
    let mut st = NgramState::default();
    let mut rows = vec![0u64; layout.rows_per_token()];
    for &t in ids {
        if t == u32::MAX {
            st.reset();
            continue;
        }
        st.push(layout, Some(t), &mut rows);
        f(&rows);
    }
}

struct TokenArgs {
    dir: PathBuf,
    rank: usize,
    world: usize,
    trace: Option<PathBuf>,
    steps: usize,
    trace_hop: usize,
    tokens_per_step: usize,
    gap_ms: Vec<f64>,
    hot_mb: usize,
    dynamic_frac: f32,
    profile: Option<PathBuf>,
    touch: bool,
    touch_lead_us: u64,
    keepalive_ms: f64,
    spec_topk: usize,
    spec_lead_us: u64,
    backend: String,
    granule: String,
    scales: String,
    buffered: bool,
    json: Option<PathBuf>,
}

fn open_shards(dir: &Path, layout: &HashLayout, rank: usize, world: usize, scales: ScaleMode, direct: bool) -> io::Result<Vec<TableShard>> {
    layout
        .layer_ids()
        .iter()
        .enumerate()
        .map(|(i, &l)| TableShard::open_dir(dir, l as u32, rank, world, layout.rank_rows(i, rank, world), scales, direct))
        .collect()
}

fn token(a: TokenArgs) -> io::Result<()> {
    let layout = HashLayout::dsv41();
    let scales = if a.scales == "resident" { ScaleMode::Resident } else { ScaleMode::Read };
    let granule = if a.granule == "page" { ReadGranule::Page } else { ReadGranule::Sector };
    let hot_bytes = a.hot_mb << 20;
    let scale_bytes: u64 = if scales == ScaleMode::Resident {
        (0..2).map(|i| layout.rank_rows(i, a.rank, a.world)).map(|r| (r.end - r.start) * 8).sum()
    } else {
        0
    };
    guard_memory(hot_bytes as u64 + scale_bytes + (256 << 20))?;
    let t_open = Instant::now();
    let shards = open_shards(&a.dir, &layout, a.rank, a.world, scales, !a.buffered)?;
    let open_s = t_open.elapsed().as_secs_f64();
    for s in &shards {
        eprintln!("layer {}: {:?} rows {:?} direct {} ({})", s.layer_id, s.kind, s.rows, s.direct(), s.path.display());
    }
    let cfg = StoreConfig {
        backend: backend_kind(&a.backend),
        granule,
        hot: HotTierConfig { capacity_bytes: hot_bytes, dynamic_fraction: a.dynamic_frac, ..Default::default() },
        ..Default::default()
    };
    let mut store = EngramStore::new(shards, cfg)?;
    let mut preload_s = 0.0;
    if let Some(p) = &a.profile {
        let keys: Vec<(usize, u64)> = std::fs::read(p)?
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .map(|k| ((k >> 56) as usize, k & ((1 << 56) - 1)))
            .collect();
        let t = Instant::now();
        let n = store.preload(&keys)?;
        preload_s = t.elapsed().as_secs_f64();
        eprintln!("pinned {n} rows ({:.0} MB) in {preload_s:.1} s", (n * RECORD_BYTES) as f64 / 1e6);
    }
    // the token source
    let ids: Vec<u32> = match &a.trace {
        Some(p) => read_stream(p)?,
        None => {
            let mut r = Rng(0xDEAD_BEEF_CAFE_F00D);
            (0..a.steps * a.tokens_per_step * a.gap_ms.len() + 64).map(|_| (r.next() % 99_092) as u32).collect()
        }
    };
    let cols = layout.cols();
    let nl = layout.n_layers();
    let mut out = serde_json::json!({"dir": a.dir, "rank": a.rank, "world": a.world, "backend": store.backend_name(),
        "granule": a.granule, "scales": a.scales, "hot_mb": a.hot_mb, "profile": a.profile, "trace": a.trace,
        "tokens_per_step": a.tokens_per_step, "touch": a.touch, "touch_lead_us": a.touch_lead_us, "keepalive_ms": a.keepalive_ms, "spec_topk": a.spec_topk, "spec_lead_us": a.spec_lead_us, "open_s": open_s, "preload_s": preload_s,
        "host": host_facts(), "points": []});
    println!("per step ({} token(s), {} rows fetched/rank of which {} owned): latency from fetch to rows in, us", a.tokens_per_step, nl * cols * a.tokens_per_step, store.shards().iter().map(|_| cols / a.world).sum::<usize>() * a.tokens_per_step);
    println!("{:>6} | {:<44} | {:<44} | {:>6} {:>7}", "gap ms", "layer 1 (needed after layer 0)   p50 p90 p99 p99.9 max", "step (both layers)   p50 p90 p99 p99.9 max", "hot %", "reads/t");
    let _keepalive = (a.keepalive_ms > 0.0).then(|| keepalive(store.shards()[0].path.clone(), a.keepalive_ms));
    let mut st = NgramState::default();
    let mut pos = 0usize;
    let mut spec_rng = Rng(0x5EC0_0000_0000_0001);
    let mut hop_rng = Rng(0x4057_0000_0000_0003);
    let mut rows = vec![0u64; layout.rows_per_token()];
    let mut per_layer: Vec<Vec<u64>> = vec![Vec::new(); nl];
    let mut w = vec![0u8; 256 * cols * a.tokens_per_step];
    let mut s8 = vec![0u8; 8 * cols * a.tokens_per_step];
    for &gap in &a.gap_ms {
        let before = store.stats();
        let hot_before = store.hot_stats();
        let mut l1 = Vec::with_capacity(a.steps);
        let mut step = Vec::with_capacity(a.steps);
        let mut tokens = 0usize;
        for si in 0..a.steps {
            if a.trace.is_some() && a.trace_hop > 0 && si % a.trace_hop == 0 {
                pos = (hop_rng.next() % ids.len() as u64) as usize;
                st.reset();
            }
            if gap > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(gap / 1e3));
            }
            if a.touch {
                store.touch()?;
                let t = Instant::now();
                while t.elapsed() < Duration::from_micros(a.touch_lead_us) {
                    std::hint::spin_loop();
                    let _ = store.poll(intarsia_engram::Ticket::NONE);
                }
            }
            if a.spec_topk > 0 {
                // candidates for the next token: the true one (peeked from the stream) plus K-1 others
                let mut cand: Vec<u32> = (1..a.spec_topk).map(|_| (spec_rng.next() % 99_092) as u32).collect();
                let mut p = pos;
                while ids[p % ids.len()] == u32::MAX {
                    p += 1;
                }
                cand.push(ids[p % ids.len()]);
                let st_peek = if ids[pos % ids.len()] == u32::MAX { NgramState::default() } else { st };
                for &c in &cand {
                    st_peek.peek(&layout, Some(c), &mut rows);
                    for l in 0..nl {
                        store.prefetch(l, &rows[l * cols..(l + 1) * cols])?;
                    }
                }
                let t = Instant::now();
                while t.elapsed() < Duration::from_micros(a.spec_lead_us) {
                    std::hint::spin_loop();
                }
            }
            for v in per_layer.iter_mut() {
                v.clear();
            }
            for _ in 0..a.tokens_per_step {
                loop {
                    let t = ids[pos % ids.len()];
                    pos += 1;
                    if t == u32::MAX {
                        st.reset();
                        continue;
                    }
                    st.push(&layout, Some(t), &mut rows);
                    break;
                }
                for (l, v) in per_layer.iter_mut().enumerate() {
                    v.extend_from_slice(&rows[l * cols..(l + 1) * cols]);
                }
                tokens += 1;
            }
            let tickets: Vec<_> = (0..nl).map(|l| store.fetch(l, &per_layer[l])).collect::<io::Result<_>>()?;
            let mut lat = Duration::ZERO;
            for (l, t) in tickets.into_iter().enumerate() {
                let info = store.wait(t, &mut w, &mut s8)?;
                if l == 0 {
                    l1.push(info.latency);
                }
                lat = lat.max(info.latency);
            }
            step.push(lat);
        }
        let s = store.stats();
        let hs = store.hot_stats();
        let lookups = hs.lookups - hot_before.lookups;
        let hot_pct = if lookups > 0 { 100.0 * (hs.hits - hot_before.hits) as f64 / lookups as f64 } else { 0.0 };
        let reads_per_tok = (s.row_reads - before.row_reads) as f64 / tokens as f64;
        let (p1, ps) = (pct(&mut l1), pct(&mut step));
        println!("{gap:>6.0} | {:<44} | {:<44} | {hot_pct:>6.1} {reads_per_tok:>7.1}", p1.row(), ps.row());
        out["points"].as_array_mut().unwrap().push(serde_json::json!({"gap_ms": gap, "layer1": p1.json(), "step": ps.json(),
            "hot_hit_pct": hot_pct, "row_reads_per_token": reads_per_tok, "io_bytes": s.io_bytes - before.io_bytes,
            "staged_hits": s.staged_hits - before.staged_hits, "joined": s.joined - before.joined, "failed": s.failed_reads}));
    }
    out["resident_bytes"] = store.resident_bytes().into();
    write_json(a.json.as_deref(), &out)
}

/// A thread that reads one random 4 KiB block of `path` (O_DIRECT) every `ms`; stops when the guard drops.
struct Keepalive(std::sync::Arc<std::sync::atomic::AtomicBool>, Option<std::thread::JoinHandle<()>>);
impl Drop for Keepalive {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.1.take() {
            let _ = h.join();
        }
    }
}
fn keepalive(path: PathBuf, ms: f64) -> Keepalive {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s2 = stop.clone();
    let h = std::thread::spawn(move || {
        let Ok(f) = DirectFile::open(&path, true) else { return };
        let len = f.file.metadata().map(|m| m.len()).unwrap_or(1 << 20);
        let mut buf = AlignedBuf::new(4096);
        let mut r = Rng(0x4B45_4550_414C_4956);
        while !s2.load(std::sync::atomic::Ordering::Relaxed) {
            let off = (r.next() % (len / 4096 - 1)) * 4096;
            let _ = intarsia_engram::io::pread_full(f.fd(), buf.as_mut_slice(), off);
            std::thread::sleep(Duration::from_secs_f64(ms / 1e3));
        }
    });
    Keepalive(stop, Some(h))
}

// ---------------------------------------------------------------------------------------------------------------
// hit rate and profiles

/// Per (layer, row) owned by this rank: lookup counts in a stream.
fn count_rows(layout: &HashLayout, ids: &[u32], rank: usize, world: usize) -> HashMap<u64, u32, intarsia_engram::hot::FxBuild> {
    let cols = layout.rank_columns(rank, world);
    let c = layout.cols();
    let mut m: HashMap<u64, u32, intarsia_engram::hot::FxBuild> = HashMap::default();
    hash_stream(layout, ids, |rows| {
        for l in 0..layout.n_layers() {
            for col in cols.clone() {
                *m.entry(key(l, rows[l * c + col])).or_insert(0) += 1;
            }
        }
    });
    m
}

fn top_rows(counts: &HashMap<u64, u32, intarsia_engram::hot::FxBuild>, n: usize) -> Vec<u64> {
    let mut v: Vec<(u32, u64)> = counts.iter().map(|(&k, &c)| (c, k)).collect();
    let n = n.min(v.len());
    if n < v.len() {
        v.select_nth_unstable_by(n, |a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        v.truncate(n);
    }
    v.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    v.into_iter().map(|x| x.1).collect()
}

fn hit(train: &Path, test: &Path, rank: usize, world: usize, sizes: &[usize], dyn_frac: f32) -> io::Result<()> {
    let layout = HashLayout::dsv41();
    let (tr, te) = (read_stream(train)?, read_stream(test)?);
    let ntr = tr.iter().filter(|&&t| t != u32::MAX).count();
    let nte = te.iter().filter(|&&t| t != u32::MAX).count();
    let counts = count_rows(&layout, &tr, rank, world);
    let cols = layout.rank_columns(rank, world);
    let c = layout.cols();
    println!("train {ntr} tokens ({} distinct rows on rank {rank}), test {nte} tokens; rank {rank}/{world} owns columns {cols:?}", counts.len());
    println!("{:>10} {:>9} | {:>8} {:>8} {:>8} {:>8} | {:>12}", "hot rows", "MB", "static%", "2-gram%", "3-gram%", "4-gram%", "tier(+dyn)%");
    let max = sizes.iter().copied().max().unwrap_or(0);
    let ranked = top_rows(&counts, max);
    for &n in sizes {
        let set: std::collections::HashSet<u64, intarsia_engram::hot::FxBuild> = ranked.iter().take(n).copied().collect();
        let (mut hits, mut look) = ([0u64; 3], [0u64; 3]);
        // the full tier: pinned top (1 - dyn) * n, dynamic TinyLFU on the rest, simulated on the test stream
        let mut tier = HotTier::new(HotTierConfig { capacity_bytes: n * RECORD_BYTES, dynamic_fraction: dyn_frac, sketch_log2: 22 });
        let rec = [0u8; RECORD_BYTES];
        for &k in ranked.iter().take(tier.pinned_capacity()) {
            tier.pin(k, &rec);
        }
        hash_stream(&layout, &te, |rows| {
            for l in 0..layout.n_layers() {
                for col in cols.clone() {
                    let k = key(l, rows[l * c + col]);
                    let g = col / layout.n_heads();
                    look[g] += 1;
                    hits[g] += set.contains(&k) as u64;
                    if tier.lookup(k).is_none() {
                        tier.offer(k, &rec);
                    }
                }
            }
        });
        let pc = |h: u64, l: u64| if l == 0 { f64::NAN } else { 100.0 * h as f64 / l as f64 };
        let (h, l): (u64, u64) = (hits.iter().sum(), look.iter().sum());
        println!(
            "{n:>10} {:>9.0} | {:>8.1} {:>8.1} {:>8.1} {:>8.1} | {:>12.1}",
            (n * RECORD_BYTES) as f64 / 1e6,
            pc(h, l),
            pc(hits[0], look[0]),
            pc(hits[1], look[1]),
            pc(hits[2], look[2]),
            100.0 * tier.stats.hit_rate()
        );
    }
    Ok(())
}

fn profile(train: &Path, rank: usize, world: usize, n: usize, out: &Path) -> io::Result<()> {
    let layout = HashLayout::dsv41();
    let ids = read_stream(train)?;
    let counts = count_rows(&layout, &ids, rank, world);
    let top = top_rows(&counts, n);
    let mut b = Vec::with_capacity(top.len() * 8);
    for k in &top {
        b.extend_from_slice(&k.to_le_bytes());
    }
    std::fs::write(out, b)?;
    println!("{} rows ({} distinct seen) -> {}", top.len(), counts.len(), out.display());
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// trace

#[cfg(feature = "tokenizer")]
fn trace(tokenizer: &Path, token_map: Option<&Path>, out_train: &Path, out_test: &Path, test_pct: u64, max_tokens: usize, inputs: &[PathBuf]) -> io::Result<()> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| io::Error::other(e.to_string()))?;
    let map = match token_map {
        Some(p) => intarsia_engram::TokenMap::load_json(p)?,
        None => intarsia_engram::token_map::build(&tok)?,
    };
    eprintln!("token map: {} ids -> {} compressed, sha256(u32le) {}", map.map.len(), map.compressed_vocab, map.sha256_u32le());
    let exts = ["md", "txt", "rs", "py", "toml", "json", "sh", "c", "h", "cpp", "cu", "go", "js", "ts", "html", "tex", "rst", "yaml", "yml", "zig", "swift"];
    let mut files = Vec::new();
    fn walk(p: &Path, exts: &[&str], out: &mut Vec<PathBuf>) {
        if p.is_dir() {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                return;
            }
            if let Ok(rd) = std::fs::read_dir(p) {
                let mut v: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
                v.sort();
                for c in v {
                    walk(&c, exts, out);
                }
            }
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| exts.contains(&e)) {
            out.push(p.to_path_buf());
        }
    }
    for i in inputs {
        walk(i, &exts, &mut files);
    }
    let (mut tr, mut te) = (Vec::<u8>::new(), Vec::<u8>::new());
    let (mut ntr, mut nte, mut nfiles) = (0usize, 0usize, 0usize);
    for f in files {
        if ntr + nte >= max_tokens {
            break;
        }
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        if text.len() > 2_000_000 || text.is_empty() {
            continue;
        }
        let enc = tok.encode(text.as_str(), false).map_err(|e| io::Error::other(e.to_string()))?;
        // FNV of the path picks the split
        let h = f.to_string_lossy().bytes().fold(0xCBF2_9CE4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01B3));
        let (buf, n) = if h % 100 < test_pct { (&mut te, &mut nte) } else { (&mut tr, &mut ntr) };
        for &id in enc.get_ids() {
            buf.extend_from_slice(&map.get(id).unwrap_or(0).to_le_bytes());
        }
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        *n += enc.get_ids().len();
        nfiles += 1;
    }
    std::fs::write(out_train, tr)?;
    std::fs::write(out_test, te)?;
    println!("{nfiles} files: train {ntr} tokens -> {}, test {nte} tokens -> {}", out_train.display(), out_test.display());
    Ok(())
}

#[cfg(not(feature = "tokenizer"))]
fn trace(_: &Path, _: Option<&Path>, _: &Path, _: &Path, _: u64, _: usize, _: &[PathBuf]) -> io::Result<()> {
    Err(io::Error::other("trace needs --features tokenizer"))
}

// ---------------------------------------------------------------------------------------------------------------
// verify / repack

fn verify(a: &Path, b: &Path, rank: usize, world: usize, n: usize) -> io::Result<()> {
    let layout = HashLayout::dsv41();
    let sa = open_shards(a, &layout, rank, world, ScaleMode::Read, true)?;
    let sb = open_shards(b, &layout, rank, world, ScaleMode::Read, true)?;
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    let (mut ra, mut rb) = (vec![0u8; RECORD_BYTES], vec![0u8; RECORD_BYTES]);
    for (x, y) in sa.iter().zip(&sb) {
        let mut bad = 0;
        let span = x.rows.end - x.rows.start;
        let mut rows: Vec<u64> = (0..n).map(|_| x.rows.start + rng.next() % span).collect();
        rows.extend([x.rows.start, x.rows.start + 1, x.rows.end - 1]);
        for &r in &rows {
            x.read_row_sync(r, &mut ra)?;
            y.read_row_sync(r, &mut rb)?;
            if ra != rb {
                bad += 1;
                if bad < 4 {
                    eprintln!("layer {} row {r} differs", x.layer_id);
                }
            }
        }
        println!("layer {}: {:?} vs {:?}, rows {:?}: {} rows compared, {bad} differ", x.layer_id, x.kind, y.kind, x.rows, rows.len());
        if bad > 0 {
            return Err(io::Error::other("layouts disagree"));
        }
    }
    Ok(())
}

fn repack(src: &Path, layer: u32, rank: usize, world: usize, out: &Path, limit: Option<u64>, yes: bool) -> io::Result<()> {
    let layout = HashLayout::dsv41();
    let li = layout.layer_index(layer as usize).ok_or_else(|| io::Error::other("not an Engram layer"))?;
    let rows = layout.rank_rows(li, rank, world);
    let rows = match limit {
        Some(n) => rows.start..(rows.start + n).min(rows.end),
        None => rows,
    };
    let bytes = 4096 + (rows.end - rows.start).div_ceil(15) * 4096;
    println!("repack layer {layer} rank {rank}/{world} rows {rows:?} -> {} ({:.1} GB)", out.display(), bytes as f64 / 1e9);
    if !yes {
        return Err(io::Error::other("pass --yes to write it"));
    }
    let shard = TableShard::open_dir(src, layer, rank, world, layout.rank_rows(li, rank, world), ScaleMode::Read, true)?;
    let t = Instant::now();
    let mut last = Instant::now();
    let n = intarsia_engram::layout::write_paged(&shard, rows, out, 1 << 18, |done, all| {
        if last.elapsed().as_secs() >= 10 {
            last = Instant::now();
            eprintln!("  {done}/{all} rows ({:.0} MB/s)", done as f64 * 264.0 / 1e6 / t.elapsed().as_secs_f64());
        }
    })?;
    println!("wrote {n} B in {:.0} s", t.elapsed().as_secs_f64());
    io::stdout().flush()
}
