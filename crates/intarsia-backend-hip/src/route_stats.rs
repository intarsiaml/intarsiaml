//! Lane hprof: `INTARSIA_HIP_ROUTE_STATS=1` counts the distinct routed experts per MoE layer per batched decode step.
//!
//! Off by default (one `Option` check per launch). On: after each eager `moe_block_bf16_grouped` launch the stream is
//! synchronised and the block's top-k ids `[T, K]` (scratch offset `r256(T*K*4)`, kernels/hip/moe_block_grouped.hip `layout`)
//! are read back. Per row count T it keeps the distinct experts, the weight passes the grouped kernel makes (an expert
//! chosen by n pairs is read ceil(n / 8) times, `kPairChunk`, moe.hip) and the maximum pairs on one expert, summed per layer
//! (keyed by the layer's gate_up weight address). Prints the cumulative table to stderr every `PRINT_EVERY` launches.
//! Only eager steps reach `launch` (`INTARSIA_BATCH_GRAPHS=0`, `--profile-every 0`); a replayed graph shows nothing.
//!
//! `INTARSIA_HIP_EXIT_FILE=path`: once that file exists the process calls exit(0), so a `rocprofv3` wrapper flushes its trace
//! (a SIGTERM to the server kills it without running the profiler's finaliser).

use std::collections::{BTreeMap, HashMap};

const PRINT_EVERY: u64 = 4000;
const PAIR_CHUNK: u32 = 8;

#[derive(Default, Clone)]
struct Cell {
    n: u64,
    distinct_sum: u64,
    distinct_min: u32,
    distinct_max: u32,
    passes_sum: u64,
    maxpairs_sum: u64,
    maxpairs_max: u32,
    hist: BTreeMap<u32, u64>,
}

#[derive(Default)]
pub(crate) struct RouteStats {
    /// (rows, layer key) -> cell; layers numbered in first-seen order per rows
    cells: HashMap<(usize, usize), Cell>,
    layer_ids: HashMap<usize, usize>,
    calls: u64,
}

impl RouteStats {
    pub(crate) fn from_env() -> Option<std::sync::Mutex<RouteStats>> {
        std::env::var_os("INTARSIA_HIP_ROUTE_STATS").filter(|v| v != "0").map(|_| std::sync::Mutex::new(RouteStats::default()))
    }

    /// One launch: `ids` is the `[rows, k]` top-k id block, `layer` any stable per-layer key.
    pub(crate) fn record(&mut self, rows: usize, layer: usize, ids: &[i32]) {
        let next = self.layer_ids.len();
        let li = *self.layer_ids.entry(layer).or_insert(next);
        let mut cnt: HashMap<i32, u32> = HashMap::new();
        for &e in ids {
            *cnt.entry(e).or_insert(0) += 1;
        }
        let distinct = cnt.len() as u32;
        let passes: u32 = cnt.values().map(|&c| c.div_ceil(PAIR_CHUNK)).sum();
        let maxp = cnt.values().copied().max().unwrap_or(0);
        let c = self.cells.entry((rows, li)).or_default();
        if c.n == 0 {
            c.distinct_min = u32::MAX;
        }
        c.n += 1;
        c.distinct_sum += distinct as u64;
        c.distinct_min = c.distinct_min.min(distinct);
        c.distinct_max = c.distinct_max.max(distinct);
        c.passes_sum += passes as u64;
        c.maxpairs_sum += maxp as u64;
        c.maxpairs_max = c.maxpairs_max.max(maxp);
        *c.hist.entry(distinct).or_insert(0) += 1;
        self.calls += 1;
        if self.calls % PRINT_EVERY == 0 {
            self.print();
        }
    }

    pub(crate) fn print(&self) {
        let mut rows: Vec<usize> = self.cells.keys().map(|k| k.0).collect();
        rows.sort_unstable();
        rows.dedup();
        for r in rows {
            let mut agg = Cell { distinct_min: u32::MAX, ..Default::default() };
            let mut per_layer: Vec<(usize, &Cell)> = self.cells.iter().filter(|(k, _)| k.0 == r).map(|(k, c)| (k.1, c)).collect();
            per_layer.sort_by_key(|x| x.0);
            for (_, c) in &per_layer {
                agg.n += c.n;
                agg.distinct_sum += c.distinct_sum;
                agg.distinct_min = agg.distinct_min.min(c.distinct_min);
                agg.distinct_max = agg.distinct_max.max(c.distinct_max);
                agg.passes_sum += c.passes_sum;
                agg.maxpairs_sum += c.maxpairs_sum;
                agg.maxpairs_max = agg.maxpairs_max.max(c.maxpairs_max);
            }
            let n = agg.n.max(1) as f64;
            eprintln!(
                "route-stats rows={r} layer-calls={} distinct mean={:.2} min={} max={} weight-passes mean={:.2} max-pairs-on-one-expert mean={:.2} max={}",
                agg.n,
                agg.distinct_sum as f64 / n,
                agg.distinct_min,
                agg.distinct_max,
                agg.passes_sum as f64 / n,
                agg.maxpairs_sum as f64 / n,
                agg.maxpairs_max
            );
            let layers: Vec<String> = per_layer.iter().map(|(l, c)| format!("{l}:{:.1}", c.distinct_sum as f64 / c.n.max(1) as f64)).collect();
            eprintln!("route-stats rows={r} per-layer-mean-distinct {}", layers.join(" "));
        }
    }
}

/// `INTARSIA_HIP_EXIT_FILE` (see the module doc): the watcher thread is started once per process.
pub(crate) fn exit_watch() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if let Some(path) = std::env::var_os("INTARSIA_HIP_EXIT_FILE").filter(|v| !v.is_empty()) {
            std::thread::spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if std::path::Path::new(&path).exists() {
                    eprintln!("hip exit file {path:?} present: exiting");
                    std::process::exit(0);
                }
            });
        }
    });
}
