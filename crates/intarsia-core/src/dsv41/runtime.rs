//! Executes the DeepSeek-V4.1 graph ([`super::graph`]) on a real backend: one rank of a tensor-parallel group (or the only
//! rank), one sequence at a time, eagerly (every step launched in order on stream 0; no captured graphs yet).
//!
//! What the runtime supplies around the graph's steps:
//! - checkpoint tensors from a loaded rank cache (`Buf::Weight`, the routed + shared experts fused at load, see
//!   [`fuse_shared_experts`]);
//! - one workspace buffer per `Buf::Ws` name, sized for the largest chunk at the largest position; `name[i]` is the i-th
//!   outer slice of `name`, `n.last_row` the last row of `n`, and a shorter view of a name starts at its first byte;
//! - engine constants (`Buf::Const`) and the host inputs (`Buf::Host`: token ids, Engram records fetched from the tier);
//! - the stage extras: collectives through the group's [`Comm`], constant fills;
//! - kernel scratch (one buffer, grown when a kernel asks for more).
//!
//! Engram (docs/results/2026-10-31-dsv41-engram-tier.md): the rows of both Engram layers are hashed on the host and issued
//! to the tier the moment a pass's token ids are known; the pass waits for layer `l`'s records right before layer `l`
//! (an eager split of the pass at the Engram layers; nothing waits on IO inside a kernel). Every rank does this for its
//! own share of the rows with the same token ids. `touch` runs at the start of every pass and right before the logits are
//! read, so the drive never sleeps between steps.

use super::config::Dsv41Config;
use super::engram::{record_bytes, EngramTier, NgramHasher};
use super::graph::{build_absorb, build_draft, build_draft_dpos, build_pass, build_replay, build_with, PassKind, dspark_idx, dspark_stages, Arg, Buf, Dsv41Graph, Extra, GraphOpts, GraphTp, Replay, ReplayPart, Stage, WeightShapes, VERIFY_MAX_ROWS};
use super::replay;
use crate::backend::{Backend, Buffer, GraphHandle, MemKind, TensorRef};
use crate::comm::Comm;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{OpCall, ShapeInfo};
use crate::registry::Registry;
use crate::state::{SequenceState, SlotSpec, StateSlot};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Engram inputs of the runtime: the tier holding this rank's rows, the hasher, and the raw -> compressed token map.
pub struct EngramInputs {
    pub tier: Arc<dyn EngramTier>,
    pub hasher: NgramHasher,
    pub token_map: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct RuntimeOpts {
    pub act: DType,
    /// Token capacity of a sequence (shared compressed-KV / index-key slots).
    pub cap: usize,
    /// Most rows of one pass (prefill chunk).
    pub chunk: usize,
    pub tp: GraphTp,
    /// Speculative decoding with the DSpark stages: every pass absorbs its taps into the stage rings. Both ranks must
    /// agree (it is part of the handshake fingerprint).
    pub spec: bool,
}

/// Memory the runtime holds besides the weights, for the startup report.
#[derive(Clone, Debug, Default)]
pub struct RuntimeMemory {
    pub weights: u64,
    pub workspace: u64,
    pub constants: u64,
    pub scratch: u64,
    /// One sequence's state slots at `cap`.
    pub state_per_seq: u64,
}

struct Exec {
    ws: HashMap<String, TensorRef>,
    scratch: Option<TensorRef>,
    /// Last fill written per workspace name: `(offset, count, value bits)`; a step that writes the name clears it.
    filled: HashMap<String, (usize, usize, u64)>,
    /// Kernel choice per (stage, step name, op, rows class: 1 / 2..=16 / more, plus 4 for a device-position step). Step
    /// lists differ between a one-row pass and a longer one (the MoE route tables), so the index alone does not name a step.
    kernels: HashMap<(usize, String, crate::ops::Op, u8), usize>,
    /// Bumped whenever the kernel scratch is reallocated (a captured graph holds the old address).
    scratch_gen: u64,
    /// Recording a CUDA graph: nothing may upload, allocate or synchronise.
    capturing: bool,
    /// `INTARSIA_DSV41_L2_PREFETCH`, during a capture: weights to prefetch after step `(stage, step)` ([`Self::prefetch_plan`]).
    pf_plan: HashMap<(usize, usize), Vec<TensorRef>>,
    /// A prefetch launch since the segment's start (the prefetch stream has to rejoin before the capture ends).
    pf_out: bool,
    /// The graph being run or recorded is a verify pass ([`Dsv41Graph::verify`]).
    verify: bool,
}

/// A captured device-position pass (`INTARSIA_DSV41_GRAPHS=1`): the graph and its recorded segments, split before every
/// Engram layer (the host waits for that layer's records and uploads them between two replays).
struct Captured {
    g: Dsv41Graph,
    /// `(first stage, end stage, recording)`, in order.
    segs: Vec<(usize, usize, GraphHandle)>,
    /// The sequence state whose slot buffers the recording names (`SequenceState::id`).
    state_id: u64,
    /// `Exec::scratch_gen` at capture.
    scratch_gen: u64,
    /// Workspace ranges the recording's steps write, per buffer: `(base name, first byte, end byte)`. A replay runs no
    /// `run_stage`, so it undoes the fills these cover here ([`Exec::filled`] would otherwise keep claiming them, and
    /// the next graph that needs such a fill would skip it: the DSpark block's route table overwrites the cell that
    /// holds the main model's shared-expert id).
    writes: Vec<(String, usize, usize)>,
}

/// Recorded graph kinds: a plain pass (decode), a verify pass (logits of every row), a DSpark drafting block.
/// The first stream of the lanes ([`Dsv41Runtime::lanes`]; stream 1 is the L2 prefetch's) and the alignment of a lane's scratch slice.
const LANE_STREAM0: u32 = 2;
const LANE_STRIDE_ALIGN: usize = 256;
const GK_PASS: u8 = 0;
const GK_VERIFY: u8 = 1;
const GK_DRAFT: u8 = 2;

/// Graph mode of the runtime: a device-position pass seen for the first time runs eagerly (it also warms every kernel up),
/// the next one of the same shape is recorded and replayed.
#[derive(Default)]
struct Graphs {
    /// `(rows, kind)` -> recording (kind: [`GK_PASS`], [`GK_VERIFY`], [`GK_DRAFT`])
    captured: HashMap<(usize, u8), Captured>,
    /// shapes that ran eagerly once
    warm: std::collections::HashSet<(usize, u8)>,
    /// replays since load, per kind ([`GK_PASS`], [`GK_VERIFY`], [`GK_DRAFT`])
    replays: [u64; 3],
}

/// Per-pass timing (decode profiling): host time spent waiting for Engram records.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassStats {
    pub engram_wait_us: u64,
    /// Host time to build the graph of the pass.
    pub build_us: u64,
    /// Host time from the first launch to the last (enqueue only, the device may still run).
    pub enqueue_us: u64,
    /// Whole pass including the logits read (a device sync).
    pub total_us: u64,
}

/// One DSpark drafting round ([`Dsv41Runtime::draft`]): `ids[0]` the round's first id, `ids[1..]` the drafts;
/// `confidence[i]` the confidence head's score of draft `i` (`DSparkConfidenceHead`, a logit).
#[derive(Clone, Debug, PartialEq)]
pub struct Draft {
    pub ids: Vec<u32>,
    pub confidence: Vec<f32>,
}

pub struct Dsv41Runtime {
    /// DSpark stages loaded (0: no drafting).
    pub dspark: usize,
    /// The confidence head's weight row (`[dim + markov rank]`, f32), when the stages are loaded.
    confidence: Vec<f32>,
    pub cfg: Dsv41Config,
    shapes: Box<dyn WeightShapes + Send + Sync>,
    pub backend: Arc<dyn Backend>,
    pub comm: Option<Arc<dyn Comm>>,
    reg: Registry,
    /// The backend's kernel table (taken once: `Backend::kernels` builds it on every call).
    kernel_list: Vec<crate::registry::KernelImpl>,
    weights: HashMap<String, TensorRef>,
    consts: HashMap<String, TensorRef>,
    engram: Option<EngramInputs>,
    pub opts: RuntimeOpts,
    exec: Mutex<Exec>,
    slot_specs: Vec<SlotSpec>,
    pub memory: RuntimeMemory,
    pub last_pass: Mutex<PassStats>,
    /// `INTARSIA_DSV41_PROFILE` accumulator: (passes, build, enqueue, total, engram wait) in us.
    prof: Mutex<(u64, u64, u64, u64, u64)>,
    /// `INTARSIA_DSV41_OPPROF`: (one-row pass?, kernel) -> (calls, us).
    opprof: Option<Mutex<HashMap<(bool, String), (u64, u64)>>>,
    /// `INTARSIA_DSV41_EXL3_SPLITK` (default 16, 0 = off): one-row EXL3 linears whose GEMV grid (N / 128
    /// blocks) is below this run the split-K GEMV `matmul_exl3_splitk` instead (selected here by name; the plain registry
    /// selection never picks it).
    splitk_grid: usize,
    /// `INTARSIA_DSV41_EXL3_SPLITK_VERIFY` (default on, `0` = GEMV): verify passes take split-K on the same linears as the
    /// one-row decode pass, so a verify row is bitwise the decode row. Plain decode and prefill are untouched.
    splitk_verify: bool,
    /// `INTARSIA_DSV41_L2_PREFETCH` (default on since 2026-11-02, `0` = off): in a recorded pass, a paced L2 prefetch on stream 1 of the weights the next
    /// linears read: after a layer's `wq_b` its `wo_a` and `wo_b`, after its experts the next layer's `wq_a`, `wkv`, `wq_b`
    /// (16 MiB a launch). Changes no numbers.
    l2_prefetch: bool,
    /// `INTARSIA_DSV41_LANES` (default on, `0` = off): in a recorded pass the steps with a lane ([`super::graph::Step::lane`]: the `wo_a`
    /// slice matmuls, `wq_a` with `wkv`, the compressor's pair) fork onto streams 2..=5 and join before the next lane-0 step, so their small launches
    /// run side by side. Same kernels on the same bytes, a scratch of their own each: changes no numbers.
    lanes: bool,
    /// `INTARSIA_DSV41_EXL3_CP` (comma list of GEMV grids N / 128, default none): on the same one-row and verify calls, these
    /// shapes take `matmul_exl3_cp` (wo_a slice 8, wq_a 10, wo_b 40, wq_b 128, indexer wq_b 32, wkv 4).
    cp_grids: Vec<usize>,
    /// `INTARSIA_DSV41_DPOS=1` (or graphs on): passes of up to `VERIFY_MAX_ROWS` rows run the device-position graph
    /// ([`PassKind::dpos`]); longer chunks keep the host-position graph.
    dpos: std::sync::atomic::AtomicBool,
    /// `INTARSIA_DSV41_GRAPHS` (default on, `0` = eager): one-row decode passes and verify passes replay CUDA graphs of the
    /// device-position graph. Switched off for good when a capture fails.
    graphs_on: std::sync::atomic::AtomicBool,
    /// `INTARSIA_DSV41_ENGRAM_ONEGRAPH` (default on, `0` = off): one recording for the whole pass, all Engram records waited
    /// for and uploaded before it starts. Off: a segment per Engram layer, the wait overlapping the earlier layers; the first
    /// collective of every recording starts late (0.2-0.7 ms in the traces), which costs more than the overlap saves.
    engram_one_graph: bool,
    graphs: Mutex<Graphs>,
    /// Largest f32 partial of a reduction (`Comm::reserve` before the first capture).
    max_partial: usize,
    /// `INTARSIA_DSV41_REPLAY=1` (default off): prefill runs the decoder half on the last `window` rows only
    /// ([`super::replay`]). Both ranks must agree (handshake fingerprint). Changes numerics.
    replay: bool,
}

/// Little-endian `dtype` elements as f32 (f32, bf16, f16).
fn to_f32(b: &[u8], dtype: DType) -> Result<Vec<f32>> {
    Ok(match dtype {
        DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        DType::Bf16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        DType::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        d => return Err(Error::invalid(format!("deepseek_v41: {d} as f32"))),
    })
}

fn esz(d: DType) -> usize {
    d.size_bytes()
}

/// Unix seconds for the capture and scratch log lines, to tie them to `scripts/dsv41-tp2/memfine.sh`'s memory log.
fn unix_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The workspace ranges `g`'s steps write, merged per buffer ([`Captured::writes`]).
fn ws_writes(g: &Dsv41Graph) -> Vec<(String, usize, usize)> {
    let mut m: HashMap<String, (usize, usize)> = HashMap::new();
    for a in g.stages.iter().flat_map(|s| s.steps.iter()).flat_map(|s| s.outputs.iter()) {
        if let Buf::Ws(n) = &a.buf {
            let len = a.shape.iter().product::<usize>() * esz(a.dtype);
            let start = ws_index(n).map_or(0, |i| i * len);
            let e = m.entry(ws_base(n).to_string()).or_insert((start, start + len));
            e.0 = e.0.min(start);
            e.1 = e.1.max(start + len);
        }
    }
    m.into_iter().map(|(k, (a, b))| (k, a, b)).collect()
}

impl Dsv41Runtime {
    /// `weights`: this rank's tensors (already fused, [`fuse_shared_experts`]); `shapes`: their shapes (the rank cache's
    /// headers, with the fused expert shapes), from which the graph is built.
    pub fn new(
        cfg: Dsv41Config,
        shapes: Box<dyn WeightShapes + Send + Sync>,
        backend: Arc<dyn Backend>,
        comm: Option<Arc<dyn Comm>>,
        weights: HashMap<String, TensorRef>,
        engram: Option<EngramInputs>,
        opts: RuntimeOpts,
    ) -> Result<Dsv41Runtime> {
        let world = comm.as_ref().map_or(1, |c| c.world());
        if world != opts.tp.world {
            return Err(Error::invalid(format!("deepseek_v41 runtime: communicator of {world} ranks, graph for {}", opts.tp.world)));
        }
        if cfg.engram.as_ref().is_some_and(|e| !e.layer_ids.is_empty()) && engram.is_none() {
            // a truncated graph that stops before the first Engram layer needs no tier
            let first = cfg.engram.as_ref().unwrap().layer_ids.iter().min().copied().unwrap_or(0);
            if opts.tp.layers.is_none_or(|l| l > first) {
                return Err(Error::invalid("deepseek_v41 runtime: the model has Engram layers but no Engram tier was given"));
            }
        }
        let replay = replay::enabled();
        if replay {
            replay::supported(&cfg, &opts.tp)?;
            // the replay drops window keys by shifting the sparse attention's position (the kernels' window test is
            // `pos0 + row >= window`): read in the CUDA kernel (kernels/cuda/dsv41_attn.cu) and the cpu reference
            // (dsv41_ref.rs) only, and only the CUDA kernel is meant to run it
            if backend.name() != "cuda" {
                return Err(Error::Unsupported(format!("deepseek_v41 {}=1: not implemented for the {} backend", replay::ENV, backend.name())));
            }
        }
        let info = backend.info();
        let reg = Registry::new(backend.name(), &info, backend.kernels());
        let kernel_list = backend.kernels();
        let weight_bytes: u64 = weights.values().map(|t| t.nbytes() as u64).sum();
        // workspace: the largest byte size of every name over the shapes a pass can take
        let mut need: HashMap<String, (usize, DType)> = HashMap::new();
        let v = VERIFY_MAX_ROWS.min(opts.cap - 1);
        let probes = [(opts.chunk, 0), (opts.chunk, opts.cap.saturating_sub(opts.chunk)), (1, 0), (1, opts.cap - 1), (2, opts.cap - 2), (v, 0), (v, opts.cap - v)];
        let env_on = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
        let opprof_on = std::env::var_os("INTARSIA_DSV41_OPPROF").is_some_and(|v| v != "0");
        let mut graphs_on = std::env::var("INTARSIA_DSV41_GRAPHS").map_or(true, |v| v != "0");
        if graphs_on && (opprof_on || std::env::var_os("INTARSIA_DSV41_DUMP").is_some()) {
            eprintln!("dsv41: INTARSIA_DSV41_GRAPHS ignored: per-op profiling / dumps synchronise after every step");
            graphs_on = false;
        }
        if graphs_on && !backend.supports_graphs() {
            eprintln!("dsv41: INTARSIA_DSV41_GRAPHS ignored: backend '{}' has no graph capture", backend.name());
            graphs_on = false;
        }
        if graphs_on && comm.as_ref().is_some_and(|c| !c.capture_safe()) {
            eprintln!("dsv41: INTARSIA_DSV41_GRAPHS ignored: the communicator's collectives are not capturable");
            graphs_on = false;
        }
        let dpos = graphs_on || env_on("INTARSIA_DSV41_DPOS");
        let mut slot_specs = Vec::new();
        let mut graphs = Vec::new();
        for (t, pos0) in probes {
            let all_rows = opts.spec && t == v;
            let g = build_pass(&cfg, &*shapes, GraphOpts { t, pos0, act: opts.act, cap: opts.cap }, opts.tp, PassKind { all_rows, absorb: opts.spec, dpos: false })?;
            slot_specs = g.slots.clone();
            graphs.push(g);
            // the device-position graph of a short pass (sized always, so tests can switch it on: its capacity views are
            // no larger than the host graph's at the largest position)
            if t <= VERIFY_MAX_ROWS {
                match build_pass(&cfg, &*shapes, GraphOpts { t, pos0, act: opts.act, cap: opts.cap }, opts.tp, PassKind { all_rows, absorb: opts.spec, dpos: true }) {
                    Ok(g) => graphs.push(g),
                    Err(e) if dpos => return Err(e),
                    Err(_) => {}
                }
            }
        }
        let dspark = dspark_stages(&cfg, &*shapes);
        if opts.spec && dspark == 0 {
            return Err(Error::Unsupported("deepseek_v41: speculative decoding asked for, but the rank cache has no DSpark (mtp) stages".into()));
        }
        if opts.spec {
            let blk = cfg.mtp.as_ref().map_or(0, |m| m.block_size);
            for pos0 in [1, opts.cap.saturating_sub(blk).max(1)] {
                graphs.push(build_draft(&cfg, &*shapes, GraphOpts { t: blk, pos0, act: opts.act, cap: opts.cap }, opts.tp, blk)?);
            }
            graphs.push(build_absorb(&cfg, &*shapes, GraphOpts { t: opts.chunk, pos0: 0, act: opts.act, cap: opts.cap }, opts.tp)?);
        }
        for g in &graphs {
            for st in &g.stages {
                let mut args: Vec<&Arg> = st.steps.iter().flat_map(|s| s.inputs.iter().chain(s.outputs.iter())).collect();
                for (_, e) in &st.extras {
                    match e {
                        Extra::AllReduce { x } => args.push(x),
                        Extra::AllReduceInto { partial, out } => args.extend([partial, out]),
                        Extra::AllGather { send, recv } => args.extend([send, recv]),
                        Extra::Fill { .. } => {}
                    }
                }
                for a in args {
                    let name = match &a.buf {
                        Buf::Ws(n) => ws_base(n).to_string(),
                        Buf::Host(n) => format!("host.{n}"),
                        _ => continue,
                    };
                    let mut bytes = a.shape.iter().product::<usize>() * esz(a.dtype);
                    if let Buf::Ws(n) = &a.buf {
                        if let Some(i) = ws_index(n) {
                            bytes *= i + 1;
                        }
                    }
                    let e = need.entry(name).or_insert((0, a.dtype));
                    e.0 = e.0.max(bytes);
                }
                // fills address elements past a view's end
                for (_, e) in &st.extras {
                    if let Extra::Fill { buf, dtype, offset, count, .. } = e {
                        let en = need.entry(buf.clone()).or_insert((0, *dtype));
                        en.0 = en.0.max((offset + count) * esz(*dtype));
                    }
                }
            }
        }
        // the device position of the pass (`Step::pos`)
        need.insert("host.pos".into(), (4, DType::I32));
        // a captured collective must never allocate: reserve the largest f32 partial now
        let max_partial = graphs
            .iter()
            .flat_map(|g| g.stages.iter().flat_map(|s| s.extras.iter()))
            .filter_map(|(_, e)| match e {
                Extra::AllReduceInto { partial, .. } => Some(partial.shape.iter().product::<usize>()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if let (true, Some(c)) = (graphs_on, comm.as_ref()) {
            c.reserve(max_partial, 0)?;
        }
        let mut ws = HashMap::new();
        let mut ws_bytes = 0u64;
        for (name, (bytes, dt)) in need {
            let b = backend.alloc(bytes.max(256), MemKind::Device)?;
            backend.zero(&b, 0)?;
            ws_bytes += b.bytes() as u64;
            ws.insert(name, TensorRef::new(b, dt, &[bytes / esz(dt)]));
        }
        // constants
        let s = cfg.hc_mult;
        let mut consts = HashMap::new();
        let mut cbytes = 0u64;
        let mut put = |name: &str, dt: DType, shape: &[usize], data: &[u8]| -> Result<()> {
            let b = backend.alloc(data.len().max(256), MemKind::Device)?;
            backend.zero(&b, 0)?;
            if !data.is_empty() {
                backend.upload(&b, 0, data, 0)?;
            }
            cbytes += b.bytes() as u64;
            consts.insert(name.to_string(), TensorRef::new(b, dt, shape));
            Ok(())
        };
        let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let base_layer = (0..cfg.n_layers).find(|&l| cfg.ratio(l) == 0).unwrap_or(0);
        let cmp_layer = (0..cfg.n_layers).find(|&l| cfg.ratio(l) > 0).unwrap_or(0);
        put("rope.base", DType::F32, &[cfg.rope_head_dim / 2], &f32b(&cfg.inv_freq(base_layer)))?;
        put("rope.compressed_yarn", DType::F32, &[cfg.rope_head_dim / 2], &f32b(&cfg.inv_freq(cmp_layer)))?;
        let rows = opts.chunk.max(2);
        let mut one_hot = vec![0f32; rows * s];
        (0..rows).for_each(|r| one_hot[r * s] = 1.0);
        put("hc.one_hot_pre_mix", DType::F32, &[rows, s], &f32b(&one_hot))?;
        put("hc.mean_mix", DType::F32, &[rows, s], &f32b(&vec![1.0 / s as f32; rows * s]))?;
        put("swiglu_limit", DType::F32, &[1], &f32b(&[cfg.swiglu_limit]))?;
        for n in ["empty", "empty_i32", "empty_u8", "empty_out"] {
            put(n, DType::U8, &[0], &[])?;
        }
        // the DSpark confidence head on the host (`DSparkConfidenceHead`: f32 dot of [hidden, markov embed])
        let confidence = match dspark {
            0 => Vec::new(),
            n_st => {
                let name = format!("mtp.{}.confidence_head.proj.weight", n_st - 1);
                let t = weights.get(&name).ok_or_else(|| Error::model(format!("deepseek_v41: {name} not loaded")))?;
                let mut b = vec![0u8; t.nbytes()];
                backend.download(&t.buf, t.offset, &mut b, 0)?;
                to_f32(&b, t.dtype)?
            }
        };
        let l2_prefetch = std::env::var("INTARSIA_DSV41_L2_PREFETCH").map_or(true, |v| v != "0") && backend.stream_fork();
        let lanes = std::env::var("INTARSIA_DSV41_LANES").map_or(true, |v| v != "0") && backend.stream_fork();
        let mut rt = Dsv41Runtime {
            cfg,
            shapes,
            backend,
            comm,
            reg,
            kernel_list,
            weights,
            consts,
            engram,
            opts,
            exec: Mutex::new(Exec { ws, pf_plan: HashMap::new(), pf_out: false, scratch: None, filled: HashMap::new(), kernels: HashMap::new(), scratch_gen: 0, capturing: false, verify: false }),
            slot_specs,
            memory: RuntimeMemory::default(),
            last_pass: Mutex::new(PassStats::default()),
            prof: Mutex::new((0, 0, 0, 0, 0)),
            opprof: std::env::var_os("INTARSIA_DSV41_OPPROF").filter(|v| v != "0").map(|_| Mutex::new(HashMap::new())),
            splitk_grid: std::env::var("INTARSIA_DSV41_EXL3_SPLITK").ok().and_then(|v| v.parse().ok()).unwrap_or(16),
            splitk_verify: std::env::var("INTARSIA_DSV41_EXL3_SPLITK_VERIFY").map_or(true, |v| v != "0"),
            l2_prefetch,
            lanes,
            cp_grids: std::env::var("INTARSIA_DSV41_EXL3_CP").map_or(Vec::new(), |v| v.split(',').filter_map(|x| x.parse().ok()).collect()),
            dspark,
            confidence,
            dpos: std::sync::atomic::AtomicBool::new(dpos),
            graphs_on: std::sync::atomic::AtomicBool::new(graphs_on),
            engram_one_graph: std::env::var("INTARSIA_DSV41_ENGRAM_ONEGRAPH").map_or(true, |v| v != "0"),
            graphs: Mutex::new(Graphs::default()),
            max_partial,
            replay,
        };
        let state_per_seq = rt.slot_specs.iter().map(|s| s.bytes(rt.opts.cap) as u64).sum();
        rt.memory = RuntimeMemory { weights: weight_bytes, workspace: ws_bytes, constants: cbytes, scratch: 0, state_per_seq };
        Ok(rt)
    }

    /// Keep the pass's timing; with `INTARSIA_DSV41_PROFILE=N` print the mean of every N one-row passes (host build,
    /// host enqueue, whole pass with the logits sync, Engram wait): enqueue close to the whole pass means host-bound.
    fn note_pass(&self, t: usize, s: PassStats) {
        *self.last_pass.lock().unwrap() = s;
        let every = match std::env::var("INTARSIA_DSV41_PROFILE").ok().and_then(|v| v.parse::<u64>().ok()) {
            Some(n) if n > 0 && t == 1 => n,
            Some(n) if n > 0 && t <= 64 => {
                // a short multi-row pass (a small prompt; the cost of a verify-sized pass), one line each
                eprintln!(
                    "dsv41 profile pass rows={t}: build {:.2} ms, enqueue {:.2} ms, whole pass {:.2} ms, engram wait {:.3} ms",
                    s.build_us as f64 / 1e3,
                    s.enqueue_us as f64 / 1e3,
                    s.total_us as f64 / 1e3,
                    s.engram_wait_us as f64 / 1e3
                );
                return;
            }
            _ => return,
        };
        let mut acc = self.prof.lock().unwrap();
        acc.0 += 1;
        acc.1 += s.build_us;
        acc.2 += s.enqueue_us;
        acc.3 += s.total_us;
        acc.4 += s.engram_wait_us;
        if acc.0 >= every {
            let n = acc.0 as f64;
            eprintln!(
                "dsv41 profile: {} decode passes: build {:.2} ms, enqueue {:.2} ms, whole pass {:.2} ms, engram wait {:.3} ms",
                acc.0,
                acc.1 as f64 / n / 1e3,
                acc.2 as f64 / n / 1e3,
                acc.3 as f64 / n / 1e3,
                acc.4 as f64 / n / 1e3
            );
            *acc = (0, 0, 0, 0, 0);
            drop(acc);
            self.print_opprof();
        }
    }

    /// Device-position passes (`INTARSIA_DSV41_DPOS`) on or off from now on (tests: the same runtime both ways). Off also
    /// turns graphs off; the workspace must have been sized for them (`INTARSIA_DSV41_DPOS=1` at load).
    pub fn set_device_position(&self, on: bool) {
        self.dpos.store(on, std::sync::atomic::Ordering::Relaxed);
        if !on {
            self.graphs_on.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// CUDA graph replay of decode / verify passes on or off from now on (tests); on implies device positions. `false` when
    /// the backend cannot capture.
    pub fn set_graphs(&self, on: bool) -> bool {
        let on = on && self.backend.supports_graphs() && self.comm.as_ref().is_none_or(|c| c.capture_safe());
        if let (true, Some(c)) = (on, self.comm.as_ref()) {
            if c.reserve(self.max_partial, 0).is_err() {
                return false;
            }
        }
        if on {
            self.dpos.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.graphs_on.store(on, std::sync::atomic::Ordering::Relaxed);
        on
    }

    /// Graphs replayed since load: (decode passes, verify passes, drafting blocks).
    pub fn graph_replays(&self) -> (u64, u64, u64) {
        let r = self.graphs.lock().unwrap().replays;
        (r[0], r[1], r[2])
    }

    pub fn slot_specs(&self) -> &[SlotSpec] {
        &self.slot_specs
    }

    pub fn world(&self) -> usize {
        self.opts.tp.world
    }

    /// A fresh sequence (state slots zeroed).
    pub fn new_state(&self) -> Result<SequenceState> {
        let st = SequenceState::new(&*self.backend, &self.slot_specs, self.opts.cap)?;
        for s in &st.slots {
            match s {
                StateSlot::Ring(r) => self.backend.zero(&r.buf, 0)?,
                StateSlot::Shared(r) => self.backend.zero(&r.buf, 0)?,
                _ => {}
            }
        }
        Ok(st)
    }

    /// Touch the Engram storage (keep the drive awake).
    pub fn touch(&self) {
        if let Some(e) = &self.engram {
            e.tier.touch();
        }
    }

    /// Run `ids` through the model on `st` in chunks of at most `opts.chunk` rows, appending them to the sequence.
    /// With `logits`, returns the full-vocabulary f32 logits of the last row (every rank gets them).
    pub fn forward(&self, st: &mut SequenceState, ids: &[u32], logits: bool) -> Result<Option<Vec<f32>>> {
        if ids.is_empty() {
            return Err(Error::invalid("deepseek_v41: empty forward pass"));
        }
        if st.len + ids.len() > self.opts.cap {
            return Err(Error::BadRequest(format!("deepseek_v41: {} + {} tokens exceed the capacity {}", st.len, ids.len(), self.opts.cap)));
        }
        // bounded replay (opt-in): the prompt's replay floor, for a prefill of more than one row; None: the plain passes below
        let floor = self.replay_floor(st, ids.len())?;
        let start = st.len;
        let mut out = None;
        let mut i = 0;
        while i < ids.len() {
            let n = (ids.len() - i).min(self.opts.chunk);
            let last = i + n == ids.len();
            let want = if logits && last { Want::Logits } else { Want::Nothing };
            let r = match floor {
                Some(f) if f > 0 => self.pass_replay(st, &ids[i..i + n], want, f),
                _ => self.pass(st, &ids[i..i + n], want, false),
            };
            match r {
                Ok(l) => {
                    st.commit(&ids[i..i + n]);
                    if last {
                        out = l.map(Out::logits);
                    }
                }
                Err(e) => {
                    // the rows this pass wrote are undefined: the sequence restarts
                    let _ = st.truncate(0, &*self.backend);
                    st.replay_from = 0;
                    return Err(e);
                }
            }
            i += n;
        }
        if let Some(f) = floor {
            // bookkeeping check of the replay passes: they never commit (`forward` does, once per chunk), so the sequence has
            // advanced by exactly the rows given; a mismatch fails here and not as wrong numbers
            if st.len != start + ids.len() || st.tokens.len() != st.len {
                let _ = st.truncate(0, &*self.backend);
                st.replay_from = 0;
                return Err(Error::backend(format!("deepseek_v41 replay: sequence length {} / {} tokens after {} rows from {start}", st.len, st.tokens.len(), ids.len())));
            }
            st.replay_from = f;
        }
        Ok(out)
    }

    /// The replay floor of the prefill `ids` appended to `st` (bounded replay, [`super::replay`]): `Some(floor)` when the option
    /// is on and the forward has more than one row (a one-row forward is a decode step: its window never reaches below the
    /// floor, and it runs the plain pass). A floor of 0 (a prompt of at most `window` tokens: nothing to cut) also runs the
    /// plain passes, bit for bit. Errors when `st` was built by a replay whose floor lies above this one's and the rows between are
    /// kept: their decoder-layer window keys were never written ([`replay::reuse_ok`]; the engine prefills such a request from
    /// scratch, [`Self::replay_keep`]).
    fn replay_floor(&self, st: &SequenceState, rows: usize) -> Result<Option<usize>> {
        if !self.replay || rows < 2 {
            return Ok(None);
        }
        let f = replay::floor(st.len + rows, self.cfg.window);
        if !replay::reuse_ok(st.len, f, st.replay_from) {
            return Err(Error::Unsupported(format!(
                "deepseek_v41 {}=1: resuming {} tokens whose decoder-layer window keys are valid from {} only, under a floor of {f}",
                replay::ENV,
                st.len,
                st.replay_from
            )));
        }
        Ok(Some(f))
    }

    /// How many of the `keep` cached tokens a request of `prompt_len` tokens may resume from (bounded replay: `keep`, or 0
    /// when the cached rows' decoder-layer window keys cannot serve the rows after them, [`replay::reuse_ok`]). Without the
    /// option: `keep`.
    pub fn replay_keep(&self, st: &SequenceState, keep: usize, prompt_len: usize) -> usize {
        if !self.replay || keep == 0 || replay::reuse_ok(keep, replay::floor(prompt_len, self.cfg.window), st.replay_from) {
            keep
        } else {
            0
        }
    }

    /// One prefill chunk of a bounded-replay prompt whose decoder half starts at position `floor` ([`super::replay`]): an
    /// encoder-only chunk, a chunk the floor falls in (the encoder part on all rows, then the decoder half on the rows from the
    /// floor on, in the workspace the encoder part left them in) or a chunk of rows all at or past the floor. Both ranks derive
    /// the same plan from the same `(floor, st.len, ids.len())` and so run the same graphs with the same collectives.
    fn pass_replay(&self, st: &mut SequenceState, ids: &[u32], want: Want, floor: usize) -> Result<Option<Out>> {
        let (t, pos0) = (ids.len(), st.len);
        let p0 = Instant::now();
        let cut = replay::cut_layer(self.cfg.n_layers);
        let k = PassKind { all_rows: false, absorb: self.opts.spec, dpos: false };
        let o = |t: usize, pos0: usize| GraphOpts { t, pos0, act: self.opts.act, cap: self.opts.cap };
        let build = |t: usize, pos0: usize, part: ReplayPart, first: usize| {
            build_replay(&self.cfg, &*self.shapes, o(t, pos0), self.opts.tp, k, Replay { part, cut, floor, first })
        };
        match replay::classify(floor, pos0, t) {
            replay::Chunk::Whole => {
                let g = build(t, pos0, ReplayPart::Whole, 0)?;
                self.run_graph(&g, st, t, ids, &[], want, p0)
            }
            replay::Chunk::Encoder => {
                // the chunk with the prompt's last row is never encoder-only (floor < total), so nothing is asked of the head
                if want != Want::Nothing {
                    return Err(Error::invalid("deepseek_v41 replay: an encoder-only chunk has no logits"));
                }
                let g = build(t, pos0, ReplayPart::Encoder, 0)?;
                self.run_graph(&g, st, t, ids, &[], Want::Nothing, p0)
            }
            replay::Chunk::Cross { first } => {
                let e = build(t, pos0, ReplayPart::Encoder, 0)?;
                self.run_graph(&e, st, t, ids, &[], Want::Nothing, p0)?;
                let d = build(t - first, pos0 + first, ReplayPart::Decoder, first)?;
                // no token ids: the decoder part starts from the encoder part's streams (and has no Engram layer)
                self.run_graph(&d, st, t - first, &[], &[], want, p0)
            }
        }
    }

    /// One verify-sized pass (`2..=VERIFY_MAX_ROWS` rows, or 1) appended to the sequence: the full-vocabulary f32 logits
    /// of every row (`rows[j]` predicts the token after `ids[j]`).
    pub fn forward_rows(&self, st: &mut SequenceState, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        if ids.is_empty() || ids.len() > VERIFY_MAX_ROWS {
            return Err(Error::invalid(format!("deepseek_v41: a verify pass takes 1..={VERIFY_MAX_ROWS} rows, not {}", ids.len())));
        }
        if st.len + ids.len() > self.opts.cap {
            return Err(Error::BadRequest(format!("deepseek_v41: {} + {} tokens exceed the capacity {}", st.len, ids.len(), self.opts.cap)));
        }
        if !self.opts.spec {
            return Err(Error::Unsupported("deepseek_v41: a verify pass needs the runtime's spec option".into()));
        }
        match self.pass(st, ids, Want::Logits, true) {
            Ok(l) => {
                st.commit(ids);
                let l = l.expect("logits").logits();
                let v = l.len() / ids.len();
                Ok(l.chunks_exact(v).map(|r| r.to_vec()).collect())
            }
            Err(e) => {
                let _ = st.truncate(0, &*self.backend);
                Err(e)
            }
        }
    }

    /// [`Self::forward_rows`] with the argmax of every row computed on the device (first index of the strict maximum,
    /// as `sampler::argmax`): only `ids.len()` ints come back, not `ids.len() * vocab` floats.
    pub fn forward_rows_ids(&self, st: &mut SequenceState, ids: &[u32]) -> Result<Vec<u32>> {
        if ids.is_empty() || ids.len() > VERIFY_MAX_ROWS {
            return Err(Error::invalid(format!("deepseek_v41: a verify pass takes 1..={VERIFY_MAX_ROWS} rows, not {}", ids.len())));
        }
        if st.len + ids.len() > self.opts.cap {
            return Err(Error::BadRequest(format!("deepseek_v41: {} + {} tokens exceed the capacity {}", st.len, ids.len(), self.opts.cap)));
        }
        if !self.opts.spec {
            return Err(Error::Unsupported("deepseek_v41: a verify pass needs the runtime's spec option".into()));
        }
        match self.pass(st, ids, Want::Ids, true) {
            Ok(r) => {
                st.commit(ids);
                Ok(r.expect("ids").ids())
            }
            Err(e) => {
                let _ = st.truncate(0, &*self.backend);
                Err(e)
            }
        }
    }

    /// One DSpark drafting round after the sequence's last row (position `st.len - 1`, whose taps the pass that ran it
    /// absorbed): the block `[first, noise, ..]` through the stages and the head on every rank (the head's gather), then
    /// on rank 0 the Markov loop and the confidence head (`DSparkBlock.forward_head`, greedy). Rank 0 gets
    /// `Some(draft)`, other ranks `None`. The sequence is unchanged.
    pub fn draft(&self, st: &SequenceState, first: u32) -> Result<Option<Draft>> {
        self.draft_steps(st, first, usize::MAX)
    }

    /// [`Self::draft`] with only the first `steps` Markov steps (clamped to `1..=block_size`): `Draft::ids` has `steps + 1`
    /// entries (`first` and the chosen drafts) and `confidence` `steps` (each step is bitwise the full loop's).
    pub fn draft_steps(&self, st: &SequenceState, first: u32, steps: usize) -> Result<Option<Draft>> {
        let m = self.cfg.mtp.as_ref().filter(|_| self.dspark > 0 && self.opts.spec).ok_or_else(|| Error::Unsupported("deepseek_v41: drafting needs the DSpark stages and the runtime's spec option".into()))?;
        let (t, pos0) = (m.block_size, st.len);
        if pos0 == 0 || pos0 + t > self.opts.cap {
            return Err(Error::invalid(format!("deepseek_v41: a draft block at position {pos0} (capacity {})", self.opts.cap)));
        }
        let p0 = Instant::now();
        let mut ids = vec![m.noise_token_id; t];
        ids[0] = first;
        let idx: Vec<u8> = dspark_idx(self.cfg.window, t, pos0).iter().flat_map(|v| v.to_le_bytes()).collect();
        let rank0 = self.comm.as_ref().is_none_or(|c| c.rank() == 0);
        // device positions (eager or replayed) from position `window` on, as the decode / verify passes
        let dpos = pos0 >= self.cfg.window && self.dpos.load(std::sync::atomic::Ordering::Relaxed);
        let o = GraphOpts { t, pos0, act: self.opts.act, cap: self.opts.cap };
        // the Markov steps run in the drafting graph, on the rank that reads the drafts only (replicated weights)
        let steps = if rank0 { steps.clamp(1, t) } else { 0 };
        let build = || if dpos { build_draft_dpos(&self.cfg, &*self.shapes, o, self.opts.tp, steps) } else { build_draft(&self.cfg, &*self.shapes, o, self.opts.tp, steps) };
        let want = if rank0 { Want::Ids } else { Want::Nothing };
        let mut replayed = None;
        if dpos && self.graphs_on.load(std::sync::atomic::Ordering::Relaxed) {
            replayed = self.run_captured(st, &ids, (t | steps << 16, GK_DRAFT), &[("host.dspark.idx", &idx)], want, p0, &build)?;
        }
        let picked = match replayed {
            Some(l) => l,
            None => self.run_graph(&build()?, st, t, &ids, &[("host.dspark.idx", &idx)], want, p0)?,
        };
        if !rank0 {
            return Ok(None);
        }
        let mut out = vec![first];
        out.extend(picked.expect("markov ids").ids());
        // the embeddings the Markov steps fed in (`dspark.markov_e[i]` for the id `out[i]`)
        let rank = self.confidence.len() - self.cfg.dim;
        let embeds: Vec<Vec<f32>> = {
            let ex = self.exec.lock().unwrap();
            let e = ex.ws.get("dspark.markov_e").ok_or_else(|| Error::backend("no workspace dspark.markov_e"))?.clone();
            let mut eb = vec![0u8; steps * rank * esz(self.opts.act)];
            self.backend.download(&e.buf, e.offset, &mut eb, 0)?;
            let f = to_f32(&eb, self.opts.act)?;
            f.chunks_exact(rank).map(|c| c.to_vec()).collect()
        };
        // confidence = proj(cat(hx, markov embed)) per row, in f32
        let d = self.cfg.dim;
        let hx = {
            let ex = self.exec.lock().unwrap();
            let b = ex.ws.get("dspark.hx").ok_or_else(|| Error::backend("no workspace dspark.hx"))?.clone();
            let mut hb = vec![0u8; steps * d * esz(self.opts.act)];
            self.backend.download(&b.buf, b.offset, &mut hb, 0)?;
            to_f32(&hb, self.opts.act)?
        };
        let w = &self.confidence;
        let confidence = (0..steps)
            .map(|r| {
                let a: f32 = hx[r * d..(r + 1) * d].iter().zip(&w[..d]).map(|(x, y)| x * y).sum();
                a + embeds[r].iter().zip(&w[d..]).map(|(x, y)| x * y).sum::<f32>()
            })
            .collect();
        Ok(Some(Draft { ids: out, confidence }))
    }

    /// Feed the DSpark stage rings from `main_hidden` (`[t, taps * dim]` f32, the concatenated taps) at positions
    /// `st.len ..` and append `t` placeholder rows to the sequence (tests and diagnosis: the model's passes absorb their
    /// own taps). Every rank runs it.
    pub fn absorb(&self, st: &mut SequenceState, main_hidden: &[f32]) -> Result<()> {
        let n_taps = self.cfg.mtp.as_ref().map_or(0, |m| m.target_layer_ids.len());
        let w = n_taps * self.cfg.dim;
        if self.dspark == 0 || w == 0 || main_hidden.is_empty() || main_hidden.len() % w != 0 {
            return Err(Error::invalid("deepseek_v41: absorb needs the DSpark stages and [t, taps * dim] rows"));
        }
        let t = main_hidden.len() / w;
        if t > self.opts.chunk || st.len + t > self.opts.cap {
            return Err(Error::invalid(format!("deepseek_v41: absorb of {t} rows at {}", st.len)));
        }
        let g = build_absorb(&self.cfg, &*self.shapes, GraphOpts { t, pos0: st.len, act: self.opts.act, cap: self.opts.cap }, self.opts.tp)?;
        let b: Vec<u8> = match self.opts.act {
            DType::F32 => main_hidden.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DType::Bf16 => main_hidden.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect(),
            d => return Err(Error::invalid(format!("deepseek_v41: absorb in {d}"))),
        };
        self.run_graph(&g, st, t, &[], &[("host.dspark.mh", &b)], Want::Nothing, Instant::now())?;
        st.commit(&vec![0; t]);
        Ok(())
    }

    /// One pass of `t = ids.len()` rows at position `st.len`.
    fn pass(&self, st: &mut SequenceState, ids: &[u32], want: Want, all_rows: bool) -> Result<Option<Out>> {
        let (t, pos0) = (ids.len(), st.len);
        let p0 = Instant::now();
        let dpos = self.dpos.load(std::sync::atomic::Ordering::Relaxed) && t <= VERIFY_MAX_ROWS;
        let k = PassKind { all_rows, absorb: self.opts.spec, dpos };
        if dpos && (t == 1 || all_rows) && self.graphs_on.load(std::sync::atomic::Ordering::Relaxed) {
            let key = (t, if all_rows { GK_VERIFY } else { GK_PASS });
            // the graph does not depend on the host position: built at 0, it serves every position
            let build = || build_pass(&self.cfg, &*self.shapes, GraphOpts { t, pos0: 0, act: self.opts.act, cap: self.opts.cap }, self.opts.tp, k);
            if let Some(r) = self.run_captured(st, ids, key, &[], want, p0, &build)? {
                return Ok(r);
            }
        }
        let g = build_pass(&self.cfg, &*self.shapes, GraphOpts { t, pos0, act: self.opts.act, cap: self.opts.cap }, self.opts.tp, k)?;
        self.run_graph(&g, st, t, ids, &[], want, p0)
    }

    /// Host inputs of a pass: token ids, the extras, the device position (when a step reads it).
    fn upload_inputs(&self, ex: &Exec, g: &Dsv41Graph, pos0: usize, ids: &[u32], host: &[(&str, &[u8])]) -> Result<()> {
        if !ids.is_empty() {
            let idb: Vec<u8> = ids.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
            let tid = ex.ws.get("host.token_ids").ok_or_else(|| Error::backend("no token id buffer"))?.clone();
            self.backend.upload(&tid.buf, 0, &idb, 0)?;
        }
        for (name, data) in host {
            let b = ex.ws.get(*name).ok_or_else(|| Error::backend(format!("no host buffer {name}")))?.clone();
            self.backend.upload(&b.buf, 0, data, 0)?;
        }
        if g.stages.iter().any(|s| s.steps.iter().any(|x| x.pos)) {
            let b = ex.ws.get("host.pos").ok_or_else(|| Error::backend("no position buffer"))?.clone();
            self.backend.upload(&b.buf, b.offset, &(pos0 as i32).to_le_bytes(), 0)?;
        }
        Ok(())
    }

    /// Engram: hash and issue every Engram layer's rows of the pass now; layer -> (index, ticket).
    fn issue_engram(&self, g: &Dsv41Graph, st: &SequenceState, ids: &[u32], t: usize) -> Result<HashMap<usize, (usize, super::engram::EngramTicket)>> {
        let pos0 = st.len;
        let engram_layers: Vec<usize> = g.stages.iter().filter_map(|s| s.layer).collect();
        let mut tickets = HashMap::new();
        if let Some(e) = self.engram.as_ref().filter(|_| !engram_layers.is_empty() && !ids.is_empty()) {
            let lim = self.opts.tp.layers.unwrap_or(usize::MAX);
            if e.hasher.layout.layer_ids.iter().any(|&l| l < lim) {
                let lo = pos0.saturating_sub(e.hasher.layout.max_ngram - 1);
                let map = |raw: u32| e.token_map.get(raw as usize).map_or(0, |&c| c as i64);
                let mut hist: Vec<i64> = st.tokens[lo..pos0].iter().map(|&x| map(x)).collect();
                hist.extend(ids.iter().map(|&x| map(x)));
                let rows = e.hasher.hash(&hist, pos0 - lo, t)?;
                let (nl, cols) = (e.hasher.layout.layer_ids.len(), e.hasher.layout.cols());
                for (li, &layer) in e.hasher.layout.layer_ids.iter().enumerate() {
                    if layer >= lim {
                        continue;
                    }
                    let mine: Vec<u32> = (0..t).flat_map(|p| rows[(p * nl + li) * cols..(p * nl + li + 1) * cols].iter().copied()).collect();
                    tickets.insert(layer, (li, e.tier.issue(li, &mine)?));
                }
            }
        }
        Ok(tickets)
    }

    /// Wait for Engram layer `l`'s records and upload them (right before the stage of layer `l`).
    fn engram_records(&self, ex: &Exec, l: usize, ticket: super::engram::EngramTicket, t: usize, stats: &mut PassStats) -> Result<()> {
        let e = self.engram.as_ref().unwrap();
        let rb = record_bytes(e.hasher.layout.head_dim);
        let mut rec = vec![0u8; t * e.hasher.layout.cols() * rb];
        let w0 = Instant::now();
        e.tier.wait(ticket, &mut rec)?;
        stats.engram_wait_us += w0.elapsed().as_micros() as u64;
        let name = format!("host.engram.{l}.records");
        let b = ex.ws.get(&name).ok_or_else(|| Error::backend(format!("no buffer {name}")))?.clone();
        self.backend.upload(&b.buf, 0, &rec, 0)
    }

    /// The logits (or argmax ids) of a pass that ran (a device sync), and its timing.
    fn finish_pass(&self, ex: &Exec, g: &Dsv41Graph, st: &SequenceState, t: usize, want: Want, mut stats: PassStats, e0: Instant, p0: Instant) -> Result<Option<Out>> {
        stats.enqueue_us = e0.elapsed().as_micros() as u64;
        if want == Want::Nothing {
            stats.total_us = p0.elapsed().as_micros() as u64;
            self.note_pass(t, stats);
            return Ok(None);
        }
        self.touch();
        let src = match want {
            Want::Ids => g.ids.as_ref().ok_or_else(|| Error::invalid("deepseek_v41: the graph has no argmax ids"))?,
            _ => &g.logits,
        };
        let lt = self.view(ex, src, st, t)?;
        let mut bytes = vec![0u8; lt.nbytes()];
        self.backend.download(&lt.buf, lt.offset, &mut bytes, 0)?;
        stats.total_us = p0.elapsed().as_micros() as u64;
        self.note_pass(t, stats);
        Ok(Some(if want == Want::Ids {
            Out::Ids(bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect())
        } else {
            Out::Logits(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        }))
    }

    /// Run graph `g` on `st` (its `t` rows at `st.len`): token ids `ids` (empty: none), the extra host inputs `host`
    /// (`(workspace name, bytes)`), Engram rows for the graph's Engram layers. Returns the graph's logits when asked.
    fn run_graph(&self, g: &Dsv41Graph, st: &SequenceState, t: usize, ids: &[u32], host: &[(&str, &[u8])], want: Want, p0: Instant) -> Result<Option<Out>> {
        if !ids.is_empty() && ids.len() != t {
            return Err(Error::invalid(format!("deepseek_v41: {} ids for a graph of {t} rows", ids.len())));
        }
        self.touch();
        let mut ex = self.exec.lock().unwrap();
        ex.verify = g.verify;
        self.upload_inputs(&ex, g, st.len, ids, host)?;
        let mut tickets = self.issue_engram(g, st, ids, t)?;
        let mut stats = PassStats { build_us: p0.elapsed().as_micros() as u64, ..Default::default() };
        let e0 = Instant::now();
        for (si, stage) in g.stages.iter().enumerate() {
            if let Some((_li, ticket)) = stage.layer.and_then(|l| tickets.remove(&l)) {
                self.engram_records(&ex, stage.layer.unwrap(), ticket, t, &mut stats)?;
            }
            self.run_stage(&mut ex, si, stage, st, t)?;
        }
        self.finish_pass(&ex, g, st, t, want, stats, e0, p0)
    }

    /// A device-position pass as CUDA graph replays (`INTARSIA_DSV41_GRAPHS=1`). `None`: run it eagerly instead (the
    /// first pass of its shape, which also warms every kernel up, or a capture that failed: graphs then stay off).
    #[allow(clippy::too_many_arguments)]
    fn run_captured(
        &self,
        st: &SequenceState,
        ids: &[u32],
        key: (usize, u8),
        host: &[(&str, &[u8])],
        want: Want,
        p0: Instant,
        build: &dyn Fn() -> Result<Dsv41Graph>,
    ) -> Result<Option<Option<Out>>> {
        let t = ids.len();
        let mut gs = self.graphs.lock().unwrap();
        if !gs.warm.contains(&key) {
            gs.warm.insert(key);
            return Ok(None);
        }
        self.touch();
        let mut ex = self.exec.lock().unwrap();
        let stale = gs.captured.get(&key).is_none_or(|c| c.state_id != st.id || c.scratch_gen != ex.scratch_gen);
        if stale {
            // a rank without a logits sync after its passes (TP rank 1) may still be running the old recording
            if gs.captured.contains_key(&key) {
                self.backend.synchronize(0)?;
            }
            gs.captured.remove(&key);
            let g = build()?;
            match self.capture(&mut ex, &g, st, t) {
                Ok(segs) => {
                    let n: usize = segs.len();
                    let steps = g.stages.iter().flat_map(|s| s.steps.iter());
                    let (all, exl3) = steps.fold((0, 0), |(a, e), s| (a + 1, e + usize::from(s.op == crate::ops::Op::MatmulExl3)));
                    let writes = ws_writes(&g);
                    gs.captured.insert(key, Captured { g, segs, state_id: st.id, scratch_gen: ex.scratch_gen, writes });
                    let what = ["pass", "verify pass", "drafting block"][key.1 as usize];
                    eprintln!("dsv41: captured the {t}-row {what} as {n} CUDA graph segment(s), {all} steps, {exl3} EXL3 linears (unix {})", unix_secs());
                }
                Err(e) => {
                    eprintln!("dsv41: CUDA graph capture of a {t}-row graph failed ({e}); graphs off, eager from now on");
                    self.graphs_on.store(false, std::sync::atomic::Ordering::Relaxed);
                    return Ok(None);
                }
            }
        }
        let c = gs.captured.get(&key).expect("captured");
        self.upload_inputs(&ex, &c.g, st.len, ids, host)?;
        // fills (constant table cells) are host uploads: re-applied before every replay (cached while untouched)
        for stage in &c.g.stages {
            for (_, e) in &stage.extras {
                if matches!(e, Extra::Fill { .. }) {
                    self.extra_inner(&mut ex, e, st, t)?;
                }
            }
        }
        let mut tickets = self.issue_engram(&c.g, st, ids, t)?;
        let mut stats = PassStats { build_us: p0.elapsed().as_micros() as u64, ..Default::default() };
        let e0 = Instant::now();
        if self.engram_one_graph {
            let mut all: Vec<_> = tickets.drain().collect();
            all.sort_by_key(|(l, _)| *l);
            for (l, (_li, ticket)) in all {
                self.engram_records(&ex, l, ticket, t, &mut stats)?;
            }
        }
        for (first, _end, h) in &c.segs {
            if let Some(l) = c.g.stages[*first].layer {
                if let Some((_li, ticket)) = tickets.remove(&l) {
                    self.engram_records(&ex, l, ticket, t, &mut stats)?;
                }
            }
            self.backend.graph_replay(h, 0)?;
        }
        for (base, lo, hi) in &c.writes {
            if let Some(&(o, n, _)) = ex.filled.get(base.as_str()) {
                let es = ex.ws.get(base.as_str()).map_or(1, |w| esz(w.dtype));
                if *lo < (o + n) * es && o * es < *hi {
                    ex.filled.remove(base.as_str());
                }
            }
        }
        let out = self.finish_pass(&ex, &c.g, st, t, want, stats, e0, p0)?;
        gs.replays[key.1 as usize] += 1;
        Ok(Some(out))
    }

    /// Record `g`: one segment, or (`engram_one_graph` off) segments split before every Engram layer's stage. Every kernel already ran once eagerly (the
    /// warm-up pass), the scratch is grown first, the fills applied; nothing inside the recording uploads or allocates.
    fn capture(&self, ex: &mut Exec, g: &Dsv41Graph, st: &SequenceState, t: usize) -> Result<Vec<(usize, usize, GraphHandle)>> {
        ex.verify = g.verify;
        // grow the scratch to the largest need of the graph outside the recording
        let mut need = 0usize;
        for (si, stage) in g.stages.iter().enumerate() {
            for step in &stage.steps {
                let (call, ki) = self.bind(ex, si, step, st, t)?;
                need = need.max(self.step_scratch(self.kernel_list[ki].scratch_bytes(&ShapeInfo::from_call(&call)), step.lane));
            }
            for (_, e) in &stage.extras {
                if matches!(e, Extra::Fill { .. }) {
                    self.extra_inner(ex, e, st, t)?;
                }
            }
        }
        self.grow_scratch(ex, need)?;
        let engram: Vec<usize> = self.engram.as_ref().map_or(Vec::new(), |e| e.hasher.layout.layer_ids.clone());
        let mut cuts: Vec<usize> = vec![0];
        for (si, stage) in g.stages.iter().enumerate() {
            if si > 0 && !self.engram_one_graph && stage.layer.is_some_and(|l| engram.contains(&l)) {
                cuts.push(si);
            }
        }
        cuts.push(g.stages.len());
        ex.pf_plan = if self.l2_prefetch { self.prefetch_plan(ex, g, st, t)? } else { HashMap::new() };
        let mut segs = Vec::new();
        ex.capturing = true;
        let mut res = Ok(());
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let rec = self.backend.graph_capture(0, &mut || {
                for si in a..b {
                    self.run_stage(ex, si, &g.stages[si], st, t)?;
                }
                if ex.pf_out {
                    self.backend.stream_wait(0, 1)?;
                    ex.pf_out = false;
                }
                Ok(())
            });
            match rec {
                Ok(h) => segs.push((a, b, h)),
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        ex.capturing = false;
        res.map(|_| segs)
    }

    /// Which weights a recording prefetches after which step ([`Self::l2_prefetch`]): only the EXL3 trellises of a layer's
    /// attention linears, found by step name.
    fn prefetch_plan(&self, ex: &mut Exec, g: &Dsv41Graph, st: &SequenceState, t: usize) -> Result<HashMap<(usize, usize), Vec<TensorRef>>> {
        let mut plan = HashMap::new();
        let trellis = |rt: &Self, ex: &mut Exec, si: usize, step: &super::graph::Step| -> Result<TensorRef> {
            Ok(rt.bind(ex, si, step, st, t)?.0.inputs[1].clone())
        };
        let linear = |s: &super::graph::Step| s.op == crate::ops::Op::MatmulExl3 && s.what.contains(".attn.") && !s.what.contains(".indexer.") && !s.what.contains(".compressor.");
        for (si, stage) in g.stages.iter().enumerate() {
            for (j, step) in stage.steps.iter().enumerate() {
                if linear(step) && step.what.ends_with(".wq_b") {
                    let mut r = Vec::new();
                    for s in stage.steps[j + 1..].iter().filter(|s| linear(s) && (s.what.contains(".wo_a") || s.what.ends_with(".wo_b"))) {
                        r.push(trellis(self, ex, si, s)?);
                    }
                    if !r.is_empty() {
                        plan.insert((si, j), r);
                    }
                } else if step.op == crate::ops::Op::MoeExperts {
                    let next = g.stages.iter().enumerate().skip(si + 1).find(|(_, n)| n.steps.iter().any(|s| linear(s) && s.what.ends_with(".wq_a")));
                    if let Some((ni, n)) = next {
                        let mut r = Vec::new();
                        for s in n.steps.iter().filter(|s| linear(s) && (s.what.ends_with(".wq_a") || s.what.ends_with(".wkv") || s.what.ends_with(".wq_b"))) {
                            r.push(trellis(self, ex, ni, s)?);
                        }
                        if !r.is_empty() {
                            plan.insert((si, j), r);
                        }
                    }
                }
            }
        }
        Ok(plan)
    }

    /// The scratch a step needs of the shared buffer: `need`, or with [`Self::lanes`] and a lane `l` its slice
    /// `(l - 1) * stride` onwards (every lane step of a fork has the same shape, so one stride serves them all).
    fn step_scratch(&self, need: usize, lane: u8) -> usize {
        if self.lanes && lane > 0 && need > 0 {
            lane as usize * need.next_multiple_of(LANE_STRIDE_ALIGN)
        } else {
            need
        }
    }

    fn grow_scratch(&self, ex: &mut Exec, need: usize) -> Result<()> {
        if need > 0 && ex.scratch.as_ref().is_none_or(|s| s.nbytes() < need) {
            if ex.capturing {
                return Err(Error::backend(format!("deepseek_v41: kernel scratch of {need} bytes needed inside a CUDA graph capture")));
            }
            let bytes = need.next_power_of_two().max(1 << 20);
            let b = self.backend.alloc(bytes, MemKind::Device)?;
            ex.scratch = Some(TensorRef::new(b, DType::U8, &[bytes]));
            ex.scratch_gen += 1;
            eprintln!("dsv41: kernel scratch grew to {bytes} bytes (unix {})", unix_secs());
        }
        Ok(())
    }

    /// A step as a checked call on its buffers (scratch not yet set) and its kernel (chosen once per step key).
    fn bind(&self, ex: &mut Exec, si: usize, step: &super::graph::Step, st: &SequenceState, t: usize) -> Result<(OpCall, usize)> {
        let inputs: Vec<TensorRef> = step.inputs.iter().map(|a| self.view(ex, a, st, t)).collect::<Result<_>>()?;
        let outputs: Vec<TensorRef> = step.outputs.iter().map(|a| self.view(ex, a, st, t)).collect::<Result<_>>()?;
        let mut call = OpCall::new(step.op, inputs, outputs, step.attrs.clone());
        if step.pos {
            let p = ex.ws.get("host.pos").ok_or_else(|| Error::backend("no position buffer"))?;
            call.pos = Some(TensorRef { buf: p.buf.clone(), offset: p.offset, dtype: DType::I32, shape: vec![1] });
        }
        call.check().map_err(|e| Error::model(format!("deepseek_v41 step {}: {e}", step.what)))?;
        let cls = if t == 1 { 0 } else if t <= 16 { 1 } else { 2 };
        let key = (si, step.what.clone(), step.op, cls | if step.pos { 4 } else { 0 } | if ex.verify { 8 } else { 0 });
        let ki = match ex.kernels.get(&key) {
            Some(&i) => i,
            None => {
                let k = self.reg.select(&call).map_err(|e| Error::model(format!("deepseek_v41 step {} ({}): {e}", step.what, step.op.as_str())))?;
                let mut name = k.name.as_str();
                if step.op == crate::ops::Op::MatmulExl3 && (t == 1 || (ex.verify && self.splitk_verify)) && name == "matmul_exl3_gemv" {
                    let n = call.outputs.first().map_or(0, |o| o.shape.get(1).copied().unwrap_or(0));
                    let s = ShapeInfo::from_call(&call);
                    if n / 128 < self.splitk_grid && self.kernel_list.iter().any(|x| x.name == "matmul_exl3_splitk" && x.op == step.op && x.accepts(&s)) {
                        name = "matmul_exl3_splitk";
                    }
                    // INTARSIA_DSV41_EXL3_CP: the cp.async split-K on the grids it lists
                    let g = n / 128;
                    if self.cp_grids.contains(&g) && self.kernel_list.iter().any(|x| x.name == "matmul_exl3_cp" && x.op == step.op && x.accepts(&s)) {
                        name = "matmul_exl3_cp";
                    }
                }
                let i = self
                    .kernel_list
                    .iter()
                    .position(|x| x.name == name && x.op == step.op)
                    .ok_or_else(|| Error::backend(format!("deepseek_v41: selected kernel {} is not in the backend's table", k.name)))?;
                ex.kernels.insert(key, i);
                i
            }
        };
        Ok((call, ki))
    }

    /// End a lane fork: the pass's own stream waits for every lane that ran.
    fn join_lanes(&self, fork: &mut Option<(u64, [bool; 4])>) -> Result<()> {
        if let Some((_, used)) = fork.take() {
            for (i, u) in used.iter().enumerate() {
                if *u {
                    self.backend.stream_wait(0, LANE_STREAM0 + i as u32)?;
                }
            }
        }
        Ok(())
    }

    fn run_stage(&self, ex: &mut Exec, si: usize, stage: &Stage, st: &SequenceState, t: usize) -> Result<()> {
        let mut extras = stage.extras.iter().peekable();
        // lanes only inside a recording; the open fork: the event the lanes wait for, and which lanes did
        let fork_on = self.lanes && ex.capturing;
        let mut fork: Option<(u64, [bool; 4])> = None;
        for (j, step) in stage.steps.iter().enumerate() {
            while let Some((_, e)) = extras.next_if(|(k, _)| *k == j) {
                self.join_lanes(&mut fork)?;
                self.extra(ex, e, st, t)?;
            }
            let lane = if fork_on { step.lane.min(4) as usize } else { 0 };
            let mut stream = 0;
            if lane == 0 {
                self.join_lanes(&mut fork)?;
            } else {
                let f = match &mut fork {
                    Some(f) => f,
                    None => fork.insert((self.backend.stream_record(0)?, [false; 4])),
                };
                stream = LANE_STREAM0 + lane as u32 - 1;
                if !f.1[lane - 1] {
                    self.backend.stream_wait_for(stream, f.0)?;
                    f.1[lane - 1] = true;
                }
            }
            for a in &step.outputs {
                if let Buf::Ws(n) = &a.buf {
                    // a write over a filled range undoes the fill (a shorter view that ends before it does not)
                    let base = ws_base(n);
                    let len = a.shape.iter().product::<usize>() * esz(a.dtype);
                    let start = ws_index(n).map_or(0, |i| i * len);
                    if let Some(&(o, c, _)) = ex.filled.get(base) {
                        let es = ex.ws.get(base).map_or(1, |w| esz(w.dtype));
                        if start < (o + c) * es && o * es < start + len {
                            ex.filled.remove(base);
                        }
                    }
                }
            }
            let (mut call, ki) = self.bind(ex, si, step, st, t)?;
            let kimpl = &self.kernel_list[ki];
            let kname = kimpl.name.as_str();
            let need = kimpl.scratch_bytes(&ShapeInfo::from_call(&call));
            if need > 0 {
                self.grow_scratch(ex, self.step_scratch(need, lane as u8))?;
                let off = if lane > 0 { (lane - 1) * need.next_multiple_of(LANE_STRIDE_ALIGN) } else { 0 };
                call.scratch = Some(ex.scratch.as_ref().unwrap().reshaped(&[ex.scratch.as_ref().unwrap().numel()]).rows(off, need));
            }
            let op_t0 = self.opprof.as_ref().map(|_| Instant::now());
            self.backend.launch(kname, &call, stream).map_err(|e| Error::backend(format!("deepseek_v41 step {} ({kname}): {e}", step.what)))?;
            if ex.capturing {
                if let Some(r) = ex.pf_plan.get(&(si, j)) {
                    self.backend.stream_wait(1, 0)?;
                    self.backend.l2_prefetch(r, 16 << 20, 1)?;
                    ex.pf_out = true;
                }
            }
            if let Some(t0) = op_t0 {
                self.backend.synchronize(0)?;
                self.opprof_add(t, kname, t0);
            }
            if let Some(d) = dump_dir().filter(|_| dump_rows().is_some()) {
                // INTARSIA_DSV41_DUMP_POS=<positions>[ INTARSIA_DSV41_DUMP_WHAT=<substrings>]: instead of every output, only the rows at
                // these absolute positions of the outputs whose first dimension is the chunk's row count, of the steps whose name
                // contains one of the substrings. File `{what}-o{k}.p{pos}.{dtype}.{row shape}.bin` carries no chunk-dependent number,
                // so dumps of runs with different chunk sizes compare file by file (tools/dsv41/chunkdiff.py).
                let (pos, what_f) = dump_rows().unwrap();
                if what_f.is_empty() || what_f.iter().any(|w| step.what.contains(w.as_str())) {
                    let pos0 = st.len;
                    let hit: Vec<usize> = pos.iter().copied().filter(|&p| p >= pos0 && p < pos0 + t).collect();
                    if !hit.is_empty() {
                        self.backend.synchronize(0)?;
                        for (k, o) in call.outputs.iter().enumerate() {
                            // rows of the chunk, or a one-row output of a longer chunk (the head: the chunk's last row)
                            let last_only = o.shape.first() == Some(&1) && t > 1;
                            if o.shape.first() != Some(&t) && !last_only {
                                continue;
                            }
                            let what: String = step.what.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' { ch } else { '-' }).collect();
                            let shape = o.shape[1..].iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x");
                            for &p in &hit {
                                if last_only && p != pos0 + t - 1 {
                                    continue;
                                }
                                let bytes = self.backend.download_row_major(&o.rows(if last_only { 0 } else { p - pos0 }, 1))?;
                                let f = d.join(format!("{what}-o{k}.p{p}.{}.{shape}.bin", format!("{:?}", o.dtype).to_lowercase()));
                                std::fs::write(&f, &bytes).map_err(|e| Error::backend(format!("dump {}: {e}", f.display())))?;
                            }
                        }
                    }
                }
            } else if let Some(d) = dump_dir() {
                // INTARSIA_DSV41_DUMP=<dir>: every op's outputs after it ran (cross-backend diffs, tools/dsv41/dumpdiff.py)
                self.backend.synchronize(0)?;
                let n = DUMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                for (k, o) in call.outputs.iter().enumerate() {
                    let bytes = self.backend.download_row_major(o)?;
                    let what: String = step.what.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' { ch } else { '-' }).collect();
                    let shape = o.shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x");
                    let f = d.join(format!("{n:05}-{what}-o{k}.{}.{shape}.bin", format!("{:?}", o.dtype).to_lowercase()));
                    std::fs::write(&f, &bytes).map_err(|e| Error::backend(format!("dump {}: {e}", f.display())))?;
                }
            }
        }
        self.join_lanes(&mut fork)?;
        for (_, e) in extras {
            self.extra(ex, e, st, t)?;
        }
        Ok(())
    }

    fn extra(&self, ex: &mut Exec, e: &Extra, st: &SequenceState, t: usize) -> Result<()> {
        let Some(_) = self.opprof.as_ref() else { return self.extra_inner(ex, e, st, t) };
        let t0 = Instant::now();
        self.extra_inner(ex, e, st, t)?;
        self.backend.synchronize(0)?;
        let kind = match e {
            Extra::AllReduce { .. } => "[all_reduce]",
            Extra::AllReduceInto { .. } => "[all_reduce_into]",
            Extra::AllGather { .. } => "[all_gather]",
            Extra::Fill { .. } => "[fill]",
        };
        self.opprof_add(t, kind, t0);
        Ok(())
    }

    /// `INTARSIA_DSV41_OPPROF=1`: device time per kernel / collective (a sync after every step: slow, diagnosis only),
    /// split into one-row passes and longer ones; `print_opprof` writes the table.
    fn opprof_add(&self, t: usize, name: &str, t0: Instant) {
        let us = t0.elapsed().as_micros() as u64;
        let mut m = self.opprof.as_ref().unwrap().lock().unwrap();
        let e = m.entry((t == 1, name.to_string())).or_insert((0, 0));
        e.0 += 1;
        e.1 += us;
    }

    /// The `INTARSIA_DSV41_OPPROF` table (per pass class: kernel, calls, total ms, share), then reset.
    pub fn print_opprof(&self) {
        let Some(p) = self.opprof.as_ref() else { return };
        let mut m = p.lock().unwrap();
        for decode in [true, false] {
            let mut v: Vec<_> = m.iter().filter(|((d, _), _)| *d == decode).map(|((_, n), &(c, us))| (n.clone(), c, us)).collect();
            let total: u64 = v.iter().map(|x| x.2).sum();
            if total == 0 {
                continue;
            }
            v.sort_by(|a, b| b.2.cmp(&a.2));
            eprintln!("dsv41 opprof ({} passes): total {:.1} ms", if decode { "one-row" } else { "multi-row" }, total as f64 / 1e3);
            for (n, c, us) in v.iter().take(25) {
                eprintln!("  {n:<28} {c:>8} calls {:>10.2} ms {:>5.1}%", *us as f64 / 1e3, 100.0 * *us as f64 / total as f64);
            }
        }
        m.clear();
    }

    fn extra_inner(&self, ex: &mut Exec, e: &Extra, st: &SequenceState, t: usize) -> Result<()> {
        let comm = || self.comm.as_ref().ok_or_else(|| Error::invalid("deepseek_v41: a collective without a communicator"));
        match e {
            Extra::AllReduce { x } => comm()?.all_reduce_sum(&self.view(ex, x, st, t)?, 0),
            Extra::AllReduceInto { partial, out } => comm()?.all_reduce_sum_into(&self.view(ex, partial, st, t)?, &self.view(ex, out, st, t)?, 0),
            Extra::AllGather { send, recv } => comm()?.all_gather(&self.view(ex, send, st, t)?, &self.view(ex, recv, st, t)?, 0),
            Extra::Fill { buf, dtype, offset, count, value } => {
                let key = (*offset, *count, value.to_bits());
                if ex.filled.get(buf) == Some(&key) {
                    return Ok(());
                }
                if ex.capturing {
                    // a fill an earlier step of the same pass undid: it would have to run inside the recording
                    return Err(Error::backend(format!("deepseek_v41: fill of {buf} inside a CUDA graph capture")));
                }
                let one: Vec<u8> = match dtype {
                    DType::I32 => (*value as i32).to_le_bytes().to_vec(),
                    DType::F32 => (*value as f32).to_le_bytes().to_vec(),
                    d => return Err(Error::invalid(format!("deepseek_v41: fill of {d}"))),
                };
                let data = one.repeat(*count);
                let b = ex.ws.get(buf).ok_or_else(|| Error::backend(format!("no workspace {buf}")))?.clone();
                self.backend.upload(&b.buf, offset * esz(*dtype), &data, 0)?;
                ex.filled.insert(buf.clone(), key);
                Ok(())
            }
        }
    }

    /// The tensor an `Arg` names, as a view with the arg's dtype and shape.
    fn view(&self, ex: &Exec, a: &Arg, st: &SequenceState, t: usize) -> Result<TensorRef> {
        let numel: usize = a.shape.iter().product();
        let mk = |buf: &Buffer, offset: usize| -> Result<TensorRef> {
            let r = TensorRef { buf: buf.clone(), offset, dtype: a.dtype, shape: a.shape.clone() };
            r.check_bounds().map_err(|e| Error::backend(format!("deepseek_v41 {:?}: {e}", a.buf)))?;
            Ok(r)
        };
        match &a.buf {
            Buf::Weight(n) => {
                let w = self.weights.get(n).ok_or_else(|| Error::model(format!("deepseek_v41: weight {n} not loaded")))?;
                if w.nbytes() != numel * esz(a.dtype) {
                    return Err(Error::model(format!("deepseek_v41: weight {n} is {} {:?}, the graph reads {} {:?}", w.dtype, w.shape, a.dtype, a.shape)));
                }
                mk(&w.buf, w.offset)
            }
            Buf::Ws(n) => {
                if let Some(base) = n.strip_suffix(".last_row") {
                    let b = ex.ws.get(base).ok_or_else(|| Error::backend(format!("no workspace {base}")))?;
                    let row = numel * esz(a.dtype);
                    return mk(&b.buf, b.offset + (t - 1) * row);
                }
                let base = ws_base(n);
                let b = ex.ws.get(base).ok_or_else(|| Error::backend(format!("no workspace {base}")))?;
                let off = ws_index(n).map_or(0, |i| i * numel * esz(a.dtype));
                let off = off + ws_row0(n).map_or(0, |r| r * (numel / a.shape.first().copied().unwrap_or(1).max(1)) * esz(a.dtype));
                mk(&b.buf, b.offset + off)
            }
            Buf::Host(n) => {
                let k = format!("host.{n}");
                let b = ex.ws.get(&k).ok_or_else(|| Error::backend(format!("no buffer {k}")))?;
                mk(&b.buf, b.offset)
            }
            Buf::Const(n) => {
                let c = self.consts.get(n).ok_or_else(|| Error::model(format!("deepseek_v41: unknown constant {n}")))?;
                if numel == 0 {
                    return Ok(TensorRef { buf: c.buf.clone(), offset: 0, dtype: a.dtype, shape: a.shape.clone() });
                }
                mk(&c.buf, c.offset)
            }
            Buf::Slot { slot, plane, row0 } => {
                let s = st.slots.get(*slot).ok_or_else(|| Error::backend(format!("no state slot {slot}")))?;
                let (buf, plane_bytes, row_bytes) = match s {
                    StateSlot::Ring(r) => (&r.buf, r.rows * r.width * esz(r.dtype), r.width * esz(r.dtype)),
                    StateSlot::Shared(r) => (&r.buf, r.cap_rows * r.width * esz(r.dtype), r.width * esz(r.dtype)),
                    _ => return Err(Error::backend(format!("state slot {slot} is not a ring / shared slot"))),
                };
                if numel == 0 {
                    return Ok(TensorRef { buf: buf.clone(), offset: 0, dtype: a.dtype, shape: a.shape.clone() });
                }
                mk(buf, plane * plane_bytes + row0 * row_bytes)
            }
        }
    }

    /// The graph of a pass (for plan output).
    pub fn graph(&self, t: usize, pos0: usize) -> Result<Dsv41Graph> {
        build_with(&self.cfg, &*self.shapes, GraphOpts { t, pos0, act: self.opts.act, cap: self.opts.cap }, self.opts.tp)
    }

    /// Download a workspace tensor (debugging, parity tests): `name` with `numel` elements of `dtype` from its first byte.
    pub fn read_ws(&self, name: &str, dtype: DType, numel: usize) -> Result<Vec<u8>> {
        let ex = self.exec.lock().unwrap();
        let b = ex.ws.get(name).ok_or_else(|| Error::invalid(format!("no workspace {name}")))?;
        let mut out = vec![0u8; numel * esz(dtype)];
        self.backend.download(&b.buf, b.offset, &mut out, 0)?;
        Ok(out)
    }

    /// Kernel chosen per op in the last passes (plan report): `(op, kernel)` counts.
    pub fn kernel_summary(&self) -> Vec<(String, usize)> {
        let ex = self.exec.lock().unwrap();
        let mut m: HashMap<String, usize> = HashMap::new();
        for &k in ex.kernels.values() {
            *m.entry(self.kernel_list[k].name.clone()).or_default() += 1;
        }
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort();
        v
    }

    pub fn scratch_bytes(&self) -> u64 {
        self.exec.lock().unwrap().scratch.as_ref().map_or(0, |s| s.nbytes() as u64)
    }
}

/// `name` without a `[i]` slice suffix.
fn ws_base(n: &str) -> &str {
    if let Some((b, r)) = n.split_once('@') {
        if r.parse::<usize>().is_ok() {
            return b;
        }
    }
    match n.strip_suffix(']').and_then(|x| x.rsplit_once('[')) {
        Some((b, i)) if i.parse::<usize>().is_ok() => b,
        _ => n.strip_suffix(".last_row").unwrap_or(n),
    }
}

/// `name@row`: a view of workspace `name` that starts `row` rows in (the decoder part of a bounded-replay prefill reads the
/// encoder part's streams from the first kept row).
fn ws_row0(n: &str) -> Option<usize> {
    n.split_once('@').and_then(|(_, r)| r.parse().ok())
}

fn ws_index(n: &str) -> Option<usize> {
    n.strip_suffix(']').and_then(|x| x.rsplit_once('[')).and_then(|(_, i)| i.parse().ok())
}

/// Fuse every layer's shared expert into its routed-expert tensors (the graph is built from the rank cache's own shapes:
/// `graph::fused_expert_shapes` derives the fused sizes from the routed arena plus the shared expert's trellis shapes) (a rank cache stores the routed arena of `E` experts
/// and the shared expert as three EXL3 linears, `layers.{l}.ffn.shared_experts.w1/w3/w2`): the arena grows by the shared
/// expert's tiles (device-to-device copies), scales and meta by one row (expert `E`). The shared tensors are dropped.
/// Every shared matrix must have the routed experts' (rank's) width and codebook. `marker(p)` says the codebook of
/// linear `p` (0 3inst, 1 mcg, 2 mul1), from its `{p}.codebook` tensor.
pub fn fuse_shared_experts(backend: &dyn Backend, weights: &mut HashMap<String, TensorRef>, n_layers: usize, dim: usize, experts: usize) -> Result<()> {
    let ps: Vec<(String, usize)> = (0..n_layers).map(|l| (format!("layers.{l}.ffn"), experts)).collect();
    fuse_shared_experts_at(backend, weights, &ps, dim)
}

/// [`fuse_shared_experts`] for the MoE blocks `(prefix, routed experts)` (`layers.{l}.ffn`, the DSpark `mtp.{i}.ffn`); a
/// prefix without a routed arena is skipped.
pub fn fuse_shared_experts_at(backend: &dyn Backend, weights: &mut HashMap<String, TensorRef>, prefixes: &[(String, usize)], dim: usize) -> Result<()> {
    use crate::quant::exl3_moe::{scale_offsets, META_COLS};
    let read = |t: &TensorRef| -> Result<Vec<u8>> {
        let mut v = vec![0u8; t.nbytes()];
        backend.download(&t.buf, t.offset, &mut v, 0)?;
        Ok(v)
    };
    for (p, experts) in prefixes {
        let (p, experts) = (p.as_str(), *experts);
        let Some(arena) = weights.get(&format!("{p}.experts.arena")).cloned() else { continue };
        let scales = weights.get(&format!("{p}.experts.scales")).cloned().ok_or_else(|| Error::model(format!("{p}: no experts.scales")))?;
        let meta = weights.get(&format!("{p}.experts.meta")).cloned().ok_or_else(|| Error::model(format!("{p}: no experts.meta")))?;
        let cbt = weights.get(&format!("{p}.experts.codebook")).cloned().ok_or_else(|| Error::model(format!("{p}: no experts.codebook")))?;
        if scales.shape.len() != 2 || scales.shape[0] != experts || meta.shape != [experts, META_COLS] {
            return Err(Error::model(format!("{p}: experts.scales {:?} / meta {:?} for {experts} experts", scales.shape, meta.shape)));
        }
        let cols = scales.shape[1];
        let inter = (cols - 3 * dim) / 3;
        let cb = read(&cbt)?;
        let mut tiles = Vec::new();
        for m in ["w1", "w3", "w2"] {
            let q = format!("{p}.shared_experts.{m}");
            let tr = weights.get(&format!("{q}.trellis")).cloned().ok_or_else(|| Error::model(format!("{q}.trellis not loaded")))?;
            if let Some(c) = weights.get(&format!("{q}.codebook")) {
                if read(c)? != cb {
                    return Err(Error::Unsupported(format!("{q}: codebook differs from the routed experts' (one codebook per fused launch)")));
                }
            }
            let (k, n) = (16 * tr.shape[0], 16 * tr.shape[1]);
            let want = if m == "w2" { (inter, dim) } else { (dim, inter) };
            if (k, n) != want {
                return Err(Error::model(format!("{q}: [{k} -> {n}], the fused launch needs [{} -> {}]", want.0, want.1)));
            }
            let k2 = tr.shape[2] / 8;
            let suh = weights.get(&format!("{q}.suh")).cloned().ok_or_else(|| Error::model(format!("{q}.suh not loaded")))?;
            let svh = weights.get(&format!("{q}.svh")).cloned().ok_or_else(|| Error::model(format!("{q}.svh not loaded")))?;
            tiles.push((tr, k2, read(&suh)?, read(&svh)?));
        }
        let old = arena.nbytes();
        let extra: usize = tiles.iter().map(|(t, ..)| t.nbytes()).sum();
        let nb = backend.alloc(old + extra, MemKind::Device)?;
        backend.copy(&nb, 0, &arena.buf, arena.offset, old, 0)?;
        let mut off = old;
        let mut mrow = vec![0i32; META_COLS];
        let mut srow = vec![0u8; cols * 2];
        let so = scale_offsets(dim, inter);
        for (j, (tr, k2, suh, svh)) in tiles.iter().enumerate() {
            if off % 16 != 0 {
                return Err(Error::model(format!("{p}: shared tiles not 16-byte aligned in the arena")));
            }
            backend.copy(&nb, off, &tr.buf, tr.offset, tr.nbytes(), 0)?;
            mrow[j] = (off / 16) as i32;
            mrow[3 + j] = *k2 as i32;
            let (a, b) = [(so[0], so[2]), (so[1], so[3]), (so[4], so[5])][j];
            srow[2 * a..2 * a + suh.len()].copy_from_slice(suh);
            srow[2 * b..2 * b + svh.len()].copy_from_slice(svh);
            off += tr.nbytes();
        }
        let mut sc = read(&scales)?;
        sc.extend_from_slice(&srow);
        let mut me = read(&meta)?;
        me.extend(mrow.iter().flat_map(|v| v.to_le_bytes()));
        let up = |data: &[u8]| -> Result<Buffer> {
            let b = backend.alloc(data.len(), MemKind::Device)?;
            backend.upload(&b, 0, data, 0)?;
            Ok(b)
        };
        weights.insert(format!("{p}.experts.arena"), TensorRef::new(nb, DType::I16, &[(old + extra) / 2]));
        weights.insert(format!("{p}.experts.scales"), TensorRef::new(up(&sc)?, DType::F16, &[experts + 1, cols]));
        weights.insert(format!("{p}.experts.meta"), TensorRef::new(up(&me)?, DType::I32, &[experts + 1, META_COLS]));
        for m in ["w1", "w3", "w2"] {
            for s in ["trellis", "suh", "svh", "codebook", "mcg", "mul1"] {
                weights.remove(&format!("{p}.shared_experts.{m}.{s}"));
            }
        }
        backend.synchronize(0)?;
    }
    Ok(())
}

/// The remote rank of a cross-process group runs the primary's passes on its own shard (`tp_remote::serve`); both ranks
/// run every collective of every pass (the head's gather included), only the primary reads the logits.
impl crate::tp_remote::RankModel for Dsv41Runtime {
    type State = SequenceState;
    fn new_state(&self) -> Result<SequenceState> {
        Dsv41Runtime::new_state(self)
    }
    fn forward_rank(&self, st: &mut SequenceState, ids: &[u32], _graph: bool) -> Result<()> {
        self.forward(st, ids, false).map(|_| ())
    }
    fn truncate(&self, st: &mut SequenceState, len: usize) -> Result<()> {
        st.truncate(len, &*self.backend)
    }
    fn snapshot(&self, _st: &mut SequenceState, _keep: usize) -> Result<()> {
        Err(Error::Unsupported("deepseek_v41 keeps no recurrent snapshots".into()))
    }
    fn state_len(st: &SequenceState) -> usize {
        st.len
    }
    fn verify_rank(&self, st: &mut SequenceState, ids: &[u32]) -> Result<()> {
        self.forward_rows_ids(st, ids).map(|_| ())
    }
    fn draft_rank(&self, st: &mut SequenceState, ids: &[u32]) -> Result<()> {
        let first = *ids.first().ok_or_else(|| Error::invalid("deepseek_v41: a drafting round without its first id"))?;
        self.draft(st, first).map(|_| ())
    }
}

/// What a pass hands back to the host.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Want {
    /// Nothing (no device sync).
    Nothing,
    /// The graph's `logits`.
    Logits,
    /// The graph's `ids` (the argmax of every logits row).
    Ids,
}

impl From<bool> for Want {
    fn from(logits: bool) -> Want {
        if logits {
            Want::Logits
        } else {
            Want::Nothing
        }
    }
}

enum Out {
    Logits(Vec<f32>),
    Ids(Vec<u32>),
}

impl Out {
    fn logits(self) -> Vec<f32> {
        match self {
            Out::Logits(l) => l,
            Out::Ids(_) => unreachable!("ids where logits were asked"),
        }
    }
    fn ids(self) -> Vec<u32> {
        match self {
            Out::Ids(i) => i,
            Out::Logits(_) => unreachable!("logits where ids were asked"),
        }
    }
}

/// `INTARSIA_DSV41_DUMP_POS` (comma list of absolute positions) and `INTARSIA_DSV41_DUMP_WHAT` (comma list of step-name substrings).
fn dump_rows() -> Option<(&'static [usize], &'static [String])> {
    static D: std::sync::OnceLock<Option<(Vec<usize>, Vec<String>)>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let pos: Vec<usize> = std::env::var("INTARSIA_DSV41_DUMP_POS").ok()?.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let what: Vec<String> = std::env::var("INTARSIA_DSV41_DUMP_WHAT").ok().map(|w| w.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect()).unwrap_or_default();
        Some((pos, what))
    })
    .as_ref()
    .map(|(p, w)| (p.as_slice(), w.as_slice()))
}

static DUMP_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn dump_dir() -> Option<&'static std::path::Path> {
    static D: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let d = std::path::PathBuf::from(std::env::var_os("INTARSIA_DSV41_DUMP")?);
        std::fs::create_dir_all(&d).ok()?;
        Some(d)
    })
    .as_deref()
}
