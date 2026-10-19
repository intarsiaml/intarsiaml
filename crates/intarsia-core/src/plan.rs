//! Plan resolution (ARCHITECTURE 3.6): resolve the kernel of every `(op, shape)` the recipe needs
//! when the model loads, run admission on those exact shapes, and describe the result.
//!
//! The plan is **recorded from the real forward path**: [`Model::dry_run`] executes `run_chunk` and
//! the head with a recorder installed, so `exec` selects a kernel and logs `(op, shape, kernel)`
//! instead of launching it. Blocks other agents add through `exec` are planned automatically.
//!
//! Shapes probed: decode (`T = 1`) at the first and the last context position, and prefill chunks
//! for the sizes in [`probe_sizes`] (every size up to 16, then each power of two with its
//! neighbours up to the chunk size), because kernel predicates switch on `m`.
//!
//! A missing kernel fails the load with the full list of `(op, shape)` pairs that have none.
//!
//! Admission (`admit`): every kernel the plan selects is checked against the reference backend on
//! the plan's exact shapes. To bound the cost of the f32 reference, rows are clamped to
//! [`CAPS`] (`M <= 32`), matmul/logits `N` to 16384 and attention positions to 2000; `K`, head
//! counts, head dims, dtypes and attributes are exact. Results are cached per host fingerprint
//! under `~/.cache/intarsia/admission/`.

use crate::admission::{case_from_call, case_label, judge, ShapeCaps};
use crate::backend::{Backend, Buffer, BufferImpl};
use crate::error::{Error, Result};
use crate::exec::run_host;
use crate::model::Model;
use crate::ops::{Op, OpCall, ShapeInfo};
use crate::registry::{Admission, Exactness};
use crate::rng::SplitMix64;
use crate::state::{AttnKvState, RecurrentState, SequenceState, SlotSpec, StateSlot};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Cost bounds for admission on plan shapes (see module docs).
pub const CAPS: ShapeCaps = ShapeCaps { rows: 32, n: 16384, pos0: 2000, emb_vocab: 2048 };

#[derive(Clone, Debug, Default)]
pub struct PlanOptions {
    /// Run admission on the plan's exact shapes (needs a reference backend).
    pub admit: bool,
    /// After admission only `Admitted` kernels are selectable (serve). Implies nothing about
    /// whether admission ran: with `admit = false` and `require_admitted = true` nothing is selectable.
    pub require_admitted: bool,
    /// Select reference kernels only (cpu `ref_*`). A backend without reference kernels fails the load.
    pub force_reference: bool,
    /// Admission cache directory (default `~/.cache/intarsia/admission`).
    pub cache_dir: Option<PathBuf>,
    /// Do not read or write the admission cache.
    pub no_cache: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Decode,
    Prefill,
    /// Speculative verify pass (`Model::forward_verify`, only when `Model::enable_verify` was called).
    Verify,
    /// Batched decode step (`Model::forward_batch`, only when `Model::enable_batch` was called): `t` is the number of
    /// sequences, `pos0` the length of each.
    Batch,
}

/// One op call of the plan with the kernel the registry selected for it.
#[derive(Clone, Debug)]
pub struct PlannedCall {
    pub phase: Phase,
    /// Tokens in the probed forward pass.
    pub t: usize,
    pub pos0: usize,
    pub kernel: String,
    pub call: OpCall,
    /// Recorded for the all-position logits head (accuracy tooling), not part of a serving step.
    pub extra: bool,
}

/// A collective (all-reduce) the probed forward pass makes; recorded instead of run.
#[derive(Clone, Debug)]
pub struct PlannedComm {
    pub phase: Phase,
    pub t: usize,
    pub pos0: usize,
    /// Payload bytes of this rank's tensor.
    pub bytes: usize,
    pub extra: bool,
}

/// Recorder installed by [`Model::dry_run`]; `exec` feeds it instead of launching.
#[derive(Debug, Default)]
pub struct Recorder {
    pub phase: Option<Phase>,
    pub t: usize,
    pub pos0: usize,
    /// Set while the extra (accuracy) head is being recorded.
    pub extra: bool,
    pub calls: Vec<PlannedCall>,
    /// Collectives of the pass (tensor parallelism).
    pub comm: Vec<PlannedComm>,
    /// `(op, description)` of every call that found no kernel.
    pub missing: Vec<(Op, String)>,
}

/// An `(op, kernel)` row of the resolved plan.
#[derive(Clone, Debug, Serialize)]
pub struct PlanOp {
    pub op: Op,
    pub kernel: String,
    pub backend: String,
    pub exactness: Exactness,
    pub admitted: bool,
    pub calls_per_decode_step: usize,
    pub scratch_bytes: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ResolvedPlan {
    pub ops: Vec<PlanOp>,
    /// Unique `(kernel, shape)` calls, one per distinct case key (the admission work list).
    pub unique_calls: Vec<PlannedCall>,
    pub scratch_bytes: usize,
    /// The decode step selects the same kernel for every op at the first and the last position, so one captured
    /// graph is valid at every position (a kernel whose `when` predicate looks at `kv_len` would break that).
    pub decode_position_stable: bool,
    pub probes: usize,
    /// Tensor parallelism: all-reduces in one decode step (`T = 1`) and the bytes each rank contributes.
    pub comm_calls_per_decode_step: usize,
    pub comm_bytes_per_decode_step: usize,
    /// Prefill sizes probed.
    pub sizes: Vec<usize>,
}

/// Prefill sizes to probe for a chunk size: all up to 16, then each power of two and its neighbours.
pub fn probe_sizes(chunk: usize) -> Vec<usize> {
    let mut s = BTreeSet::new();
    for t in 1..=16.min(chunk) {
        s.insert(t);
    }
    let mut p = 32;
    while p / 2 < chunk {
        for t in [p - 1, p, p + 1] {
            if t <= chunk {
                s.insert(t);
            }
        }
        p *= 2;
    }
    s.insert(chunk);
    s.insert(chunk.saturating_sub(1).max(1));
    s.into_iter().collect()
}

#[derive(Debug)]
struct Phantom(usize);
impl BufferImpl for Phantom {
    fn bytes(&self) -> usize {
        self.0
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A sequence state with the right shapes and no memory behind it (for recording only).
pub fn phantom_state(specs: &[SlotSpec], cap: usize, len: usize) -> SequenceState {
    let mut slots = Vec::new();
    for s in specs {
        slots.push(match s {
            SlotSpec::AttnKv { n_kv_heads, head_dim, dtype } => {
                let bytes = n_kv_heads * cap * head_dim * dtype.size_bytes();
                StateSlot::AttnKv(AttnKvState {
                    k: Buffer::new(Phantom(bytes)),
                    v: Buffer::new(Phantom(bytes)),
                    len,
                    cap,
                    n_kv_heads: *n_kv_heads,
                    head_dim: *head_dim,
                    dtype: *dtype,
                })
            }
            SlotSpec::Recurrent { bytes } => {
                StateSlot::Recurrent(RecurrentState { buf: Buffer::new(Phantom((*bytes).max(16))), bytes: *bytes })
            }
        });
    }
    SequenceState { id: 0, slots, len, cap, tokens: vec![0; len], snapshots: Vec::new(), peers: Vec::new() }
}

fn shape_str(s: &ShapeInfo) -> String {
    format!(
        "m={} n={} k={} heads={}/{} hd={} kv_len={} act={:?} w={:?}",
        s.m, s.n, s.k, s.n_heads, s.n_kv_heads, s.head_dim, s.kv_len, s.act, s.weight
    )
}

/// Record the plan: probe decode and prefill shapes, collect calls and missing kernels.
fn record(model: &Model) -> Result<(Vec<PlannedCall>, Vec<usize>, (usize, usize))> {
    let chunk = model.prefill_chunk();
    let cap = model.plan.kv_cap_tokens.max(1);
    let sizes = probe_sizes(chunk);
    let mut probes: Vec<(Phase, usize, usize)> = vec![(Phase::Decode, 1, 0), (Phase::Decode, 1, cap - 1)];
    for &t in &sizes {
        if t > 1 {
            probes.push((Phase::Prefill, t, 0));
        }
    }
    // the last full chunk of a context-filling prompt
    if cap > chunk {
        probes.push((Phase::Prefill, chunk, cap - chunk));
    }
    // speculative verify passes (spec on only): every row count, at both ends of the context
    for t in 2..=model.verify_rows() {
        if t < cap {
            probes.push((Phase::Verify, t, 0));
            probes.push((Phase::Verify, t, cap - t));
        }
    }
    // batched decode steps (batching on only): every row count, all sequences at the start and at the end of the context
    for t in 2..=model.batch_rows() {
        probes.push((Phase::Batch, t, 0));
        probes.push((Phase::Batch, t, cap - 1));
    }
    let mut calls = Vec::new();
    let mut comm_decode = (0usize, 0usize);
    let mut decode0: Vec<Op> = Vec::new();
    // op -> (shapes without a kernel, first description, where it is needed)
    let mut missing: BTreeMap<Op, (BTreeSet<String>, String, BTreeSet<String>)> = BTreeMap::new();
    for (phase, t, pos0) in probes {
        // a batch step appends one token to each of its sequences
        if pos0 + if phase == Phase::Batch { 1 } else { t } > cap {
            continue;
        }
        let rec = model.dry_run(phase, t, pos0)?;
        if phase == Phase::Decode && pos0 == 0 {
            decode0 = rec.calls.iter().filter(|c| !c.extra).map(|c| c.call.op).collect();
        }
        if phase == Phase::Batch && pos0 == 0 && rec.missing.is_empty() {
            let batch: Vec<Op> = rec.calls.iter().filter(|c| !c.extra).map(|c| c.call.op).collect();
            check_batch_ops(&decode0, &batch, t)?;
        }
        for (op, m) in rec.missing {
            let e = missing.entry(op).or_insert_with(|| (BTreeSet::new(), m.clone(), BTreeSet::new()));
            e.0.insert(m);
            e.2.insert(format!("{phase:?} t={t}"));
        }
        if phase == Phase::Decode && pos0 == 0 {
            for c in rec.comm.iter().filter(|c| !c.extra) {
                comm_decode.0 += 1;
                comm_decode.1 += c.bytes;
            }
        }
        calls.extend(rec.calls);
    }
    if !missing.is_empty() {
        let b = model.backend.info();
        let ref_only = if model.registry.reference_only {
            " (reference kernels only: this backend registers none for it)"
        } else {
            ""
        };
        let mut msg =
            format!("{} op(s) of this model have no kernel on {}/{}{ref_only}:", missing.len(), b.backend, b.arch);
        for (op, (shapes, first, wh)) in &missing {
            let wh: Vec<&String> = wh.iter().take(4).collect();
            msg.push_str(&format!(
                "\n  - {op}: {} shape(s) without a kernel, e.g. {first}; needed by {}{}",
                shapes.len(),
                wh.iter().map(|w| w.as_str()).collect::<Vec<_>>().join(", "),
                if missing[op].2.len() > 4 { ", ..." } else { "" }
            ));
        }
        return Err(Error::NoKernel(msg));
    }
    Ok((calls, sizes, comm_decode))
}

/// Ops a batched step runs once per sequence (the attention core on that sequence's own cache); every other op runs once
/// for all rows. `GateSplitQkNormRopeKv` holds the per-row cache append like `QkNormRopeKv`, so it is listed, although
/// `Model::enable_batch` rejects gated attention today and no batched plan can contain it.
const PER_ROW_OPS: [Op; 6] = [Op::QkNormRopeKv, Op::GateSplitQkNormRopeKv, Op::QkNorm, Op::Rope, Op::KvAppend, Op::AttnDecode];

/// A batched step must be the solo decode step with its row-parallel ops widened: the same op multiset, the per-row
/// attention ops `rows` times. A different fusion choice (a fused pair in the solo step, two projections in the batch)
/// would change the arithmetic and with it the bits, so it fails the load rather than the identity.
fn check_batch_ops(decode: &[Op], batch: &[Op], rows: usize) -> Result<()> {
    let count = |v: &[Op], op: Op| v.iter().filter(|&&o| o == op).count();
    let mut all: Vec<Op> = decode.iter().chain(batch).copied().collect();
    all.sort();
    all.dedup();
    for op in all {
        let (d, b) = (count(decode, op), count(batch, op));
        let want = if PER_ROW_OPS.contains(&op) { d * rows } else { d };
        if b != want {
            return Err(Error::Unsupported(format!(
                "a {rows}-row batched step runs {op} {b} times where {want} follow from the solo decode step ({d}): \
                 the two would fuse differently, so rows would not be bitwise the solo step"
            )));
        }
    }
    Ok(())
}

/// Case key shared by dedupe and the admission cache: kernel + clamped host case label.
fn case_key(model_kernel: &str, label: &str) -> String {
    format!("{model_kernel}|{label}")
}

/// Describe the first pair of probed decode positions whose kernel lists differ and what differs
/// (`ops differ between pos 0 and pos 4096: attn_decode: a vs b`); `None` when all positions agree.
fn position_diff(by_pos: &BTreeMap<usize, Vec<(Op, &str)>>) -> Option<String> {
    let mut it = by_pos.iter();
    let (p0, first) = it.next()?;
    let (p1, other) = it.find(|(_, v)| *v != first)?;
    let mut diffs: Vec<String> = Vec::new();
    for i in 0..first.len().max(other.len()) {
        let (a, b) = (first.get(i), other.get(i));
        if a != b {
            let name = a.or(b).map(|x| x.0.as_str().to_string()).unwrap_or_default();
            let k = |x: Option<&(Op, &str)>| x.map_or("(none)".to_string(), |x| x.1.to_string());
            let d = format!("{name}: {} vs {}", k(a), k(b));
            if !diffs.contains(&d) {
                diffs.push(d);
            }
        }
    }
    Some(format!("ops differ between pos {p0} and pos {p1}: {}", diffs.join("; ")))
}

/// Resolve every `(op, shape)` of the model against the registry and (re)allocate the scratch
/// workspace. Idempotent; call again after the registry changed (admission, `--reference`).
pub fn resolve(model: &mut Model) -> Result<()> {
    let (calls, sizes, comm_decode) = record(model)?;
    let mut scratch = 0usize;
    let mut per_op: BTreeMap<(Op, String), PlanOp> = BTreeMap::new();
    let mut unique: BTreeMap<String, PlannedCall> = BTreeMap::new();
    for c in &calls {
        let k =
            model.registry.kernel(&c.kernel).ok_or_else(|| Error::Other(format!("kernel {} vanished", c.kernel)))?;
        let s = ShapeInfo::from_call(&c.call);
        let sb = k.scratch_bytes(&s);
        scratch = scratch.max(sb);
        let e = per_op.entry((c.call.op, c.kernel.clone())).or_insert_with(|| PlanOp {
            op: c.call.op,
            kernel: c.kernel.clone(),
            backend: k.backend.clone(),
            exactness: k.exact,
            admitted: model.registry.admission(&c.kernel) == Some(Admission::Admitted) || k.is_reference,
            calls_per_decode_step: 0,
            scratch_bytes: 0,
        });
        e.scratch_bytes = e.scratch_bytes.max(sb);
        if c.phase == Phase::Decode && c.pos0 == 0 && !c.extra {
            e.calls_per_decode_step += 1;
        }
        let label = case_label(&c.call, &CAPS).unwrap_or_else(|| format!("{} {}", c.call.op, shape_str(&s)));
        unique.entry(case_key(&c.kernel, &label)).or_insert_with(|| c.clone());
    }
    model.reserve_scratch(scratch)?;
    // decode kernels per probed position: graph replay needs them identical across positions
    let mut by_pos: BTreeMap<usize, Vec<(Op, &str)>> = BTreeMap::new();
    for c in calls.iter().filter(|c| c.phase == Phase::Decode && !c.extra) {
        by_pos.entry(c.pos0).or_default().push((c.call.op, c.kernel.as_str()));
    }
    let decode_position_stable = by_pos.values().all(|v| Some(v) == by_pos.values().next());
    if !decode_position_stable && model.backend.supports_graphs() {
        if let Some(why) = position_diff(&by_pos) {
            eprintln!("WARNING: decode graph disabled: {why}");
        }
    }
    model.resolved = ResolvedPlan {
        decode_position_stable,
        ops: per_op.into_values().collect(),
        unique_calls: unique.into_values().collect(),
        scratch_bytes: scratch,
        probes: calls.len(),
        comm_calls_per_decode_step: comm_decode.0,
        comm_bytes_per_decode_step: comm_decode.1,
        sizes,
    };
    Ok(())
}

// ---------------------------------------------------------------------------
// admission cache
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CacheEntry {
    pass: bool,
    max_abs: f64,
    rel_l2: f64,
    error: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    fingerprint: String,
    entries: BTreeMap<String, CacheEntry>,
}

fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Host + kernel-pack identity. The running executable's size and mtime stand in for the kernel
/// pack hash: rebuilding (new or changed kernels) invalidates cached admission results.
pub fn host_fingerprint(backend: &dyn Backend) -> String {
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| {
            let mt =
                m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
            format!("{}:{}", m.len(), mt)
        })
        .unwrap_or_default();
    format!("{}|exe:{}|core:{}", backend.fingerprint(), exe, env!("CARGO_PKG_VERSION"))
}

struct Cache {
    path: Option<PathBuf>,
    file: CacheFile,
    dirty: bool,
}

impl Cache {
    fn open(opts: &PlanOptions, fp: &str, backend: &str, arch: &str) -> Cache {
        let mut c = Cache {
            path: None,
            file: CacheFile { version: 1, fingerprint: fp.to_string(), ..Default::default() },
            dirty: false,
        };
        if opts.no_cache {
            return c;
        }
        let dir = opts.cache_dir.clone().or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache").join("intarsia").join("admission"))
        });
        if let Some(dir) = dir {
            let path = dir.join(format!("{backend}-{arch}-{:016x}.json", fnv1a(fp)));
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(f) = serde_json::from_str::<CacheFile>(&text) {
                    if f.version == 1 && f.fingerprint == fp {
                        c.file = f;
                    }
                }
            }
            c.path = Some(path);
        }
        c
    }

    fn save(&mut self) {
        if !self.dirty {
            return;
        }
        if let Some(p) = &self.path {
            let write = || -> std::io::Result<()> {
                if let Some(d) = p.parent() {
                    std::fs::create_dir_all(d)?;
                }
                let tmp = p.with_extension(format!("tmp{}", std::process::id()));
                std::fs::write(&tmp, serde_json::to_vec_pretty(&self.file).unwrap_or_default())?;
                std::fs::rename(&tmp, p)?;
                // every rebuild has its own fingerprint: drop caches of binaries nobody has run for two weeks
                let old = std::time::Duration::from_secs(14 * 24 * 3600);
                for e in std::fs::read_dir(p.parent().unwrap_or(std::path::Path::new(".")))?.flatten() {
                    let stale = e
                        .metadata()
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|a| a > old);
                    if stale && e.path().extension().is_some_and(|x| x == "json") {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
                Ok(())
            };
            if let Err(e) = write() {
                eprintln!("admission cache: cannot write {}: {e}", p.display());
            }
        }
        self.dirty = false;
    }
}

/// Outcome of [`admit`].
#[derive(Debug, Default)]
pub struct AdmitSummary {
    pub kernels_checked: usize,
    pub cases_run: usize,
    pub cases_cached: usize,
    pub rejected: Vec<(String, String)>,
}

/// Admit the kernels the plan selects on the plan's exact shapes (clamped, see module docs).
/// Marks them `Admitted`/`Rejected` in the model's registry and re-resolves until the plan only
/// uses admitted kernels; a kernel that fails is replaced by the next candidate, and the load fails
/// with the reasons when none is left.
pub fn admit(model: &mut Model, reference: &dyn Backend, opts: &PlanOptions) -> Result<AdmitSummary> {
    let cand = model.backend.clone();
    let info = cand.info();
    let fp = host_fingerprint(&*cand);
    let mut cache = Cache::open(opts, &fp, &info.backend, &info.arch);
    let ref_kernels = reference.kernels();
    let cand_kernels = cand.kernels();
    let mut sum = AdmitSummary::default();
    let same_backend = reference.name() == cand.name();
    let mut rng = SplitMix64::new(0x1a7a);

    for _round in 0..8 {
        resolve(model)?;
        let pending: BTreeSet<String> = model
            .resolved
            .ops
            .iter()
            .filter(|o| model.registry.admission(&o.kernel) == Some(Admission::Untested))
            .map(|o| o.kernel.clone())
            .collect();
        if pending.is_empty() {
            break;
        }
        for kname in pending {
            let kern = cand_kernels
                .iter()
                .find(|k| k.name == kname)
                .ok_or_else(|| Error::Other(format!("kernel {kname} not in backend list")))?;
            if kern.is_reference && same_backend {
                model.registry.set_admission(&kname, Admission::Admitted); // the oracle itself
                continue;
            }
            let refimpl = ref_kernels
                .iter()
                .find(|k| k.op == kern.op && k.is_reference)
                .ok_or_else(|| Error::NoKernel(format!("reference implementation of {}", kern.op)))?;
            sum.kernels_checked += 1;
            let mut bad: Option<String> = None;
            let work: Vec<PlannedCall> =
                model.resolved.unique_calls.iter().filter(|c| c.kernel == kname).cloned().collect();
            let mut handled_sweep = false;
            for c in work {
                let (host, label) = match case_from_call(&c.call, &CAPS, &mut rng) {
                    Some(x) => x,
                    None => {
                        // an op without an exact-shape case builder: its own sweep decides, once per kernel
                        if handled_sweep {
                            continue;
                        }
                        handled_sweep = true;
                        let key = case_key(&kname, "sweep");
                        let e = match cache.file.entries.get(&key) {
                            Some(e) => {
                                sum.cases_cached += 1;
                                e.clone()
                            }
                            None => {
                                sum.cases_run += 1;
                                let e = sweep_entry(model, reference, &*cand, &kname)?;
                                cache.file.entries.insert(key, e.clone());
                                cache.dirty = true;
                                e
                            }
                        };
                        if !e.pass {
                            bad = Some(format!(
                                "sweep: {}",
                                e.error.unwrap_or_else(|| format!("rel_l2 {:.3e}", e.rel_l2))
                            ));
                        }
                        continue;
                    }
                };
                let key = case_key(&kname, &label);
                let e = match cache.file.entries.get(&key) {
                    Some(e) => {
                        sum.cases_cached += 1;
                        e.clone()
                    }
                    None => {
                        sum.cases_run += 1;
                        let (pass, max_abs, rel_l2, error) = match (|| -> Result<_> {
                            let want = run_host(reference, &refimpl.name, &host)?;
                            let got = run_host(&*cand, &kname, &host)?;
                            Ok(judge(kern, &got, &want))
                        })() {
                            Ok((p, a, r)) => (p, a, r, None),
                            Err(e) => (false, 0.0, 0.0, Some(e.to_string())),
                        };
                        let e = CacheEntry { pass, max_abs, rel_l2, error };
                        cache.file.entries.insert(key, e.clone());
                        cache.dirty = true;
                        e
                    }
                };
                if !e.pass {
                    bad = Some(format!(
                        "{label}: {}",
                        e.error.clone().unwrap_or_else(|| format!(
                            "max_abs {:.3e} rel_l2 {:.3e} (tol {:.1e}/{:.1e}{})",
                            e.max_abs,
                            e.rel_l2,
                            kern.tol.max_abs,
                            kern.tol.rel_l2,
                            kern.tol_f32.map_or(String::new(), |t| format!(", f32 outputs {:.1e}/{:.1e}", t.max_abs, t.rel_l2))
                        ))
                    ));
                }
                if bad.is_some() {
                    break;
                }
            }
            match bad {
                None => model.registry.set_admission(&kname, Admission::Admitted),
                Some(why) => {
                    model.registry.set_admission(&kname, Admission::Rejected);
                    sum.rejected.push((kname.clone(), why));
                }
            }
        }
        cache.save();
    }
    cache.save();
    // everything the final plan selects must be admitted
    resolve(model).map_err(|e| match e {
        Error::NoKernel(m) if !sum.rejected.is_empty() => {
            let why: Vec<String> = sum.rejected.iter().map(|(k, w)| format!("{k}: {w}")).collect();
            Error::NoKernel(format!(
                "{m}\nkernels rejected by admission on this model's shapes:\n  {}",
                why.join("\n  ")
            ))
        }
        e => e,
    })?;
    Ok(sum)
}

fn sweep_entry(model: &Model, reference: &dyn Backend, cand: &dyn Backend, kernel: &str) -> Result<CacheEntry> {
    use crate::admission::{run_admission, AdmissionOpts, SweepConfig};
    let d = &model.ldims;
    let sweep = SweepConfig {
        hidden: d.hidden,
        n_heads: d.n_heads,
        n_kv_heads: d.n_kv_heads,
        head_dim: d.head_dim,
        intermediate: d.intermediate.max(d.hidden),
        vocab: d.vocab,
        rope_theta: d.rope_theta,
        quick: false,
        seed: 1234,
    };
    let rep = run_admission(
        reference,
        cand,
        &AdmissionOpts { sweep: Some(sweep), only_ops: None, only_kernel: Some(kernel.to_string()) },
    )?;
    let s = rep.kernels.iter().find(|k| k.kernel == kernel);
    Ok(match s {
        Some(s) => CacheEntry {
            pass: s.admitted,
            max_abs: s.worst_max_abs,
            rel_l2: s.worst_rel_l2,
            error: (!s.admitted).then(|| {
                rep.rows
                    .iter()
                    .find(|r| !r.pass)
                    .map(|r| format!("{} {}", r.case, r.error.clone().unwrap_or_default()))
                    .unwrap_or_else(|| {
                        "no admission case covers this kernel: add sweep cases for its op in admission::sweep".into()
                    })
            }),
        },
        None => CacheEntry { pass: false, max_abs: 0.0, rel_l2: 0.0, error: Some("kernel not tested".into()) },
    })
}

// ---------------------------------------------------------------------------
// plan description (GET /v1/intarsia/plan, `intarsia plan show`)
// ---------------------------------------------------------------------------

/// The plan as JSON. Contract (consumed by the server and the UI):
///
/// ```text
/// {"model": str, "backend": str, "device": {"name": str, "mem_total": int},
///  "exactness": "fast"|"deterministic"|"exact",
///  "layers": [{"index": int, "kind": "attention"|"linear_attention"|..., "blocks": [str]}],
///  "ops": [{"op": str, "kernel": str, "backend": str, "exactness": str, "admitted": bool,
///           "calls_per_decode_step": int}]}
/// ```
/// Extra keys (additive): `act_dtype`, `kv_dtype`, `scratch_bytes`, `prefill_chunk`, `probed_sizes`,
/// `require_admitted`, `reference_only`. `op` strings equal the keys of `TelemetrySnapshot.ops`.
pub fn plan_json(model: &Model, model_id: &str) -> Value {
    let info = model.backend.info();
    let mut worst = Exactness::Exact;
    for o in &model.resolved.ops {
        worst = match (worst, o.exactness) {
            (Exactness::Fast, _) | (_, Exactness::Fast) => Exactness::Fast,
            (Exactness::Deterministic, _) | (_, Exactness::Deterministic) => Exactness::Deterministic,
            _ => Exactness::Exact,
        };
    }
    let layers: Vec<Value> = model
        .layer_kinds()
        .into_iter()
        .enumerate()
        .map(|(i, blocks)| json!({"index": i, "kind": blocks.first().copied().unwrap_or("unknown"), "blocks": blocks}))
        .collect();
    let ops: Vec<Value> = model
        .resolved
        .ops
        .iter()
        .map(|o| {
            json!({
                "op": o.op.as_str(), "kernel": o.kernel, "backend": o.backend,
                "exactness": o.exactness.as_str(), "admitted": o.admitted,
                "calls_per_decode_step": o.calls_per_decode_step, "scratch_bytes": o.scratch_bytes,
            })
        })
        .collect();
    let tp = if model.tp_world() > 1 {
        let l = &model.ldims;
        let ranks: Vec<Value> = model
            .rank_backends()
            .iter()
            .enumerate()
            .map(|(r, b)| {
                let i = b.info();
                let m = model.rank_model(r);
                json!({
                    "rank": r, "device": i.name, "mem_total": i.total_mem, "free_mem": i.free_mem,
                    "weights_bytes": m.map(|m| m.weights_bytes),
                    "kv_cap_tokens": m.map(|m| m.plan.kv_cap_tokens),
                    "holds_lm_head": r == 0,
                })
            })
            .collect();
        json!({
            "world": model.tp_world(),
            "comm": model.comm_name(),
            "partials": model.tp_partials(),
            "per_rank": {"n_heads": l.n_heads, "n_kv_heads": l.n_kv_heads, "head_dim": l.head_dim, "intermediate": l.intermediate,
                         "hidden": l.hidden, "vocab": l.vocab},
            "global": {"n_heads": model.dims.n_heads, "n_kv_heads": model.dims.n_kv_heads, "intermediate": model.dims.intermediate},
            "lm_head": "replicated on rank 0 only (full vocabulary); other ranks stop after the last layer",
            "all_reduce_calls_per_decode_step": model.resolved.comm_calls_per_decode_step,
            "all_reduce_bytes_per_decode_step": model.resolved.comm_bytes_per_decode_step,
            "ranks": ranks,
        })
    } else {
        Value::Null
    };
    json!({
        "model": model_id,
        "backend": info.backend,
        "device": {"name": info.name, "mem_total": info.total_mem, "arch": info.arch},
        "tp": tp,
        "exactness": worst.as_str(),
        "layers": layers,
        "ops": ops,
        "act_dtype": model.act.name(),
        "kv_dtype": model.kv.name(),
        "scratch_bytes": model.resolved.scratch_bytes,
        "prefill_chunk": model.prefill_chunk(),
        "probed_sizes": model.resolved.sizes,
        "require_admitted": model.registry.require_admitted,
        "reference_only": model.registry.reference_only,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_diff_names_the_differing_ops() {
        let mut by_pos: BTreeMap<usize, Vec<(Op, &str)>> = BTreeMap::new();
        by_pos.insert(0, vec![(Op::Matmul, "mm"), (Op::AttnDecode, "attn_decode_bf16")]);
        by_pos.insert(10, vec![(Op::Matmul, "mm"), (Op::AttnDecode, "attn_decode_bf16")]);
        assert_eq!(position_diff(&by_pos), None);
        by_pos.insert(4096, vec![(Op::Matmul, "mm"), (Op::AttnDecode, "attn_decode_splitkv_bf16")]);
        assert_eq!(
            position_diff(&by_pos).unwrap(),
            "ops differ between pos 0 and pos 4096: attn_decode: attn_decode_bf16 vs attn_decode_splitkv_bf16"
        );
    }

    #[test]
    fn probe_sizes_cover_small_and_boundaries() {
        let s = probe_sizes(256);
        for t in [1, 2, 8, 9, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256] {
            assert!(s.contains(&t), "{t} missing from {s:?}");
        }
        assert!(s.iter().all(|&t| t >= 1 && t <= 256));
        assert_eq!(probe_sizes(1), vec![1]);
        let s4 = probe_sizes(4);
        assert_eq!(s4, vec![1, 2, 3, 4]);
    }

    #[test]
    fn fnv_is_stable() {
        assert_eq!(fnv1a(""), 0xcbf29ce484222325);
        assert_ne!(fnv1a("a"), fnv1a("b"));
    }
}
