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
use super::graph::{build_with, Arg, Buf, Dsv41Graph, Extra, GraphOpts, GraphTp, Stage, WeightShapes};
use crate::backend::{Backend, Buffer, MemKind, TensorRef};
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
    /// Kernel choice per (stage, step name, op, rows class: 1 / 2..=16 / more). Step lists differ between a one-row pass
    /// and a longer one (the MoE route tables), so the index alone does not name a step.
    kernels: HashMap<(usize, String, crate::ops::Op, u8), usize>,
}

/// Per-pass timing (decode profiling): host time spent waiting for Engram records.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassStats {
    pub engram_wait_us: u64,
}

pub struct Dsv41Runtime {
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
}

fn esz(d: DType) -> usize {
    d.size_bytes()
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
        let info = backend.info();
        let reg = Registry::new(backend.name(), &info, backend.kernels());
        let kernel_list = backend.kernels();
        let weight_bytes: u64 = weights.values().map(|t| t.nbytes() as u64).sum();
        // workspace: the largest byte size of every name over the shapes a pass can take
        let mut need: HashMap<String, (usize, DType)> = HashMap::new();
        let probes = [(opts.chunk, 0), (opts.chunk, opts.cap.saturating_sub(opts.chunk)), (1, 0), (1, opts.cap - 1), (2, opts.cap - 2)];
        let mut slot_specs = Vec::new();
        for (t, pos0) in probes {
            let g = build_with(&cfg, &*shapes, GraphOpts { t, pos0, act: opts.act, cap: opts.cap }, opts.tp)?;
            slot_specs = g.slots.clone();
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
        for n in ["empty", "empty_i32", "empty_u8"] {
            put(n, DType::U8, &[0], &[])?;
        }
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
            exec: Mutex::new(Exec { ws, scratch: None, filled: HashMap::new(), kernels: HashMap::new() }),
            slot_specs,
            memory: RuntimeMemory::default(),
            last_pass: Mutex::new(PassStats::default()),
        };
        let state_per_seq = rt.slot_specs.iter().map(|s| s.bytes(rt.opts.cap) as u64).sum();
        rt.memory = RuntimeMemory { weights: weight_bytes, workspace: ws_bytes, constants: cbytes, scratch: 0, state_per_seq };
        Ok(rt)
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
        let mut out = None;
        let mut i = 0;
        while i < ids.len() {
            let n = (ids.len() - i).min(self.opts.chunk);
            let last = i + n == ids.len();
            let r = self.pass(st, &ids[i..i + n], logits && last);
            match r {
                Ok(l) => {
                    st.commit(&ids[i..i + n]);
                    if last {
                        out = l;
                    }
                }
                Err(e) => {
                    // the rows this pass wrote are undefined: the sequence restarts
                    let _ = st.truncate(0, &*self.backend);
                    return Err(e);
                }
            }
            i += n;
        }
        Ok(out)
    }

    /// One pass of `t = ids.len()` rows at position `st.len`.
    fn pass(&self, st: &mut SequenceState, ids: &[u32], want_logits: bool) -> Result<Option<Vec<f32>>> {
        let (t, pos0) = (ids.len(), st.len);
        self.touch();
        let g = build_with(&self.cfg, &*self.shapes, GraphOpts { t, pos0, act: self.opts.act, cap: self.opts.cap }, self.opts.tp)?;
        let mut ex = self.exec.lock().unwrap();
        // host inputs: token ids
        let idb: Vec<u8> = ids.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
        let tid = ex.ws.get("host.token_ids").ok_or_else(|| Error::backend("no token id buffer"))?.clone();
        self.backend.upload(&tid.buf, 0, &idb, 0)?;
        // Engram: hash and issue every Engram layer's rows now
        let mut tickets: HashMap<usize, (usize, super::engram::EngramTicket)> = HashMap::new();
        if let Some(e) = &self.engram {
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
        let mut stats = PassStats::default();
        for (si, stage) in g.stages.iter().enumerate() {
            if let Some(l) = stage.layer {
                if let Some((_li, ticket)) = tickets.remove(&l) {
                    let e = self.engram.as_ref().unwrap();
                    let rb = record_bytes(e.hasher.layout.head_dim);
                    let mut rec = vec![0u8; t * e.hasher.layout.cols() * rb];
                    let w0 = Instant::now();
                    e.tier.wait(ticket, &mut rec)?;
                    stats.engram_wait_us += w0.elapsed().as_micros() as u64;
                    let name = format!("host.engram.{l}.records");
                    let b = ex.ws.get(&name).ok_or_else(|| Error::backend(format!("no buffer {name}")))?.clone();
                    self.backend.upload(&b.buf, 0, &rec, 0)?;
                }
            }
            self.run_stage(&mut ex, si, stage, st, t)?;
        }
        *self.last_pass.lock().unwrap() = stats;
        if !want_logits {
            return Ok(None);
        }
        self.touch();
        let lt = self.view(&ex, &g.logits, st, t)?;
        let mut bytes = vec![0u8; lt.nbytes()];
        self.backend.download(&lt.buf, lt.offset, &mut bytes, 0)?;
        Ok(Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()))
    }

    fn run_stage(&self, ex: &mut Exec, si: usize, stage: &Stage, st: &SequenceState, t: usize) -> Result<()> {
        let mut extras = stage.extras.iter().peekable();
        for (j, step) in stage.steps.iter().enumerate() {
            while let Some((_, e)) = extras.next_if(|(k, _)| *k == j) {
                self.extra(ex, e, st, t)?;
            }
            let inputs: Vec<TensorRef> = step.inputs.iter().map(|a| self.view(ex, a, st, t)).collect::<Result<_>>()?;
            let outputs: Vec<TensorRef> = step.outputs.iter().map(|a| self.view(ex, a, st, t)).collect::<Result<_>>()?;
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
            let mut call = OpCall::new(step.op, inputs, outputs, step.attrs.clone());
            call.check().map_err(|e| Error::model(format!("deepseek_v41 step {}: {e}", step.what)))?;
            let key = (si, step.what.clone(), step.op, if t == 1 { 0 } else if t <= 16 { 1 } else { 2 });
            let ki = match ex.kernels.get(&key) {
                Some(&i) => i,
                None => {
                    let k = self.reg.select(&call).map_err(|e| Error::model(format!("deepseek_v41 step {} ({}): {e}", step.what, step.op.as_str())))?;
                    let i = self
                        .kernel_list
                        .iter()
                        .position(|x| x.name == k.name && x.op == step.op)
                        .ok_or_else(|| Error::backend(format!("deepseek_v41: selected kernel {} is not in the backend's table", k.name)))?;
                    ex.kernels.insert(key, i);
                    i
                }
            };
            let kimpl = &self.kernel_list[ki];
            let kname = kimpl.name.as_str();
            let need = kimpl.scratch_bytes(&ShapeInfo::from_call(&call));
            if need > 0 {
                if ex.scratch.as_ref().is_none_or(|s| s.nbytes() < need) {
                    let bytes = need.next_power_of_two().max(1 << 20);
                    let b = self.backend.alloc(bytes, MemKind::Device)?;
                    ex.scratch = Some(TensorRef::new(b, DType::U8, &[bytes]));
                }
                call.scratch = Some(ex.scratch.as_ref().unwrap().reshaped(&[ex.scratch.as_ref().unwrap().numel()]).rows(0, need));
            }
            self.backend.launch(kname, &call, 0).map_err(|e| Error::backend(format!("deepseek_v41 step {} ({kname}): {e}", step.what)))?;
        }
        for (_, e) in extras {
            self.extra(ex, e, st, t)?;
        }
        Ok(())
    }

    fn extra(&self, ex: &mut Exec, e: &Extra, st: &SequenceState, t: usize) -> Result<()> {
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
    match n.strip_suffix(']').and_then(|x| x.rsplit_once('[')) {
        Some((b, i)) if i.parse::<usize>().is_ok() => b,
        _ => n.strip_suffix(".last_row").unwrap_or(n),
    }
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
    use crate::quant::exl3_moe::{scale_offsets, META_COLS};
    let read = |t: &TensorRef| -> Result<Vec<u8>> {
        let mut v = vec![0u8; t.nbytes()];
        backend.download(&t.buf, t.offset, &mut v, 0)?;
        Ok(v)
    };
    for l in 0..n_layers {
        let p = format!("layers.{l}.ffn");
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
}
