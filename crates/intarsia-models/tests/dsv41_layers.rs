//! Layer-level parity of the DeepSeek-V4.1 graph (`dsv41::graph::build`) and the cpu reference ops against the official
//! code, layers 0, 1 and 2 of the Mia EXL3 checkpoint, on the 18-token fixture (`golden/dsv41/layers_*`, 16 prefill rows
//! then two decode steps). A test-side interpreter walks the graph's `Step`s: `Weight` buffers come from a small
//! safetensors extract of the checkpoint, `Ws` from a map, `Slot` from rings / shared rows allocated from `Dsv41Graph::slots`,
//! `Const` from the config. Routed experts have no reference kernel: their step's output is the fixture's `routed_y` tap
//! (and the Engram gather's output the fixture's decoded rows); everything else runs through the cpu kernels, and the
//! tapped intermediate of each sub-block is compared with the fixture.
//!
//! Gated on files that are not in the repository (they would add 0.5 GB): `INTARSIA_DSV41_WEIGHTS` = the extract made by
//! `tools/dsv41/extract_weights.py` (layers 0-2 without the routed experts + the 18 embedding rows), and
//! `INTARSIA_DSV41_FIXTURES` = a directory with the full `layers_bf16.safetensors` / `layers_fp32.safetensors` (the
//! committed `golden/dsv41/layers_*_ci` subset lacks the layer-0/1 taps). Without them the tests print `skipped`.
#[path = "../../intarsia-backend-cpu/tests/dsv41_common/mod.rs"]
mod dsv41_common;

use dsv41_common::*;
use intarsia_core::dsv41::config::Dsv41Config;
use intarsia_core::dsv41::graph::{build, Arg, Buf, GraphOpts, Step};
use intarsia_core::exec::OutSpec;
use intarsia_core::state::SlotSpec;
use intarsia_core::{DType, HostTensor, Op};
use intarsia_models::dsv41::{detect, HeaderShapes};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const CAP: usize = 4352;

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var_os(k).map(PathBuf::from)
}

struct Setup {
    cfg: Dsv41Config,
    shapes: HeaderShapes,
    weights: St,
    fx: St,
    /// the bf16 fixture (Engram rows are decoded once, and exact in bf16, for both flows)
    fx_bf16: St,
}

fn setup(flow: &str) -> Option<Setup> {
    let (Some(w), Some(f)) = (env_path("INTARSIA_DSV41_WEIGHTS"), env_path("INTARSIA_DSV41_FIXTURES")) else {
        println!("skipped: set INTARSIA_DSV41_WEIGHTS and INTARSIA_DSV41_FIXTURES (see the file header)");
        return None;
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv41/mia_headers");
    let w = if w.is_dir() { w.join("dsv41_l012.safetensors") } else { w };
    let fx = St::open(&f.join(format!("layers_{flow}.safetensors")));
    let fx_bf16 = St::open(&f.join("layers_bf16.safetensors"));
    Some(Setup { cfg: detect(&dir).unwrap().unwrap(), shapes: HeaderShapes::open(&dir).unwrap(), weights: St::open(&w), fx, fx_bf16 })
}

struct Interp<'a> {
    s: &'a Setup,
    act: DType,
    ws: HashMap<String, HostTensor>,
    /// per slot, per plane: `[rows, width]`
    slots: Vec<Vec<HostTensor>>,
    /// replace the output of the routed-experts step / Engram gather by these
    overrides: HashMap<String, HostTensor>,
    ids: Vec<i32>,
    /// emulate the removed fp8 qdq of `o` (evidence test only)
    force_oq: bool,
    /// (what-substring, op) -> compare callback results
    report: Vec<(String, Err3)>,
}

impl<'a> Interp<'a> {
    fn new(s: &'a Setup, act: DType, graph_slots: &[SlotSpec], force_oq: bool) -> Interp<'a> {
        let slots = graph_slots
            .iter()
            .map(|sp| match sp {
                SlotSpec::Ring { rows, width, planes, dtype, .. } => (0..*planes).map(|_| HostTensor::zeros(*dtype, &[*rows, *width])).collect(),
                SlotSpec::Shared { ratio, width, dtype, .. } => vec![HostTensor::zeros(*dtype, &[CAP.div_ceil(*ratio), *width])],
                other => panic!("unexpected slot {other:?}"),
            })
            .collect();
        Interp { s, act, ws: HashMap::new(), slots, overrides: HashMap::new(), ids: vec![], force_oq, report: vec![] }
    }

    fn weight(&self, name: &str) -> HostTensor {
        if let Some(base) = name.strip_suffix(".codebook") {
            let v = if self.s.weights.has(&format!("{base}.mul1")) { 2 } else if self.s.weights.has(&format!("{base}.mcg")) { 1 } else { 0 };
            return HostTensor::from_i32(&[1], &[v]);
        }
        if name == "embed.weight" {
            return self.s.weights.get("embed.weight").host();
        }
        self.s.weights.get(name).host()
    }

    fn konst(&self, name: &str) -> HostTensor {
        match name {
            "rope.base" => f32t(&[32], &self.s.cfg.inv_freq(0)),
            "rope.compressed_yarn" => f32t(&[32], &self.s.cfg.inv_freq(2)),
            "empty" => HostTensor::zeros(self.act, &[0, 512]),
            "empty_i32" => HostTensor::from_i32(&[0], &[]),
            "empty_u8" => HostTensor::zeros(DType::U8, &[0]),
            n => panic!("unknown const {n}"),
        }
    }

    fn read(&self, a: &Arg) -> HostTensor {
        let t = match &a.buf {
            Buf::Weight(n) => self.weight(n),
            Buf::Ws(n) => self.read_ws(n, &a.shape),
            Buf::Const(n) if n == "hc.one_hot_pre_mix" => {
                let rows = a.shape[0];
                f32t(&a.shape, &[1.0, 0.0, 0.0, 0.0].repeat(rows))
            }
            Buf::Const(n) => {
                let c = self.konst(n);
                if c.numel() == 0 { HostTensor::zeros(c.dtype, &a.shape) } else { c }
            }
            Buf::Host(n) => match n.as_str() {
                "token_ids" => HostTensor::from_i32(&[self.ids.len()], &self.ids),
                n => panic!("unknown host input {n}"),
            },
            Buf::Slot { slot, plane, row0 } => {
                let p = &self.slots[*slot][*plane];
                let (rows, w) = (a.shape[0], p.shape[1]);
                let rb = w * p.dtype.size_bytes();
                HostTensor::new(p.dtype, vec![rows, w], p.data[row0 * rb..(row0 + rows) * rb].to_vec()).unwrap()
            }
        };
        if matches!(a.buf, Buf::Weight(_)) || t.numel() == 0 {
            return t;
        }
        assert_eq!(t.numel(), a.shape.iter().product::<usize>(), "{:?}: have {:?}, step wants {:?}", a.buf, t.shape, a.shape);
        t.reshaped(&a.shape)
    }

    /// A workspace tensor; `name[i]` is the i-th outer slice of `name` (per-group EXL3 slices), and a tensor whose slices were
    /// written one by one is assembled from them.
    fn read_ws(&self, n: &str, shape: &[usize]) -> HostTensor {
        if let Some(t) = self.ws.get(n) {
            return t.clone();
        }
        if let Some((base, idx)) = n.strip_suffix(']').and_then(|x| x.rsplit_once('[')) {
            let i: usize = idx.parse().unwrap();
            let b = self.ws.get(base).unwrap_or_else(|| panic!("workspace {base} not produced yet"));
            let per = b.nbytes() / b.shape[0];
            return HostTensor::new(b.dtype, b.shape[1..].to_vec(), b.data[i * per..(i + 1) * per].to_vec()).unwrap();
        }
        if shape.iter().product::<usize>() == 0 {
            // an empty index list / compressed block before the first group completes: the graph reads it, nothing writes it
            return HostTensor::zeros(if n.starts_with("topk_idxs") { DType::I32 } else { self.act }, shape);
        }
        let parts: Vec<&HostTensor> = (0..shape[0]).map_while(|i| self.ws.get(&format!("{n}[{i}]"))).collect();
        assert_eq!(parts.len(), shape[0], "workspace {n} not produced yet");
        let mut data = vec![];
        parts.iter().for_each(|p| data.extend_from_slice(&p.data));
        HostTensor::new(parts[0].dtype, shape.to_vec(), data).unwrap()
    }

    fn write(&mut self, a: &Arg, t: HostTensor) {
        match &a.buf {
            Buf::Ws(n) => {
                self.ws.insert(n.clone(), t);
            }
            Buf::Slot { slot, plane, row0 } => {
                let p = &mut self.slots[*slot][*plane];
                let rb = p.shape[1] * p.dtype.size_bytes();
                assert_eq!(t.nbytes() % rb, 0);
                p.data[row0 * rb..row0 * rb + t.nbytes()].copy_from_slice(&t.data);
            }
            b => panic!("cannot write {b:?}"),
        }
    }

    fn run_step(&mut self, st: &Step) -> Vec<HostTensor> {
        let outs: Vec<HostTensor>;
        if let Some(o) = self.overrides.get(&st.what).cloned() {
            let want = &st.outputs[0];
            outs = vec![HostTensor::from_f32_as(want.dtype, &want.shape, &o.to_f32())];
        } else {
            let inputs: Vec<HostTensor> = st.inputs.iter().map(|a| self.read(a)).collect();
            let outputs: Vec<OutSpec> = st
                .outputs
                .iter()
                .map(|a| match &a.buf {
                    Buf::Slot { .. } => {
                        let init = self.read(a);
                        OutSpec { dtype: init.dtype, shape: init.shape.clone(), init: Some(init) }
                    }
                    _ => OutSpec::new(a.dtype, &a.shape),
                })
                .collect();
            let hc = call(st.op, inputs, outputs, st.attrs.clone());
            let mut o = go(&hc);
            if self.force_oq && st.what.ends_with("attn.rope_inverse(o)") {
                let mut a = intarsia_core::OpAttrs::default();
                (a.dsv.qdq_fmt, a.dsv.qdq_block, a.dsv.qdq_scale) = (intarsia_core::dsv41::ops::QdqFmt::Fp8E4m3, 32, intarsia_core::dsv41::ops::QdqScale::Ue8m0);
                let n = o[0].numel();
                o = go(&call(Op::ActQdq, vec![o.remove(0).reshaped(&[n / 4096, 4096])], vec![OutSpec::new(st.outputs[0].dtype, &[n / 4096, 4096])], a));
            }
            outs = o;
        }
        for (a, t) in st.outputs.iter().zip(&outs) {
            self.write(a, t.clone());
        }
        outs
    }
}

fn graph_for(s: &Setup, act: DType, t: usize, pos0: usize) -> intarsia_core::dsv41::graph::Dsv41Graph {
    build(&s.cfg, &s.shapes, GraphOpts { t, pos0, act, cap: CAP }).unwrap()
}

/// Step name suffix -> fixture taps of its outputs (tap names without the `{flow}/L{l}/{step}/` prefix).
fn taps_of(what: &str, op: Op) -> Vec<&'static str> {
    let w = what.rsplit_once('.').map_or(what, |_| what);
    let tail = |n: &str| w.ends_with(n);
    match op {
        Op::HcMixes if tail("hc_attn") => vec!["hc_attn.pre", "hc_attn.post", "hc_attn.comb"],
        Op::HcMixes if tail("hc_ffn") => vec!["hc_ffn.pre", "hc_ffn.post", "hc_ffn.comb"],
        Op::HcPre if tail("hc_pre(attn)") => vec!["hc_pre_attn"],
        Op::HcPre if tail("hc_pre(ffn)") => vec!["hc_pre_ffn"],
        Op::RmsNorm if tail("attn_norm.weight") => vec!["attn_norm"],
        Op::RmsNorm if tail("ffn_norm.weight") => vec!["ffn_norm"],
        Op::RopeTail if tail("attn.rope(q)") => vec!["attn.q"],
        Op::ActQdq if tail("attn.act_quant(kv)") => vec!["attn.kv_window"],
        Op::AttnSparse => vec!["attn.o_sparse"],
        Op::RopeTail if tail("attn.rope_inverse(o)") => vec!["attn.o_unrot"],
        Op::Transpose01 if tail("wo_a(token-major)") => vec!["attn.o_lora"],
        Op::MatmulExl3 if tail("attn.wo_b") => vec!["attn.out"],
        Op::HcPost if tail("hc_post(attn)") => vec!["hc_post_attn"],
        Op::MatmulExl3 if tail("shared_experts.w2") => vec!["ffn.shared"],
        Op::Add if tail("ffn.combine") => vec!["ffn.out"],
        Op::HcPost if tail("hc_post(ffn)") => vec!["out.h"],
        Op::MatmulExl3 if tail("engram.wkv") => vec!["engram.kv"],
        Op::EngramMerge => vec!["engram.out"],
        Op::ActQdq if tail("attn.act_quant(latent)") => vec!["attn.latent"],
        Op::ActQdq if tail("attn.act_quant(index_k)") => vec!["indexer.k"],
        _ => vec![],
    }
}

/// Run layer `l` for one chunk with the fixture's layer input; returns the layer output and fills `ip.report`.
fn run_layer(s: &Setup, ip: &mut Interp, g: &intarsia_core::dsv41::graph::Dsv41Graph, flow: &str, l: usize, step: &str, t: usize) -> Vec<f32> {
    let key = |n: &str| format!("{flow}/L{l}/{step}/{n}");
    let stage = g.stages.iter().find(|x| x.layer == Some(l)).unwrap();
    // for an Engram layer the fixture's `in.h` is the Engram output (the generator taps the block after the Engram), the
    // layer's own input is `engram.in`
    let in_h = if s.fx.has(&key("engram.in")) { s.fx.get(&key("engram.in")).host() } else { s.fx.get(&key("in.h")).host() };
    for n in ["x0", "x1"] {
        ip.ws.insert(n.into(), in_h.clone().reshaped(&[t, 4, 5120]));
    }
    if s.fx.has(&key("in.pre_mix")) {
        ip.ws.insert("hc_ffn.pre".into(), s.fx.get(&key("in.pre_mix")).host().reshaped(&[t, 4]));
    }
    ip.overrides.clear();
    if s.fx.has(&key("ffn.routed_y")) {
        ip.overrides.insert(format!("layers.{l}.ffn.experts"), s.fx.get(&key("ffn.routed_y")).host());
    }
    let rows_key = format!("bf16/L{l}/{step}/engram.rows");
    if s.fx_bf16.has(&rows_key) {
        ip.overrides.insert(format!("layers.{l}.engram.embed"), s.fx_bf16.get(&rows_key).host());
    }
    ip.report.clear();
    let mut last = vec![];
    for st in &stage.steps {
        let outs = ip.run_step(st);
        last = outs[0].to_f32();
        if st.op == Op::MoeRoute {
            let (w, ids) = (outs[0].to_f32(), outs[1].to_i32().unwrap());
            if s.fx.has(&key("ffn.gate_idx")) {
                let (ww, wi) = (s.fx.get(&key("ffn.gate_w")).f32(), s.fx.get(&key("ffn.gate_idx")).i32());
                let k = ids.len() / t;
                let mut bad = 0;
                for r in 0..t {
                    let (mut a, mut b) = (ids[r * k..(r + 1) * k].to_vec(), wi[r * k..(r + 1) * k].to_vec());
                    a.sort();
                    b.sort();
                    bad += usize::from(a != b);
                }
                let e = err(&w, &ww);
                println!("LAYER {flow} L{l} {step} {:<28} expert sets differing {bad}/{t}; weights max_abs {:.2e} (of max {:.2e})", "MoeRoute", e.max_abs, e.scale);
            }
        }
        for (i, tap) in taps_of(&st.what, st.op).into_iter().enumerate() {
            if !s.fx.has(&key(tap)) {
                continue;
            }
            let want = s.fx.get(&key(tap));
            let got = outs[i].to_f32();
            if got.len() != want.numel() {
                continue;
            }
            let e = err(&got, &want.f32());
            println!("LAYER {flow} L{l} {step} {:<28} vs {:<18} max_abs/max|ref| {:.3e}  rel_rms {:.3e}  (max|ref| {:.2e})", st.what.trim_start_matches(&format!("layers.{l}.")), tap, e.max_rel_max, e.rel_rms, e.scale);
            ip.report.push((tap.to_string(), e));
        }
    }
    last
}

fn layers_flow(flow: &str, act: DType, tols: [f64; 3]) {
    let Some(s) = setup(flow) else { return };
    let t = 16;
    // embedding + HcExpand (stage `embed`) against layer 0's input
    let g = graph_for(&s, act, t, 0);
    let mut ip = Interp::new(&s, act, &g.slots, false);
    ip.ids = (0..t as i32).collect();
    let embed = g.stages.iter().find(|x| x.name == "embed").unwrap();
    for st in &embed.steps {
        ip.run_step(st);
    }
    let emb = ip.ws["x0"].to_f32();
    check(&format!("{flow} embed + HcExpand vs L0 in.h"), &emb, &s.fx.get(&format!("{flow}/L0/s0/in.h")).f32(), 1e-6);
    // each layer starts from the fixture's own input, so an error in one layer cannot hide or shift another's; the three
    // chunks (16-row prefill at 0, decode steps at 16 and 17) run in order on one set of slots per layer
    for l in 0..3 {
        let mut ip = Interp::new(&s, act, &g.slots, false);
        for (step, t, pos0) in [("s0", 16usize, 0usize), ("s1", 1, 16), ("s2", 1, 17)] {
            let g = graph_for(&s, act, t, pos0);
            let out = run_layer(&s, &mut ip, &g, flow, l, step, t);
            let name = if step == "s0" { "prefill 16".to_string() } else { format!("decode at position {pos0}") };
            check(&format!("{flow} layer {l} out.h ({name})"), &out, &s.fx.get(&format!("{flow}/L{l}/{step}/out.h")).f32(), tols[l]);
        }
    }
}

/// bf16 hidden state: every sub-block rounds to bf16, the official kernels round differently in places (attention
/// probabilities, split accumulations), so per-layer outputs agree to a few bf16 ulps (measured 2-5e-3 of max|ref|).
#[test]
fn layers_0_to_2_bf16_match_official_taps() {
    layers_flow("bf16", DType::Bf16, [1e-2, 1e-2, 1e-2]);
}

#[test]
fn layers_0_to_2_fp32_match_official_taps() {
    layers_flow("fp32", DType::F32, [1e-4, 1e-4, 1e-4]);
}

/// The official code quantises the attention output to fp8 before `wo_a` only for the FP8 base checkpoint; the Mia EXL3
/// checkpoint's reference engine (TensorFold) and the fixtures do not. The graph therefore has no `act_quant(o)` step for
/// EXL3 (it used to insert it for every checkpoint, which made fp32 layer-0 `o_lora` disagree with the fixture by 2.5e-2
/// instead of 2.5e-6); this test pins the graph and records the size of the divergence the step would cause.
#[test]
fn exl3_graph_has_no_fp8_qdq_of_the_attention_output() {
    let Some(s) = setup("fp32") else { return };
    let act = DType::F32;
    let g = graph_for(&s, act, 16, 0);
    for st in &g.stages {
        assert!(!st.steps.iter().any(|x| x.what.ends_with("act_quant(o)")), "{}: fp8 qdq of o in an EXL3 graph", st.name);
    }
    let mut worst = [0f64; 2];
    for (i, force) in [false, true].into_iter().enumerate() {
        let mut ip = Interp::new(&s, act, &g.slots, force);
        run_layer(&s, &mut ip, &g, "fp32", 0, "s0", 16);
        worst[i] = ip.report.iter().find(|(t, _)| t == "attn.o_lora").unwrap().1.max_rel_max;
    }
    println!("PARITY o-quant: fp32 layer-0 attn.o_lora max_abs/max|ref| = {:.2e} as built, {:.2e} with an fp8 qdq of `o` inserted", worst[0], worst[1]);
    assert!(worst[0] < 1e-5 && worst[1] > 100.0 * worst[0]);
}

/// Chunked == whole through the real graph and the real weights: layer 2 (window ring, ratio-2 compressor with a partial
/// group, index keys, indexer, sparse attention) on the 16 prefill rows as one chunk, as uneven chunks and as 16 one-token
/// steps, hidden state and the pooled / window state must be bit-identical (the cpu EXL3 kernel and every op are row
/// independent). The official code is not (bf16 matmul batch shape, `attn4k_L2_bf16_report.json`), this engine must be.
#[test]
fn layer2_chunked_equals_whole_through_the_graph_bit_for_bit() {
    let Some(s) = setup("bf16") else { return };
    let (flow, act, l) = ("bf16", DType::Bf16, 2usize);
    let key = |n: &str| format!("{flow}/L{l}/s0/{n}");
    let (in_h, pre, routed) = (s.fx.get(&key("in.h")).f32(), s.fx.get(&key("in.pre_mix")).f32(), s.fx.get(&key("ffn.routed_y")).f32());
    let run = |sched: &[usize]| -> (Vec<f32>, Vec<Vec<u8>>) {
        let g0 = graph_for(&s, act, 16, 0);
        let mut ip = Interp::new(&s, act, &g0.slots, false);
        let (mut pos, mut out) = (0usize, vec![]);
        for &t in sched {
            let g = graph_for(&s, act, t, pos);
            let stage = g.stages.iter().find(|x| x.layer == Some(l)).unwrap();
            let h = HostTensor::from_f32_as(act, &[t, 4, 5120], &in_h[pos * 4 * 5120..(pos + t) * 4 * 5120]);
            ip.ws.insert("x0".into(), h.clone());
            ip.ws.insert("x1".into(), h);
            ip.ws.insert("hc_ffn.pre".into(), f32t(&[t, 4], &pre[pos * 4..(pos + t) * 4]));
            ip.overrides.clear();
            ip.overrides.insert(format!("layers.{l}.ffn.experts"), f32t(&[t, 5120], &routed[pos * 5120..(pos + t) * 5120]));
            let mut last = vec![];
            for st in &stage.steps {
                last = ip.run_step(st)[0].to_f32();
            }
            out.extend(last);
            pos += t;
        }
        (out, ip.slots.iter().map(|sl| sl.iter().flat_map(|p| p.data.clone()).collect()).collect())
    };
    let (whole, ws) = run(&[16]);
    // the shared compressed-KV / index-key rows past the filled ones and rings are compared as stored
    let layer_slots: Vec<usize> = (0..ws.len()).filter(|i| !ws[*i].iter().all(|b| *b == 0)).collect();
    assert!(layer_slots.len() >= 4, "layer 2 fills its window, compressor tail, compressed KV and index key slots");
    for sched in [vec![8usize, 8], vec![5, 7, 4], vec![1; 16], vec![3, 1, 1, 9, 2], vec![15, 1]] {
        let (out, st) = run(&sched);
        assert!(out == whole, "schedule {sched:?}: layer output differs from the single chunk");
        assert!(st == ws, "schedule {sched:?}: state slots differ from the single chunk");
        println!("PARITY layer 2 chunked==whole through the graph, schedule {sched:?}: bit-identical output and slots");
    }
}

/// Offline (no weights): the Mia graph's real ring / shared slots in a `SequenceState` accept a rollback of up to the ring
/// margin (16 tokens: speculative verify) and refuse a longer one; the chunked/rollback exactness of the ops themselves is
/// `intarsia-backend-cpu/tests/dsv41_chunked.rs`.
#[test]
fn sequence_state_truncate_follows_the_graph_ring_margin() {
    use intarsia_core::state::SequenceState;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv41/mia_headers");
    let (cfg, shapes) = (detect(&dir).unwrap().unwrap(), HeaderShapes::open(&dir).unwrap());
    let g = build(&cfg, &shapes, GraphOpts { t: 1, pos0: 0, act: DType::Bf16, cap: 512 }).unwrap();
    let b = intarsia_backend_cpu::CpuBackend::new();
    let fresh = |n: usize| {
        let mut st = SequenceState::new(&b, &g.slots, 512).unwrap();
        st.commit(&(0..n as u32).collect::<Vec<_>>());
        st
    };
    let mut st = fresh(300);
    assert_eq!(st.ring_margin(), Some(16));
    assert_eq!(st.restorable_len(284), 284);
    assert_eq!(st.restorable_len(283), 0, "beyond the margin only the empty state is restorable");
    st.truncate(284, &b).unwrap();
    assert_eq!(st.len, 284);
    let mut st = fresh(300);
    assert!(st.truncate(283, &b).is_err(), "a rollback of 17 tokens must be refused");
    st.truncate(0, &b).unwrap();
}
