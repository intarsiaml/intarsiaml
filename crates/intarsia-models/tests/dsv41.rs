//! DeepSeek-V4.1 skeleton (lane 1a): the recipe matches, loading refuses with a pointer to `plan show`, and the graph of
//! the real Mia EXL3 checkpoint (header-only fixture: every tensor's dtype and shape, routed experts 1..382 omitted) records
//! as checked `OpCall`s on the cpu backend with the expected op sequence, state slots and MTP blocks.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::dsv41::config::Dsv41Config;
use intarsia_core::dsv41::graph::WeightShapes;
use intarsia_core::dsv41::graph::{build, build_draft, build_draft_dpos, build_pass, build_replay, build_with, Buf, Dsv41Graph, GraphOpts, GraphTp, PassKind, Replay, ReplayPart};
use intarsia_core::state::{SharedKind, SlotSpec};
use intarsia_core::{DType, Op};
use intarsia_models::dsv41::{detect, plan, HeaderShapes};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv41")
}

fn mia() -> (Dsv41Config, HeaderShapes) {
    let dir = fixture().join("mia_headers");
    (detect(&dir).unwrap().expect("a deepseek_v41 config"), HeaderShapes::open(&dir).unwrap())
}

#[test]
fn recipe_matches_and_loading_refuses_until_the_kernels_land() {
    let cfg: Value = serde_json::from_str(include_str!("fixtures/dsv41/mia-config.json")).unwrap();
    let (src, r) = intarsia_models::select_recipe(&cfg, &[]).unwrap();
    assert_eq!((src.as_str(), r.model.family.as_str()), ("builtin:deepseek_v41", "deepseek_v41"));
    let e = intarsia_models::read_model_spec(&fixture().join("mia_headers"), &Default::default()).unwrap_err().to_string();
    assert!(e.contains("does not execute yet") && e.contains("plan show"), "{e}");
}

/// Op sequence of one layer of the decode-step graph.
fn layer_ops(g: &intarsia_core::dsv41::graph::Dsv41Graph, l: usize) -> Vec<Op> {
    g.stages.iter().find(|s| s.layer == Some(l)).unwrap().steps.iter().map(|s| s.op).collect()
}

#[test]
fn mia_graph_records_every_layer_with_the_official_structure() {
    let (c, w) = mia();
    let g = build(&c, &w, GraphOpts { t: 1, pos0: 4095, act: DType::Bf16, cap: 131_072 }).unwrap();
    assert_eq!(g.residual_streams, 4);
    assert_eq!(g.stages.len(), 42, "embed + 40 layers + head");
    for st in &g.stages {
        for s in &st.steps {
            s.phantom_call().unwrap();
            // no step writes a buffer it reads (kernels may assume outputs never alias inputs)
            for o in &s.outputs {
                assert!(s.inputs.iter().all(|i| i.buf != o.buf), "{}: {:?} is read and written", s.what, o.buf);
            }
        }
    }
    let count = |l: usize, op: Op| layer_ops(&g, l).iter().filter(|&&o| o == op).count();
    // every layer: two hyper-connection sub-blocks, MQA with sink over the window, 8 wo_a group slices, routed + shared MoE.
    // At decode each sub-block is one HcSublayer that folds the previous HcPost; the FFN's HcPost stays a step of its own
    // where the next layer reads the streams first (Engram on 1 and 14, the DSpark tap on 37-39, the head after 39).
    for l in 0..40 {
        let tap = usize::from(l >= 37);
        let post = usize::from([0, 13, 36, 37, 38, 39].contains(&l));
        assert_eq!((count(l, Op::HcSublayer), count(l, Op::HcMixes), count(l, Op::HcPre), count(l, Op::HcPost)), (2, 0, tap, post), "layer {l}");
        assert_eq!((count(l, Op::WindowKv), count(l, Op::AttnSparse), count(l, Op::MoeRoute), count(l, Op::MoeExperts)), (1, 1, 1, 1), "layer {l}");
        assert_eq!(count(l, Op::Transpose01), 2, "layer {l}");
    }
    // compressors on the KV sources only (layer 20's ratio 1 is a plain projection), indexers on the index sources,
    // candidate blocks on layer 20, Engram on 1 and 14
    let with = |op: Op| (0..40).filter(|&l| count(l, op) > 0).collect::<Vec<_>>();
    assert_eq!(with(Op::CompressPool), [2, 8, 14]);
    assert_eq!(with(Op::IndexerTopk), [2, 8, 14, 20, 24, 28, 32, 36]);
    assert_eq!(with(Op::IndexerBlocks), [20]);
    assert_eq!(with(Op::EngramMerge), [1, 14]);
    // the DSpark tap (mean over streams of the attention input) on the MTP target layers
    let taps: Vec<usize> = (0..40).filter(|&l| g.stages[l + 1].steps.iter().any(|s| s.what.ends_with("dspark_tap"))).collect();
    assert_eq!(taps, [37, 38, 39]);
    let steps: Vec<usize> = (0..40).map(|l| layer_ops(&g, l).len()).collect();
    assert_eq!((steps[0], steps[1], steps[2], steps[3], steps[14], steps[20], steps[24], steps[39]), (28, 30, 42, 27, 45, 41, 32, 29),
        "6 fewer steps per layer than the separate HcMixes / HcPre / RmsNorm / HcPost (5 where the FFN HcPost stays)");
    // a prefill chunk (T > 16) keeps both HcPosts separate and emits the HcSublayer without the fold
    let pf = build(&c, &w, GraphOpts { t: 37, pos0: 0, act: DType::Bf16, cap: 131_072 }).unwrap();
    for l in [0usize, 5, 39] {
        let ops = layer_ops(&pf, l);
        let n = |op: Op| ops.iter().filter(|&&o| o == op).count();
        assert_eq!((n(Op::HcSublayer), n(Op::HcPost), n(Op::HcMixes)), (2, 2, 0), "prefill layer {l}");
    }
    assert!(pf.stages.iter().flat_map(|s| &s.steps).filter(|s| s.op == Op::HcSublayer).all(|s| s.inputs[6].shape == [0]));

    // the indexers on 24..36 mask with layer 20's candidates; the others pass an empty mask
    let idx_cand = |l: usize| g.stages[l + 1].steps.iter().find(|s| s.op == Op::IndexerTopk).unwrap().inputs[3].shape.clone();
    assert_eq!(idx_cand(20), [0]);
    assert_eq!(idx_cand(24), [1, (4096usize).div_ceil(8)]);
    // non-source layers read their source's compressed KV slot and top-k indices
    let attn = |l: usize| g.stages[l + 1].steps.iter().find(|s| s.op == Op::AttnSparse).unwrap().inputs.clone();
    assert_eq!(attn(5)[2].shape, [2048, 512], "ratio 2 at position 4095: 2048 compressed rows");
    assert_eq!(attn(5)[3].buf, Buf::Ws("topk_idxs.2".into()));
    assert_eq!(attn(25)[3].buf, Buf::Ws("topk_idxs.24".into()));
    assert_eq!(attn(0)[2].shape, [0, 512], "window-only layers attend no compressed rows");
    assert_eq!(attn(25)[2].buf, attn(20)[2].buf, "one shared slot for every ratio-1 layer");

    // routed + shared experts: ONE MoeExperts in its EXL3 encoding (i16 arena, scales [E + 1, 3H + 3I], meta [E + 1, 8],
    // codebook, limit), the shared expert as expert 384 picked by a 7th slot of weight 1
    let ex = |l: usize| g.stages[l + 1].steps.iter().find(|s| s.op == Op::MoeExperts).unwrap().inputs.clone();
    let shapes: Vec<(DType, Vec<usize>)> = ex(0)[3..].iter().map(|a| (a.dtype, a.shape.clone())).collect();
    let shared = |l: usize| -> usize { ["w1", "w3", "w2"].iter().map(|m| w.0[&format!("layers.{l}.ffn.shared_experts.{m}.trellis")].1.iter().product::<usize>()).sum() };
    // layer 0's experts are 3 bits: gate / up [320, 144, 48] and down [144, 320, 48] words, 384 experts
    assert_eq!(shapes[0], (DType::I16, vec![384 * 3 * 320 * 144 * 48 + shared(0)]));
    assert_eq!(shapes[1..], [(DType::F16, vec![385, 3 * 5120 + 3 * 2304]), (DType::I32, vec![385, 8]), (DType::I32, vec![1]), (DType::F32, vec![1])]);
    assert_eq!(ex(20)[3].shape, [384 * 3 * 320 * 144 * 32 + shared(20)], "layer 20's experts are 2 bits");
    assert_eq!((ex(0)[1].shape.clone(), ex(0)[2].shape.clone()), (vec![1, 7], vec![1, 7]), "6 routed slots + the shared one");

    // state: 40 window rings, per KV source a compressor tail (ratio 2), the compressed KV and the index keys
    assert_eq!(g.slots.len(), 40 + 3 * 3 + 2);
    // every ring can roll back 16 tokens (speculative verify): window 128 + 16 rows, ratio-2 tail 18 rows
    for s in &g.slots {
        if let SlotSpec::Ring { rows, margin, kind, .. } = s {
            assert_eq!(*margin, 16);
            assert_eq!(*rows, if *kind == intarsia_core::state::RingKind::WindowKv { 144 } else { 18 });
        }
    }
    let shared: Vec<(SharedKind, usize, Vec<usize>)> = g
        .slots
        .iter()
        .filter_map(|s| match s {
            SlotSpec::Shared { kind, writer, readers, .. } => Some((*kind, *writer, readers.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(shared.iter().filter(|s| s.0 == SharedKind::CompressedKv).map(|s| s.1).collect::<Vec<_>>(), [2, 8, 14, 20]);
    assert_eq!(shared.last().unwrap(), &(SharedKind::IndexK, 20, vec![20, 24, 28, 32, 36]));
    assert_eq!(shared[2].2, (8..14).collect::<Vec<_>>());

    // the DSpark / MTP blocks are parsed (128 experts, top 3), not executed
    assert_eq!(g.mtp.len(), 3);
    assert!(g.mtp.iter().all(|m| m.experts == 128 && m.top_k == 3));

    // a prefill chunk from 0 that ends inside a ratio-2 group still records (the group tail waits in the ring)
    let p = build(&c, &w, GraphOpts { t: 37, pos0: 0, act: DType::Bf16, cap: 131_072 }).unwrap();
    for st in &p.stages {
        for s in &st.steps {
            s.phantom_call().unwrap();
        }
    }
    let pool = p.stages[3].steps.iter().find(|s| s.op == Op::CompressPool).unwrap();
    assert_eq!(pool.outputs[0].shape, [18, 512]);
}

#[test]
fn plan_show_lists_every_new_op_and_resolves_cpu_kernels() {
    let dir = fixture().join("mia_headers");
    let (c, _) = mia();
    let p = plan(&dir, &c, &CpuBackend::new(), 4096).unwrap();
    assert_eq!(p["family"], "deepseek_v41");
    assert_eq!(p["residual_streams"], 4);
    let ops: BTreeMap<String, (String, u64)> = p["ops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| (o["op"].as_str().unwrap().to_string(), (o["kernel"].as_str().unwrap().to_string(), o["calls_per_decode_step"].as_u64().unwrap())))
        .collect();
    for op in intarsia_core::dsv41::ops::OPS {
        // the base checkpoint's ops (fp8 block-scaled linears, a dense grouped wo_a) do not occur in the EXL3 graph
        // and the clamp runs inside the fused routed + shared MoeExperts (no separate shared-expert SiluGluClamp)
        if matches!(op, Op::MatmulFp8Block | Op::GroupedMatmul | Op::SiluGluClamp) {
            assert!(!ops.contains_key(op.as_str()));
            continue;
        }
        // the fused HcSublayer (62) replaces HcMixes + HcPre + RmsNorm (+ the previous HcPost at decode): HcMixes no
        // longer occurs on its own (docs/design/dsv41-ops.md "Wiring the fused HcSublayer")
        if op == Op::HcMixes {
            assert!(!ops.contains_key(op.as_str()));
            continue;
        }
        let (k, _) = &ops[op.as_str()];
        assert_eq!(k, &format!("cpu_f32_{}", op.as_str()));
    }
    assert_eq!(ops["matmul_exl3"].1, 502, "622 before the shared expert joined the routed launch");
    assert_eq!(ops["hc_pre"].1, 4, "the collapse before the head and three DSpark taps (84 before the fused HcSublayer)");
    assert_eq!(ops["hc_sublayer"].1, 80, "two per layer");
    assert_eq!(ops["hc_post"].1, 6, "the FFN HcPost before Engram layers 1 and 14, the taps on 37-39 and the head");
    let text = intarsia_core::dsv41::plan::render(&p);
    // the routed experts are the EXL3 encoding of the existing MoeExperts, resolved to its cpu kernel; no stub on Mia
    assert_eq!(ops["moe_experts"], ("cpu_f32_moe_experts".to_string(), 40));
    assert!(text.contains("residual streams 4") && text.contains("compressed_kv") && !text.contains("stub (lane 1b)"), "{text}");
}

/// The trimmed index and quantisation config committed beside the fixture agree with its headers: same tensor names,
/// and the EXL3 bits per weight Mia declares are the trellis width (`16 * bits` words per tile).
#[test]
fn trimmed_index_and_quant_config_match_the_headers() {
    let (_, w) = mia();
    let idx: Value = serde_json::from_str(include_str!("fixtures/dsv41/mia-index.trimmed.json")).unwrap();
    let names = idx["weight_map"].as_object().unwrap();
    assert!(names.len() > 500);
    for n in names.keys() {
        assert!(w.0.contains_key(n), "{n} in the index but not in the headers");
    }
    let q: Value = serde_json::from_str(include_str!("fixtures/dsv41/mia-quantization_config.trimmed.json")).unwrap();
    let mut checked = 0;
    for (m, v) in q["tensor_storage"].as_object().unwrap() {
        let Some(bits) = v["bits_per_weight"].as_f64() else { continue };
        let (dt, s) = w.0.get(&format!("{m}.trellis")).unwrap_or_else(|| panic!("{m}.trellis"));
        assert_eq!((*dt, s[2] as f64), (DType::I16, 16.0 * bits), "{m}");
        checked += 1;
    }
    assert!(checked > 100, "{checked}");
    // the base checkpoint names the same modules as fp8 weights with E8M0 scales
    let base: Value = serde_json::from_str(include_str!("fixtures/dsv41/base-index.trimmed.json")).unwrap();
    assert!(base["weight_map"].get("layers.2.attn.wq_a.scale").is_some() && base["weight_map"].get("layers.2.attn.indexer.wk.scale").is_none());
}

/// Every step of the Mia graph (decode at a long position, a 16-row and a 512-row chunk) has a CUDA kernel on sm_121: the
/// CUDA registry is data, so this runs without a GPU and catches a missing kernel before a Spark run does.
#[test]
fn every_step_resolves_to_a_cuda_kernel() {
    let (c, w) = mia();
    let info = intarsia_core::DeviceInfo {
        name: "registry only".into(),
        backend: "cuda".into(),
        arch: intarsia_backend_cuda::kernels::ARCH.into(),
        total_mem: 128 << 30,
        free_mem: 100 << 30,
        unified: true,
        bandwidth_gbs: None,
    };
    let reg = intarsia_core::registry::Registry::new("cuda", &info, intarsia_backend_cuda::kernels::all());
    let mut missing = BTreeMap::new();
    // and one rank of TP=2 on a rank cache's shapes when given (`INTARSIA_DSV41_CACHE_R0`: a `--world 2` rank cache)
    let r0 = std::env::var_os("INTARSIA_DSV41_CACHE_R0").map(PathBuf::from);
    let shapes_r0 = r0.as_ref().map(|d| (HeaderShapes::open(d).unwrap(), intarsia_models::dsv41_cache::open_manifest(d).unwrap()));
    let mut graphs = vec![];
    for (t, pos0) in [(1, 70_000), (16, 0), (16, 4096), (512, 0), (512, 8192)] {
        graphs.push((t, build(&c, &w, GraphOpts { t, pos0, act: DType::Bf16, cap: 131_072 }).unwrap()));
        if let Some((ws, m)) = &shapes_r0 {
            let tp = intarsia_core::dsv41::graph::GraphTp { world: m.world, layers: m.layers, first: m.first_layer, rope_inplace: false };
            graphs.push((t, build_with(&c, ws, GraphOpts { t, pos0, act: DType::Bf16, cap: 131_072 }, tp).unwrap()));
        }
    }
    for (t, g) in graphs {
        for st in g.stages.iter().flat_map(|s| s.steps.iter()) {
            let call = st.phantom_call().unwrap();
            if let Err(e) = reg.select(&call) {
                missing.entry(format!("{} t={t}", st.op.as_str())).or_insert_with(|| format!("{}: {e}", st.what));
            }
        }
    }
    assert!(missing.is_empty(), "steps without a CUDA kernel:\n{}", missing.iter().map(|(k, v)| format!("  {k}: {v}")).collect::<Vec<_>>().join("\n"));
}

/// Which attention buffer a step touches: `(reads, writes)` of the workspace buffer `name` (inputs and outputs), counting the
/// `Extra` host actions of the stage at their positions as touches too (none of them may name `q` or `o`).
fn touches(s: &intarsia_core::dsv41::graph::Step, name: &str) -> (bool, bool) {
    let b = Buf::Ws(name.into());
    (s.inputs.iter().any(|x| x.buf == b), s.outputs.iter().any(|x| x.buf == b))
}

fn extra_names(e: &intarsia_core::dsv41::graph::Extra) -> Vec<Buf> {
    use intarsia_core::dsv41::graph::Extra;
    match e {
        Extra::AllReduce { x } => vec![x.buf.clone()],
        Extra::AllReduceInto { partial, out } => vec![partial.buf.clone(), out.buf.clone()],
        Extra::AllGather { send, recv } => vec![send.buf.clone(), recv.buf.clone()],
        Extra::Fill { buf, .. } => vec![Buf::Ws(buf.clone())],
    }
}

/// The base graph (no flag): `ws:q` is written by `wq_b` and read by `rope(q)` only, `ws:o` is written by the attention and read by
/// `rope_inverse(o)` only, and `ws:q_rope` / `ws:o_derot` carry the rotated rows to their one consumer. Returns the number of ropes.
fn check_base_buffers(base: &Dsv41Graph) -> usize {
    let mut ropes = 0;
    for st in &base.stages {
        for (_, e) in &st.extras {
            for b in extra_names(e) {
                assert!(!matches!(&b, Buf::Ws(n) if ["q", "o", "q_rope", "o_derot"].contains(&n.as_str())), "{}: an extra names {b:?}", st.name);
            }
        }
        for s in &st.steps {
            let ok = |name: &str, rd: &dyn Fn() -> bool, wr: &dyn Fn() -> bool| {
                let (r, w) = touches(s, name);
                assert!(!r || rd(), "{}: {} reads the unrotated ws:{name}", st.name, s.what);
                assert!(!w || wr(), "{}: {} writes ws:{name}", st.name, s.what);
            };
            ok("q", &|| s.what.ends_with("rope(q)"), &|| s.what.ends_with(".wq_b"));
            ok("o", &|| s.what.ends_with("rope_inverse(o)"), &|| s.op == Op::AttnSparse);
            ok("q_rope", &|| s.op == Op::AttnSparse, &|| s.what.ends_with("rope(q)"));
            ok("o_derot", &|| s.what.contains("act_quant(o)") || s.what.contains("wo_a"), &|| s.what.ends_with("rope_inverse(o)"));
            ropes += usize::from(s.what.ends_with("rope(q)") || s.what.ends_with("rope_inverse(o)"));
        }
    }
    ropes
}

/// The graph with `GraphTp::rope_inplace`, in execution order over ALL stages: each of `ws:q` and `ws:o` is `Raw` after its producer
/// and `Rotated` after the in-place rope; the rope must find it `Raw`, and every other reader anywhere later in the graph (any
/// stage) must find it `Rotated` (so a reader of the unrotated rows, in a later step or a later layer, fails). Returns the number
/// of in-place `q` and `o` ropes.
fn check_no_unrotated_read(ip: &Dsv41Graph) -> (usize, usize) {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum St {
        Unwritten,
        Raw,
        Rotated,
    }
    let (mut q, mut o) = (St::Unwritten, St::Unwritten);
    let (mut n_q, mut n_o) = (0, 0);
    for st in &ip.stages {
        for (_, e) in &st.extras {
            for b in extra_names(e) {
                assert!(!matches!(&b, Buf::Ws(n) if ["q", "o", "q_rope", "o_derot"].contains(&n.as_str())), "{}: an extra names {b:?}", st.name);
            }
        }
        for s in &st.steps {
            for n in ["q_rope", "o_derot"] {
                assert_eq!(touches(s, n), (false, false), "{}: {} still uses ws:{n}", st.name, s.what);
            }
            let in_place = s.op == Op::RopeTail && s.inputs[0].buf == s.outputs[0].buf;
            for (name, state) in [("q", &mut q), ("o", &mut o)] {
                let (r, w) = touches(s, name);
                if !r && !w {
                    continue;
                }
                let is_q = name == "q";
                if in_place {
                    assert_eq!(s.attrs.dsv.inverse, !is_q, "{}: {} rotates the wrong way", st.name, s.what);
                    assert_eq!(*state, St::Raw, "{}: {} finds ws:{name} {:?}, not freshly produced", st.name, s.what, *state);
                    assert_eq!(s.lane, 0, "{}: {}", st.name, s.what);
                    *state = St::Rotated;
                    if is_q { n_q += 1 } else { n_o += 1 }
                } else if w {
                    // a producer: wq_b (q) / the attention (o), on lane 0, not reading the buffer it writes
                    assert!(!r, "{}: {} reads and writes ws:{name}", st.name, s.what);
                    assert!(if is_q { s.what.ends_with(".wq_b") } else { s.op == Op::AttnSparse }, "{}: {} writes ws:{name}", st.name, s.what);
                    assert_eq!(s.lane, 0, "{}: {}", st.name, s.what);
                    *state = St::Raw;
                } else {
                    // a plain reader: legal only on the rotated rows, and only the consumer
                    assert_eq!(*state, St::Rotated, "{}: {} reads ws:{name} while it is {:?} (unrotated rows)", st.name, s.what, *state);
                    let ok = if is_q { s.op == Op::AttnSparse && s.inputs[0].buf == Buf::Ws("q".into()) } else { s.what.contains("act_quant(o)") || s.what.contains("wo_a") };
                    assert!(ok, "{}: {} is not the consumer of the rotated ws:{name}", st.name, s.what);
                }
            }
        }
    }
    (n_q, n_o)
}

/// `in_place_rope_leaves_no_reader_of_the_unrotated_rows`: the base graph is checked by [`check_base_buffers`], the flagged one by
/// [`check_no_unrotated_read`], and the two must be the same graph except for the two ropes' output buffers (`q_rope` -> `q`,
/// `o_derot` -> `o`): same steps, lanes, positions, extras, shapes and dtypes. Returns the number of in-place `q` and `o` ropes.
fn check_in_place_rope(base: &Dsv41Graph, ip: &Dsv41Graph) -> (usize, usize) {
    assert_eq!(base.stages.len(), ip.stages.len());
    let ropes = check_base_buffers(base);
    for (sb, si) in base.stages.iter().zip(&ip.stages) {
        assert_eq!(sb.steps.len(), si.steps.len(), "{}", sb.name);
        assert_eq!(sb.extras, si.extras, "{}", sb.name);
        for (a, b) in sb.steps.iter().zip(&si.steps) {
            assert_eq!((a.op, &a.what, a.lane, a.pos), (b.op, &b.what, b.lane, b.pos));
            // (a device-position step is checked with its position tensor by the runtime, not as a plain host call)
            if !b.pos {
                b.phantom_call().unwrap_or_else(|e| panic!("{}: {}: {e}", sb.name, b.what));
            }
            // the rename is exactly: q_rope -> q, o_derot -> o; `q` and `o` themselves keep their names (the base graph touches them
            // only at the producer and the rope, checked above)
            let same = |x: &Buf, y: &Buf| match (x, y) {
                (Buf::Ws(p), Buf::Ws(q)) => (p == "q_rope" && q == "q") || (p == "o_derot" && q == "o") || p == q,
                _ => x == y,
            };
            for (x, y) in a.inputs.iter().zip(&b.inputs).chain(a.outputs.iter().zip(&b.outputs)) {
                assert!(same(&x.buf, &y.buf) && x.shape == y.shape && x.dtype == y.dtype, "{}: {}", sb.name, a.what);
            }
            // no step but the two ropes aliases an input
            if !(b.op == Op::RopeTail && b.inputs[0].buf == b.outputs[0].buf) {
                assert!(b.outputs.iter().all(|o| b.inputs.iter().all(|i| i.buf != o.buf)), "{}: {}", sb.name, b.what);
            }
        }
    }
    let (n_q, n_o) = check_no_unrotated_read(ip);
    assert_eq!(n_q + n_o, ropes, "every attention rope is in place");
    (n_q, n_o)
}

/// Every graph form the CUDA runtime builds, flagged against unflagged, under `check_in_place_rope`; `layers` is the number of
/// attention layers (ropes per kind) of the graph. `label` names the configuration in the assertion messages.
fn in_place_rope_battery(label: &str, c: &Dsv41Config, w: &dyn intarsia_core::dsv41::graph::WeightShapes, world: usize, layers: usize) {
    let off = GraphTp { world, ..GraphTp::default() };
    let on = GraphTp { rope_inplace: true, ..off };
    for (t, pos0) in [(1, 4095), (4, 100), (16, 0), (512, 0), (512, 8192), (2048, 0)] {
        let o = GraphOpts { t, pos0, act: DType::Bf16, cap: 131_072 };
        let (base, ip) = (build_with(c, w, o, off).unwrap_or_else(|e| panic!("{label}: {e}")), build_with(c, w, o, on).unwrap());
        assert_eq!(check_in_place_rope(&base, &ip), (layers, layers), "{label}: t = {t} pos0 = {pos0}: one q rope and one o rope per layer");
    }
    // the device-position graphs (CUDA graph capture) of the 1 to 4 row passes, plain, verify (all rows) and with the DSpark absorb
    for (t, all_rows, absorb) in [(1, false, false), (1, false, true), (4, true, true), (4, true, false), (3, false, true)] {
        let o = GraphOpts { t, pos0: 131_072 - t, act: DType::Bf16, cap: 131_072 };
        let k = PassKind { all_rows, absorb, dpos: true };
        let (base, ip) = match (build_pass(c, w, o, off, k), build_pass(c, w, o, on, k)) {
            (Ok(b), Ok(i)) => (b, i),
            (Err(e), Err(_)) => {
                println!("{label}: skipped t = {t} absorb {absorb}: {e}");
                continue;
            }
            (b, i) => panic!("{label}: t = {t}: {:?} / {:?}", b.err(), i.err()),
        };
        assert_eq!(check_in_place_rope(&base, &ip), (layers, layers), "{label}: dpos t = {t} all_rows = {all_rows} absorb = {absorb}");
        assert!(ip.stages.iter().flat_map(|s| &s.steps).any(|s| s.pos && s.op == Op::RopeTail), "a dpos graph has positioned ropes");
    }
    // the bounded-replay prefill graphs (INTARSIA_DSV41_REPLAY=1, `replay.rs`): the encoder part (layers 0..cut, then the cut layer's
    // mixes and kv-source update: no attention q / o at the cut layer), the decoder part (layers cut.., read from the encoder
    // part's streams) and a whole chunk under the floor. Same checks: the flagged graph is the unflagged one with the two
    // ropes' buffers renamed, and no step reads an unrotated q / o. One q rope and one o rope per attention layer of the part.
    if layers == 40 {
        let cut = 20;
        for (part, t, pos0, first, absorb, want) in [
            (ReplayPart::Encoder, 512, 0, 0, false, cut),
            (ReplayPart::Encoder, 2048, 2048, 0, true, cut),
            (ReplayPart::Encoder, 17, 3961, 0, false, cut),
            (ReplayPart::Decoder, 118, 3978, 1930, false, layers - cut),
            (ReplayPart::Decoder, 118, 3978, 1930, true, layers - cut),
            (ReplayPart::Decoder, 128, 3978, 0, false, layers - cut),
            (ReplayPart::Decoder, 1, 3978, 2047, false, layers - cut),
            (ReplayPart::Whole, 512, 4096, 0, false, layers),
            (ReplayPart::Whole, 16, 4100, 0, true, layers),
        ] {
            let o = GraphOpts { t, pos0, act: DType::Bf16, cap: 131_072 };
            let k = PassKind { all_rows: false, absorb, dpos: false };
            let r = Replay { part, cut, floor: 3978, first };
            let (base, ip) = match (build_replay(c, w, o, off, k, r), build_replay(c, w, o, on, k, r)) {
                (Ok(b), Ok(i)) => (b, i),
                (Err(e), Err(_)) => {
                    println!("{label}: skipped replay {part:?} t = {t} absorb {absorb}: {e}");
                    continue;
                }
                (b, i) => panic!("{label}: replay {part:?} t = {t}: {:?} / {:?}", b.err(), i.err()),
            };
            assert_eq!(check_in_place_rope(&base, &ip), (want, want), "{label}: replay {part:?} t = {t} pos0 = {pos0} absorb = {absorb}");
            check_rope_rows(&ip, t, pos0);
        }
    }
    // the drafting graphs (DSpark stages), host and device position
    let o = GraphOpts { t: 5, pos0: 100, act: DType::Bf16, cap: 131_072 };
    for dpos in [false, true] {
        let mk = |tp: GraphTp| if dpos { build_draft_dpos(c, w, o, tp, 4) } else { build_draft(c, w, o, tp, 4) };
        let (base, ip) = (mk(off).unwrap_or_else(|e| panic!("{label}: {e}")), mk(on).unwrap());
        let (n_q, n_o) = check_in_place_rope(&base, &ip);
        assert!(n_q > 0 && n_q == n_o, "{label}: draft graph (dpos {dpos}): {n_q} q ropes, {n_o} o ropes");
    }
}

/// Every attention rope of a graph of `t` rows at `pos0` rotates exactly the graph's own rows at the graph's own positions (never a
/// row range of another graph: the decoder part of a replay chunk is `t - first` rows at `pos0 + first`), and `q` / `o` are produced
/// and consumed at that row count. The sparse attention may be built at a shifted position (the replay floor) but its ropes may not.
fn check_rope_rows(g: &Dsv41Graph, t: usize, pos0: usize) {
    let mut ropes = 0;
    for st in &g.stages {
        for s in &st.steps {
            let q_or_o = s.what.ends_with(".rope(q)") || s.what.ends_with(".rope_inverse(o)");
            if q_or_o {
                ropes += 1;
                assert_eq!(s.inputs[0].shape[0], t, "{}: {} rotates {} rows of a {t}-row graph", st.name, s.what, s.inputs[0].shape[0]);
                assert_eq!(s.attrs.pos0, pos0, "{}: {} rotates at position {} in a graph at {pos0}", st.name, s.what, s.attrs.pos0);
            }
            if s.what.ends_with(".wq_b") || s.op == Op::AttnSparse {
                assert_eq!(s.outputs[0].shape[0], t, "{}: {}", st.name, s.what);
            }
            if s.what.ends_with(".rope(kv)") || s.what.ends_with(".rope(index_q)") {
                assert_eq!((s.inputs[0].shape[0], s.attrs.pos0), (t, pos0), "{}: {}", st.name, s.what);
            }
        }
    }
    assert!(ropes > 0, "no attention rope found");
}

/// A bounded-replay chunk that crosses the floor runs two graphs one after the other on the same workspace: the encoder part, then the
/// decoder part. With `q` and `o` rotated in place, the pair must behave as one graph for the checks: concatenated in execution
/// order, no step reads an unrotated q / o (the decoder part's first layer finds nothing of the encoder part's left over, it
/// produces `q` first), and every rotation happens once per producer. Also for the chunk sizes at the ends (a 1-row decoder part,
/// the whole 128).
#[test]
fn in_place_rope_across_the_replay_cut() {
    let (c, w) = mia();
    let on = GraphTp { rope_inplace: true, ..GraphTp::default() };
    let k = PassKind { all_rows: false, absorb: true, dpos: false };
    for (chunk_t, pos0, floor) in [(2048usize, 2048usize, 3978usize), (512, 3584, 3978), (512, 3500, 3978), (130, 3900, 3978)] {
        let first = floor - pos0;
        let kept = chunk_t - first;
        let o = |t, p| GraphOpts { t, pos0: p, act: DType::Bf16, cap: 131_072 };
        let enc = build_replay(&c, &w, o(chunk_t, pos0), on, k, Replay { part: ReplayPart::Encoder, cut: 20, floor, first: 0 }).unwrap();
        let dec = build_replay(&c, &w, o(kept, floor), on, k, Replay { part: ReplayPart::Decoder, cut: 20, floor, first }).unwrap();
        check_rope_rows(&enc, chunk_t, pos0);
        check_rope_rows(&dec, kept, floor);
        assert_eq!(check_no_unrotated_read(&enc), (20, 20));
        assert_eq!(check_no_unrotated_read(&dec), (20, 20));
        let mut both = enc.clone();
        both.stages.extend(dec.stages.iter().cloned());
        assert_eq!(check_no_unrotated_read(&both), (40, 40), "encoder part then decoder part ({chunk_t} rows at {pos0}, floor {floor})");
        // the decoder part never names the encoder part's q / o rows: its first q / o touch is a producer
        let first_touch = dec.stages.iter().flat_map(|s| s.steps.iter()).find(|s| touches(s, "q").0 || touches(s, "q").1).unwrap();
        assert!(first_touch.what.ends_with(".wq_b"), "{}", first_touch.what);
        let first_o = dec.stages.iter().flat_map(|s| s.steps.iter()).find(|s| touches(s, "o").0 || touches(s, "o").1).unwrap();
        assert_eq!(first_o.op, Op::AttnSparse, "{}", first_o.what);
    }
}

#[test]
fn in_place_rope_leaves_no_reader_of_the_unrotated_rows() {
    let (c, w) = mia();
    in_place_rope_battery("exl3 world 1", &c, &w, 1, 40);
}

/// The Mia header shapes as another configuration: `dense_wo_a` hides the EXL3 `wo_a` slices and gives each layer the dense
/// block-diagonal `wo_a.weight` (the base after convert.py); `world` shards what a rank holds under tensor parallelism (the
/// `wq_b` columns, the `wo_b` rows, the sinks, the routed and shared experts' intermediate width, the head's vocabulary), as
/// the rank cache does; the graph reads the trellis shapes for every EXL3 linear, so those are the shapes changed.
struct Variant<'a> {
    inner: &'a HeaderShapes,
    c: &'a Dsv41Config,
    dense_wo_a: bool,
    world: usize,
}

impl intarsia_core::dsv41::graph::WeightShapes for Variant<'_> {
    fn get(&self, name: &str) -> Option<(DType, Vec<usize>)> {
        let (c, world) = (self.c, self.world);
        if self.dense_wo_a {
            if name.contains(".wo_a.slice.") {
                return None;
            }
            if let Some(p) = name.strip_suffix(".wo_a.weight") {
                self.inner.get(&format!("{p}.wq_b.trellis"))?;
                return Some((DType::Bf16, vec![c.o_groups / world * c.o_lora_rank, c.n_heads * c.head_dim / c.o_groups]));
            }
        }
        let (dt, mut s) = self.inner.get(name)?;
        if world > 1 {
            let last2 = name.rsplit('.').take(2).collect::<Vec<_>>(); // ["trellis", "w1"]
            if name.ends_with(".attn.wq_b.trellis") {
                s[1] /= world;
            } else if name.ends_with(".attn.wo_b.trellis") {
                s[0] /= world;
            } else if name.ends_with(".attn.attn_sink") {
                s[0] /= world;
            } else if (name.contains(".experts.") || name.contains(".shared_experts.")) && last2[0] == "trellis" {
                match last2[1] {
                    "w1" | "w3" => s[1] /= world,
                    "w2" => s[0] /= world,
                    _ => {}
                }
            } else if name == "head.trellis" {
                s[1] /= world;
            } else if name == "head.weight" {
                s[0] /= world;
            }
        }
        Some((dt, s))
    }
}

/// The configurations the first test does not reach: the FP8 checkpoint's `act_quant(o)` between the de-rotation and `wo_a`, the dense
/// grouped `wo_a`, and a rank of TP=2 (the collectives are `Extra`s in the stages), with the flag on.
#[test]
fn in_place_rope_in_the_fp8_dense_and_tp_configurations() {
    let (c, w) = mia();
    let fp8 = Dsv41Config { quant: intarsia_core::dsv41::config::Dsv41Quant::Fp8 { block: 128, expert_fp4: false }, ..c.clone() };
    let v = |dense_wo_a, world| Variant { inner: &w, c: &c, dense_wo_a, world };
    // FP8 mode emits the act_quant(o) step: assert it is really in the flagged graph and is the consumer
    let o = GraphOpts { t: 16, pos0: 0, act: DType::Bf16, cap: 131_072 };
    let g = build_with(&fp8, &w, o, GraphTp { rope_inplace: true, ..GraphTp::default() }).unwrap();
    assert_eq!(g.stages.iter().flat_map(|s| &s.steps).filter(|s| s.what.ends_with("act_quant(o)")).count(), 40);
    in_place_rope_battery("fp8 act_quant(o), world 1", &fp8, &w, 1, 40);
    in_place_rope_battery("dense wo_a, world 1", &c, &v(true, 1), 1, 40);
    in_place_rope_battery("dense wo_a + fp8, world 1", &fp8, &v(true, 1), 1, 40);
    in_place_rope_battery("exl3, TP=2 rank", &c, &v(false, 2), 2, 40);
    in_place_rope_battery("dense wo_a + fp8, TP=2 rank", &fp8, &v(true, 2), 2, 40);
    // TP: the collectives sit in the stages as extras; none names q or o (check_base_buffers / check_no_unrotated_read assert it) and
    // there are some, so the extras path really ran
    let g = build_with(&c, &v(false, 2), o, GraphTp { world: 2, rope_inplace: true, ..GraphTp::default() }).unwrap();
    assert!(g.stages.iter().any(|s| !s.extras.is_empty()), "a TP graph has collectives");
}

/// A copy of the graph with one extra step appended to stage `stage`.
fn with_step(g: &Dsv41Graph, stage: usize, at: usize, s: intarsia_core::dsv41::graph::Step) -> Dsv41Graph {
    let mut g = g.clone();
    g.stages[stage].steps.insert(at, s);
    g
}

/// The check has teeth: a reader of the unrotated rows injected anywhere later in the graph (a later step, a later layer, before the
/// next layer's producer or between it and its rope) makes it fail, for `q` and for `o`; so does a second writer.
#[test]
fn the_unrotated_read_check_fails_on_an_injected_late_reader() {
    let (c, w) = mia();
    let o = GraphOpts { t: 16, pos0: 0, act: DType::Bf16, cap: 131_072 };
    let off = GraphTp::default();
    let on = GraphTp { rope_inplace: true, ..off };
    let (base, ip) = (build_with(&c, &w, o, off).unwrap(), build_with(&c, &w, o, on).unwrap());
    assert_eq!(check_in_place_rope(&base, &ip), (40, 40));
    let reader = |name: &str, shape: &[usize]| intarsia_core::dsv41::graph::Step {
        op: Op::Add,
        what: format!("injected late reader of ws:{name}"),
        inputs: vec![intarsia_core::dsv41::graph::Arg { buf: Buf::Ws(name.into()), dtype: DType::Bf16, shape: shape.to_vec() }; 2],
        outputs: vec![intarsia_core::dsv41::graph::Arg { buf: Buf::Ws("injected".into()), dtype: DType::Bf16, shape: shape.to_vec() }],
        attrs: Default::default(),
        pos: false,
        lane: 0,
    };
    // the check must fail, and fail with `needle` (not on a bad insertion index of this test)
    let fails = |f: &dyn Fn(), needle: &str| -> bool {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(()) => false,
            Err(e) => {
                let m = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                println!("  caught: {m}");
                m.contains(needle)
            }
        }
    };
    let pos = |g: &Dsv41Graph, stage: usize, f: &dyn Fn(&intarsia_core::dsv41::graph::Step) -> bool| g.stages[stage].steps.iter().position(|s| f(s)).unwrap();
    let shape = ip.stages[2].steps[pos(&ip, 2, &|s| s.what.ends_with("rope(q)"))].outputs[0].shape.clone();
    // 1. nothing injected: fine (control for the lines below)
    check_no_unrotated_read(&ip);
    // 2. flagged graph: a reader of ws:q in layer 1 after its producer and before its rope reads the unrotated rows; so does a reader
    //    of ws:o between the attention and its rope
    let wq_b = pos(&ip, 2, &|s| s.what.ends_with(".wq_b"));
    let g = with_step(&ip, 2, wq_b + 1, reader("q", &shape));
    assert!(fails(&|| drop(check_no_unrotated_read(&g)), "unrotated rows"), "q read between wq_b and its rope");
    let attn = pos(&ip, 2, &|s| s.op == Op::AttnSparse);
    let g = with_step(&ip, 2, attn + 1, reader("o", &shape));
    assert!(fails(&|| drop(check_no_unrotated_read(&g)), "unrotated rows"), "o read between the attention and its rope");
    // 3. a stray reader of ws:q at the start of layer 1 (it holds layer 0's rotated q: not the attention, refused); a stray reader of
    //    ws:o at the end of layer 9, long after its rope
    let g = with_step(&ip, 2, 0, reader("q", &shape));
    assert!(fails(&|| drop(check_no_unrotated_read(&g)), "is not the consumer"), "a stray reader of ws:q");
    let g = with_step(&ip, 10, ip.stages[10].steps.len(), reader("o", &shape));
    assert!(fails(&|| drop(check_no_unrotated_read(&g)), "is not the consumer"), "a stray reader of ws:o in layer 9");
    // 4. the base graph, where those readers WOULD read unrotated rows
    let g = with_step(&base, 5, pos(&base, 5, &|s| s.what.ends_with("rope(q)")) + 1, reader("q", &shape));
    assert!(fails(&|| drop(check_base_buffers(&g)), "reads the unrotated ws:q"), "base: a late reader of the unrotated q");
    let g = with_step(&base, 5, base.stages[5].steps.len(), reader("o", &shape));
    assert!(fails(&|| drop(check_base_buffers(&g)), "reads the unrotated ws:o"), "base: a late reader of the unrotated o");
    // 5. the rope applied twice (a replayed rope without its producer)
    let rope_q = pos(&ip, 2, &|s| s.what.ends_with("rope(q)"));
    let dup = ip.stages[2].steps[rope_q].clone();
    let g = with_step(&ip, 2, rope_q + 1, dup);
    assert!(fails(&|| drop(check_no_unrotated_read(&g)), "not freshly produced"), "a rope applied twice");
}

/// Host cost of building the decode graph (the eager runtime builds one per pass): printed, bounded loosely.
#[test]
fn decode_graph_build_is_cheap() {
    let (c, w) = mia();
    let t0 = std::time::Instant::now();
    let n = 20;
    for i in 0..n {
        build(&c, &w, GraphOpts { t: 1, pos0: 4000 + i, act: DType::Bf16, cap: 131_072 }).unwrap();
    }
    let per = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    let g = build(&c, &w, GraphOpts { t: 1, pos0: 4000, act: DType::Bf16, cap: 131_072 }).unwrap();
    let steps: usize = g.stages.iter().map(|s| s.steps.len()).sum();
    println!("decode graph build: {per:.2} ms, {steps} steps");
    assert!(per < 200.0, "{per:.2} ms per decode graph");
}

/// Speculative decoding graphs of the Mia checkpoint (its `mtp.{0,1,2}` DSpark stages): the verify pass (every row's
/// logits, taps absorbed into the stage rings), one drafting round and the Markov step, on one rank (the header fixture
/// has the unsharded shapes; TP=2 runs in the CPU runtime tests); the plain graph keeps the layout without the rings.
#[test]
fn mia_dspark_graphs_record_with_the_official_structure() {
    use intarsia_core::dsv41::graph::{build_draft, build_markov, build_pass, build_with, dspark_idx, dspark_stages, GraphTp, PassKind};
    let (c, w) = mia();
    assert_eq!(dspark_stages(&c, &w), 3);
    let m = c.mtp.clone().unwrap();
    assert_eq!((m.block_size, m.n_routed_experts, m.top_k, m.target_layer_ids.clone()), (5, 128, 3, vec![37, 38, 39]));
    let check = |g: &intarsia_core::dsv41::graph::Dsv41Graph| {
        for st in &g.stages {
            for s in &st.steps {
                s.phantom_call().unwrap_or_else(|e| panic!("{}: {e}", s.what));
                for o in &s.outputs {
                    assert!(s.inputs.iter().all(|i| i.buf != o.buf), "{}: {:?} is read and written", s.what, o.buf);
                }
            }
        }
    };
    let plain = build_with(&c, &w, GraphOpts { t: 4, pos0: 100, act: DType::Bf16, cap: 4096 }, GraphTp::default()).unwrap();
    for world in [1] {
        let tp = GraphTp { world, layers: None, first: 0, rope_inplace: false };
        let o = GraphOpts { t: 4, pos0: 100, act: DType::Bf16, cap: 4096 };
        let v = build_pass(&c, &w, o, tp, PassKind { all_rows: true, absorb: true, dpos: false }).unwrap();
        check(&v);
        assert_eq!(v.logits.shape, [4, c.vocab], "world {world}: every row's logits");
        assert_eq!(v.slots.len(), plain.slots.len() + 3, "one window ring per stage");
        assert!(v.stages.iter().any(|s| s.name == "dspark absorb"));
        // a prompt chunk of the same size keeps the last row's logits only
        let p = build_pass(&c, &w, o, tp, PassKind { all_rows: false, absorb: true, dpos: false }).unwrap();
        assert_eq!(p.logits.shape, [1, c.vocab]);
        let d = build_draft(&c, &w, GraphOpts { t: 5, pos0: 101, act: DType::Bf16, cap: 4096 }, tp, 4).unwrap();
        check(&d);
        assert_eq!(d.logits.shape, [5, c.vocab]);
        assert_eq!(d.ids.as_ref().map(|a| a.shape.clone()), Some(vec![4]), "four Markov steps");
        assert_eq!(d.stages.iter().flat_map(|s| &s.steps).filter(|s| s.op == Op::Argmax).count(), 4);
        assert_eq!(format!("{:?}", d.slots), format!("{:?}", v.slots), "the draft graph reads the verify graph's rings");
        let moes: Vec<_> = d.stages.iter().flat_map(|s| &s.steps).filter(|s| s.op == Op::MoeExperts).collect();
        assert_eq!(moes.len(), 3);
        assert_eq!(moes[0].inputs[1].shape, [5, 4], "3 routed slots + the shared one");
    }
    let mk = build_markov(&c, &w, DType::Bf16).unwrap();
    check(&mk);
    assert_eq!(mk.logits.shape, [1, c.vocab]);
    assert!(build_pass(&c, &w, GraphOpts { t: 17, pos0: 0, act: DType::Bf16, cap: 4096 }, GraphTp::default(), PassKind { all_rows: true, absorb: false, dpos: false }).is_err());

    // the block key table: row r sees window rows 0..=r (valid positions) and the block rows after it
    let win = c.window;
    let idx = dspark_idx(win, 3, 1000);
    assert_eq!(idx, vec![0, (win + 1) as i32, (win + 2) as i32, 0, 1, (win + 2) as i32, 0, 1, 2]);
    // at position 2 only window rows at positions >= 0 are valid (j >= W - 2)
    let idx = dspark_idx(win, 3, 2);
    let (w1, w2) = ((win + 1) as i32, (win + 2) as i32);
    assert_eq!(idx, vec![w1, w2, -1, w2, -1, -1, -1, -1, -1]);
}
