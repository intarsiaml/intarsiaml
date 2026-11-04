//! DeepSeek-V4.1 skeleton (lane 1a): the recipe matches, loading refuses with a pointer to `plan show`, and the graph of
//! the real Mia EXL3 checkpoint (header-only fixture: every tensor's dtype and shape, routed experts 1..382 omitted) records
//! as checked `OpCall`s on the cpu backend with the expected op sequence, state slots and MTP blocks.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::dsv41::config::Dsv41Config;
use intarsia_core::dsv41::graph::{build, build_draft, build_draft_dpos, build_pass, build_with, Buf, Dsv41Graph, GraphOpts, GraphTp, PassKind};
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

/// `in_place_rope_leaves_no_reader_of_the_unrotated_rows`: compare a graph with `GraphTp::rope_inplace` to the one without and
/// return the number of in-place `q` and `o` ropes.
fn check_in_place_rope(base: &Dsv41Graph, ip: &Dsv41Graph) -> (usize, usize) {
    assert_eq!(base.stages.len(), ip.stages.len());
    let (mut n_q, mut n_o) = (0, 0);
    for (sb, si) in base.stages.iter().zip(&ip.stages) {
        // same steps in the same order, lanes and positions; the only difference is the rope output buffer and its readers
        assert_eq!(sb.steps.len(), si.steps.len(), "{}", sb.name);
        assert_eq!(sb.extras, si.extras, "{}", sb.name);
        for (j, (a, b)) in sb.steps.iter().zip(&si.steps).enumerate() {
            assert_eq!((a.op, &a.what, a.lane, a.pos), (b.op, &b.what, b.lane, b.pos));
            // (a device-position step is checked with its position tensor by the runtime, not as a plain host call)
            if !b.pos {
                b.phantom_call().unwrap_or_else(|e| panic!("{}: {}: {e}", sb.name, b.what));
            }
            let same = |x: &Buf, y: &Buf| match (x, y) {
                (Buf::Ws(p), Buf::Ws(q)) => (p == "q_rope" && q == "q") || (p == "o_derot" && q == "o") || p == q,
                _ => x == y,
            };
            assert!(a.inputs.iter().zip(&b.inputs).all(|(x, y)| same(&x.buf, &y.buf) && x.shape == y.shape && x.dtype == y.dtype), "{}: {}", sb.name, a.what);
            assert!(a.outputs.iter().zip(&b.outputs).all(|(x, y)| same(&x.buf, &y.buf) && x.shape == y.shape && x.dtype == y.dtype), "{}: {}", sb.name, a.what);
            // no step but the two ropes aliases an input
            if !(b.op == Op::RopeTail && b.inputs[0].buf == b.outputs[0].buf) {
                assert!(b.outputs.iter().all(|o| b.inputs.iter().all(|i| i.buf != o.buf)), "{}: {}", sb.name, b.what);
                continue;
            }
            let (buf, inverse) = (&b.inputs[0].buf, b.attrs.dsv.inverse);
            assert_eq!(buf, &Buf::Ws(if inverse { "o" } else { "q" }.into()), "{}: {}", sb.name, b.what);
            assert_eq!(b.lane, 0, "{}: {}", sb.name, b.what);
            // the producer is the nearest earlier step writing the buffer; nothing but the rope touches it before the rope
            let prod = &si.steps[..j].iter().rposition(|s| s.outputs.iter().any(|x| &x.buf == buf)).unwrap();
            let prod = *prod;
            assert_eq!(si.steps[prod].lane, 0);
            if inverse {
                assert_eq!(si.steps[prod].op, Op::AttnSparse, "{}", sb.name);
            } else {
                assert!(si.steps[prod].what.ends_with(".wq_b"), "{}: q is produced by {}", sb.name, si.steps[prod].what);
            }
            for s in &si.steps[prod + 1..j] {
                assert!(s.inputs.iter().chain(&s.outputs).all(|x| &x.buf != buf), "{}: {} touches the unrotated rows", sb.name, s.what);
            }
            // the first step after the rope that touches the buffer reads it, and it is the consumer of the rotated rows
            let k = si.steps[j + 1..].iter().position(|s| s.inputs.iter().chain(&s.outputs).any(|x| &x.buf == buf));
            let next = &si.steps[j + 1 + k.expect("a reader after the rope")];
            assert!(next.inputs.iter().any(|x| &x.buf == buf) && next.outputs.iter().all(|x| &x.buf != buf), "{}: {}", sb.name, next.what);
            if inverse {
                assert!(next.what.contains("act_quant(o)") || next.what.contains("wo_a"), "{}: first reader of the de-rotated o is {}", sb.name, next.what);
                n_o += 1;
            } else {
                assert_eq!((next.op, &next.inputs[0].buf), (Op::AttnSparse, buf), "{}: first reader of the rotated q is {}", sb.name, next.what);
                n_q += 1;
            }
        }
    }
    (n_q, n_o)
}

/// `GraphTp::rope_inplace` (CUDA): the attention's `q` and `o` ropes write the buffer they read. Nothing may read the unrotated
/// rows (between the step that produced the buffer and the rope only the rope touches it, and the first step after the rope
/// is the attention for `q` and the output projection for `o`), the other steps, lanes and extras are unchanged, and the
/// graph without the flag has no aliasing at all.
#[test]
fn in_place_rope_leaves_no_reader_of_the_unrotated_rows() {
    let (c, w) = mia();
    let off = GraphTp::default();
    let on = GraphTp { rope_inplace: true, ..off };
    for (t, pos0) in [(1, 4095), (4, 100), (16, 0), (512, 0), (512, 8192), (2048, 0)] {
        let o = GraphOpts { t, pos0, act: DType::Bf16, cap: 131_072 };
        let (base, ip) = (build_with(&c, &w, o, off).unwrap(), build_with(&c, &w, o, on).unwrap());
        assert_eq!(check_in_place_rope(&base, &ip), (40, 40), "t = {t} pos0 = {pos0}: one q rope and one o rope per layer");
        for st in base.stages.iter().flat_map(|s| s.steps.iter()) {
            assert!(st.outputs.iter().all(|o| st.inputs.iter().all(|i| i.buf != o.buf)), "{}", st.what);
        }
    }
    // what the CUDA runtime really runs: the device-position graphs (CUDA graph capture; `dpos` makes every positioned step read the
    // position from the device) of the 1 to 4 row passes, plain, verify (all rows) and with the DSpark absorb
    for (t, all_rows, absorb) in [(1, false, false), (1, false, true), (4, true, true), (4, true, false), (3, false, true)] {
        let o = GraphOpts { t, pos0: 131_072 - t, act: DType::Bf16, cap: 131_072 };
        let k = PassKind { all_rows, absorb, dpos: true };
        let (base, ip) = match (build_pass(&c, &w, o, off, k), build_pass(&c, &w, o, on, k)) {
            (Ok(b), Ok(i)) => (b, i),
            (Err(e), Err(_)) => {
                println!("skipped t = {t} absorb {absorb}: {e}");
                continue;
            }
            (b, i) => panic!("t = {t}: {:?} / {:?}", b.err(), i.err()),
        };
        assert_eq!(check_in_place_rope(&base, &ip), (40, 40), "dpos t = {t} all_rows = {all_rows} absorb = {absorb}");
        assert!(ip.stages.iter().flat_map(|s| &s.steps).any(|s| s.pos && s.op == Op::RopeTail), "a dpos graph has positioned ropes");
    }
    // the drafting graphs (DSpark stages), host and device position, have the same two ropes
    let o = GraphOpts { t: 5, pos0: 100, act: DType::Bf16, cap: 131_072 };
    for dpos in [false, true] {
        let mk = |tp: GraphTp| if dpos { build_draft_dpos(&c, &w, o, tp, 4) } else { build_draft(&c, &w, o, tp, 4) };
        let (base, ip) = (mk(off).unwrap(), mk(on).unwrap());
        let (n_q, n_o) = check_in_place_rope(&base, &ip);
        assert!(n_q > 0 && n_q == n_o, "draft graph (dpos {dpos}): {n_q} q ropes, {n_o} o ropes");
    }
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
