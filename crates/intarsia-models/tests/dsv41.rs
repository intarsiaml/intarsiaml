//! DeepSeek-V4.1 skeleton (lane 1a): the recipe matches, loading refuses with a pointer to `plan show`, and the graph of
//! the real Mia EXL3 checkpoint (header-only fixture: every tensor's dtype and shape, routed experts 1..382 omitted) records
//! as checked `OpCall`s on the cpu backend with the expected op sequence, state slots and MTP blocks.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::dsv41::config::Dsv41Config;
use intarsia_core::dsv41::graph::{build, Buf, GraphOpts};
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
    // every layer: two hyper-connection sub-blocks, MQA with sink over the window, 8 wo_a group slices, routed + shared MoE
    for l in 0..40 {
        let tap = usize::from(l >= 37);
        assert_eq!((count(l, Op::HcMixes), count(l, Op::HcPre), count(l, Op::HcPost)), (2, 2 + tap, 2), "layer {l}");
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
    assert_eq!((steps[0], steps[1], steps[2], steps[3], steps[14], steps[20], steps[24], steps[39]), (38, 41, 53, 38, 56, 52, 43, 39));

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

    // routed experts: MoeExperts in its EXL3 encoding (i16 arena, scales [E, 3H + 3I], meta [E, 8], codebook, limit)
    let ex = |l: usize| g.stages[l + 1].steps.iter().find(|s| s.op == Op::MoeExperts).unwrap().inputs.clone();
    let shapes: Vec<(DType, Vec<usize>)> = ex(0)[3..].iter().map(|a| (a.dtype, a.shape.clone())).collect();
    // layer 0's experts are 3 bits: gate / up [320, 144, 48] and down [144, 320, 48] words, 384 experts
    assert_eq!(shapes[0], (DType::I16, vec![384 * 3 * 320 * 144 * 48]));
    assert_eq!(shapes[1..], [(DType::F16, vec![384, 3 * 5120 + 3 * 2304]), (DType::I32, vec![384, 8]), (DType::I32, vec![1]), (DType::F32, vec![1])]);
    assert_eq!(ex(20)[3].shape, [384 * 3 * 320 * 144 * 32], "layer 20's experts are 2 bits");

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
        if matches!(op, Op::MatmulFp8Block | Op::GroupedMatmul) {
            assert!(!ops.contains_key(op.as_str()));
            continue;
        }
        // the fused HcSublayer is not wired into the graph yet (lane dsv41/tp2 switches to it; drop this skip then)
        if op == Op::HcSublayer {
            continue;
        }
        let (k, _) = &ops[op.as_str()];
        assert_eq!(k, &format!("cpu_f32_{}", op.as_str()));
    }
    assert_eq!(ops["matmul_exl3"].1, 622);
    assert_eq!(ops["hc_pre"].1, 84, "two per layer, the collapse before the head, three DSpark taps");
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
