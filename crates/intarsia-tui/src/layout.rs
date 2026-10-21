//! The architecture map as data: `GET /v1/intarsia/plan` (+ the op names the telemetry reports) in, rows of nodes out.
//!
//! Nothing here is specific to a model: layer groups come from the plan's `layers[].blocks` strings, kernel nodes from
//! `ops[]`, so a new block kind or op shows up on the map without a code change. The only hard-coded knowledge is the
//! engine's fixed spine (queue -> prefill/decode -> layers -> kernels -> KV / sampler / output) and a small prefix table
//! saying which ops belong to the KV cache and the sampler instead of the generic kernel row ([`OWNERS`]).

use serde_json::Value;

/// One kernel choice of the plan (`ops[]`).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanOp {
    pub op: String,
    pub kernel: String,
    pub backend: String,
    pub exactness: String,
    pub admitted: bool,
    pub calls_per_decode_step: Option<u64>,
}

/// One layer of the plan (`layers[]`): the block kinds it runs, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanLayer {
    pub index: usize,
    pub blocks: Vec<String>,
}

/// The parts of `Engine::plan()` the map needs. Unknown keys are ignored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub model: String,
    pub backend: String,
    pub device: String,
    pub exactness: String,
    pub layers: Vec<PlanLayer>,
    pub ops: Vec<PlanOp>,
}

impl Plan {
    /// `None` for `null` (an engine without a plan) or anything that is not an object.
    pub fn from_json(v: &Value) -> Option<Plan> {
        let o = v.as_object()?;
        let s = |k: &str| o.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let layers = o
            .get("layers")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .enumerate()
                    .map(|(i, l)| PlanLayer {
                        index: l["index"].as_u64().map_or(i, |x| x as usize),
                        blocks: l["blocks"]
                            .as_array()
                            .map(|b| b.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                            .filter(|b: &Vec<String>| !b.is_empty())
                            .unwrap_or_else(|| vec![l["kind"].as_str().unwrap_or("unknown").to_string()]),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let ops = o
            .get("ops")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| {
                        Some(PlanOp {
                            op: x["op"].as_str()?.to_string(),
                            kernel: x["kernel"].as_str().unwrap_or("-").to_string(),
                            backend: x["backend"].as_str().unwrap_or("-").to_string(),
                            exactness: x["exactness"].as_str().unwrap_or("-").to_string(),
                            admitted: x["admitted"].as_bool().unwrap_or(true),
                            calls_per_decode_step: x["calls_per_decode_step"].as_u64(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Some(Plan {
            model: s("model"),
            backend: s("backend"),
            device: o.get("device").and_then(|d| d["name"].as_str()).unwrap_or("").to_string(),
            exactness: s("exactness"),
            layers,
            ops,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Queue,
    Prefill,
    Decode,
    /// A distinct layer signature (its blocks) and how many layers share it.
    LayerGroup,
    /// A kernel / op (drawn compactly, in a grid).
    Op,
    Kv,
    Sampler,
    Output,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Stable id: `queue`, `prefill`, `decode`, `layers:<sig>`, `op:<name>`, `kv`, `sampler`, `output`.
    pub id: String,
    pub kind: NodeKind,
    pub title: String,
    /// Static facts from the plan (shown under the title, before any live metric).
    pub detail: Vec<String>,
    /// Layer groups: the glyph this group has in [`ArchMap::layer_strip`].
    pub glyph: Option<char>,
    /// Ops: the op names this node stands for (KV and sampler list the ops they own).
    pub ops: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub nodes: Vec<Node>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArchMap {
    pub rows: Vec<Row>,
    /// One glyph per layer in order (same glyph = same block signature).
    pub layer_strip: Vec<char>,
    pub layer_count: usize,
    /// False when the engine returned no plan and the map is built from telemetry op names only.
    pub from_plan: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    Kernel,
    Kv,
    Sampler,
}

/// Which node an op is attributed to, by name prefix. Anything not listed is a plain kernel.
pub const OWNERS: &[(&str, Owner)] = &[
    ("kv", Owner::Kv),
    ("argmax", Owner::Sampler),
    ("sample", Owner::Sampler),
    ("logits", Owner::Sampler),
    ("top_", Owner::Sampler),
    ("topk", Owner::Sampler),
    ("topp", Owner::Sampler),
    ("softmax", Owner::Sampler),
];

pub fn owner_of(op: &str) -> Owner {
    OWNERS.iter().find(|(p, _)| op.starts_with(p)).map_or(Owner::Kernel, |(_, o)| *o)
}

const GLYPHS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Build the map. `telemetry_ops` are the op names the live snapshot reports (added when the plan lacks them, and the only
/// source when there is no plan).
pub fn build(plan: Option<&Plan>, telemetry_ops: &[String]) -> ArchMap {
    // The plan lists one entry per (op, kernel variant), e.g. matmul with gemv / gemv_rows / nax kernels. Telemetry and the
    // map work per op name: merge variants, keep the busiest one (most calls per decode step) as the shown kernel.
    let mut ops: Vec<(PlanOp, usize)> = Vec::new();
    for o in plan.map(|p| p.ops.as_slice()).unwrap_or(&[]) {
        match ops.iter().position(|(x, _)| x.op == o.op) {
            Some(i) => {
                ops[i].1 += 1;
                if o.calls_per_decode_step > ops[i].0.calls_per_decode_step {
                    let n = ops[i].1;
                    ops[i] = (o.clone(), n);
                }
            }
            None => ops.push((o.clone(), 1)),
        }
    }
    for name in telemetry_ops {
        if !ops.iter().any(|(o, _)| &o.op == name) {
            ops.push((
                PlanOp {
                    op: name.clone(),
                    kernel: "-".into(),
                    backend: "-".into(),
                    exactness: "-".into(),
                    admitted: true,
                    calls_per_decode_step: None,
                },
                1,
            ));
        }
    }
    let node = |id: &str, kind, title: &str, detail: Vec<String>| Node {
        id: id.into(),
        kind,
        title: title.into(),
        detail,
        glyph: None,
        ops: Vec::new(),
    };
    let mut rows = vec![
        Row { nodes: vec![node("queue", NodeKind::Queue, "request queue / scheduler", vec![])] },
        Row {
            nodes: vec![
                node("prefill", NodeKind::Prefill, "prefill", plan.map_or(vec![], |p| p.exactness_line())),
                node("decode", NodeKind::Decode, "decode", vec![]),
            ],
        },
    ];

    // layers: distinct signatures in order of first appearance
    let mut sigs: Vec<(String, Vec<String>, usize)> = Vec::new();
    let mut strip = Vec::new();
    let layers: &[PlanLayer] = plan.map_or(&[], |p| &p.layers);
    for l in layers {
        let sig = l.blocks.join("+");
        let idx = match sigs.iter().position(|(s, _, _)| *s == sig) {
            Some(i) => i,
            None => {
                sigs.push((sig, l.blocks.clone(), 0));
                sigs.len() - 1
            }
        };
        sigs[idx].2 += 1;
        strip.push(glyph(idx));
    }
    let layer_nodes: Vec<Node> = sigs
        .iter()
        .enumerate()
        .map(|(i, (sig, blocks, n))| Node {
            id: format!("layers:{sig}"),
            kind: NodeKind::LayerGroup,
            title: format!("[{}] x{n}", glyph(i)),
            detail: vec![blocks.join(" > ")],
            glyph: Some(glyph(i)),
            ops: Vec::new(),
        })
        .collect();
    if !layer_nodes.is_empty() {
        rows.push(Row { nodes: layer_nodes });
    }

    // kernels
    let kernel_nodes: Vec<Node> = ops
        .iter()
        .filter(|(o, _)| owner_of(&o.op) == Owner::Kernel)
        .map(|(o, variants)| Node {
            id: format!("op:{}", o.op),
            kind: NodeKind::Op,
            title: o.op.clone(),
            detail: vec![format!("{} {} {}{}", o.kernel, o.backend, o.exactness, if *variants > 1 { format!(" (+{} variants)", variants - 1) } else { String::new() })],
            glyph: None,
            ops: vec![o.op.clone()],
        })
        .collect();
    if !kernel_nodes.is_empty() {
        rows.push(Row { nodes: kernel_nodes });
    }

    // KV cache, sampler, output
    let owned = |who: Owner| -> Vec<String> { ops.iter().filter(|(o, _)| owner_of(&o.op) == who).map(|(o, _)| o.op.clone()).collect() };
    let mut kv = node("kv", NodeKind::Kv, "KV cache", vec![]);
    kv.ops = owned(Owner::Kv);
    let mut sampler = node("sampler", NodeKind::Sampler, "sampler", vec![]);
    sampler.ops = owned(Owner::Sampler);
    let output = node("output", NodeKind::Output, "output", vec![]);
    rows.push(Row { nodes: vec![kv, sampler, output] });

    ArchMap { rows, layer_strip: strip, layer_count: layers.len(), from_plan: plan.is_some() }
}

fn glyph(i: usize) -> char {
    GLYPHS.get(i).map_or('#', |b| *b as char)
}

impl Plan {
    fn exactness_line(&self) -> Vec<String> {
        if self.exactness.is_empty() {
            vec![]
        } else {
            vec![format!("exactness {}", self.exactness)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn hybrid_plan() -> Value {
        // the pattern of a hybrid model: three linear-attention layers then one full-attention layer, MoE after each
        let mut layers = Vec::new();
        for i in 0..8 {
            let mixer = if i % 4 == 3 { "attention" } else { "linear_attention" };
            layers.push(json!({"index": i, "kind": mixer, "blocks": [mixer, "moe"]}));
        }
        json!({
            "model": "Qwen/Qwen3.5-35B-A3B", "backend": "metal",
            "device": {"name": "Apple M5 Max", "mem_total": 1, "arch": "x"},
            "exactness": "fast", "layers": layers,
            "ops": [
                {"op": "matmul", "kernel": "metal_gemv", "backend": "metal", "exactness": "fast", "admitted": true, "calls_per_decode_step": 300},
                {"op": "kv_append", "kernel": "metal_kv", "backend": "metal", "exactness": "exact", "admitted": true, "calls_per_decode_step": 10},
                {"op": "argmax", "kernel": "metal_argmax", "backend": "metal", "exactness": "exact", "admitted": true, "calls_per_decode_step": 1},
                {"op": "gdn_scan", "kernel": "metal_gdn", "backend": "metal", "exactness": "fast", "admitted": true, "calls_per_decode_step": 30}
            ]
        })
    }

    #[test]
    fn groups_layers_by_signature_in_order() {
        let plan = Plan::from_json(&hybrid_plan()).unwrap();
        let m = build(Some(&plan), &[]);
        assert!(m.from_plan);
        assert_eq!(m.layer_count, 8);
        assert_eq!(m.layer_strip.iter().collect::<String>(), "AAABAAAB");
        let layers = &m.rows[2];
        assert_eq!(layers.nodes.len(), 2);
        assert_eq!(layers.nodes[0].title, "[A] x6");
        assert_eq!(layers.nodes[0].detail, vec!["linear_attention > moe"]);
        assert_eq!(layers.nodes[1].title, "[B] x2");
    }

    #[test]
    fn spine_order_and_op_ownership() {
        let plan = Plan::from_json(&hybrid_plan()).unwrap();
        let m = build(Some(&plan), &[]);
        let ids: Vec<Vec<&str>> = m.rows.iter().map(|r| r.nodes.iter().map(|n| n.id.as_str()).collect()).collect();
        assert_eq!(ids[0], vec!["queue"]);
        assert_eq!(ids[1], vec!["prefill", "decode"]);
        assert!(ids[2][0].starts_with("layers:"));
        // gdn_scan and matmul are generic kernels; kv_append and argmax moved to their owners
        assert_eq!(ids[3], vec!["op:matmul", "op:gdn_scan"]);
        assert_eq!(ids[4], vec!["kv", "sampler", "output"]);
        assert_eq!(m.rows[4].nodes[0].ops, vec!["kv_append"]);
        assert_eq!(m.rows[4].nodes[1].ops, vec!["argmax"]);
    }

    #[test]
    fn unknown_block_kind_and_op_appear_without_code_changes() {
        let mut v = hybrid_plan();
        v["layers"].as_array_mut().unwrap().push(json!({"index": 8, "kind": "mamba3", "blocks": ["mamba3", "dense_ffn"]}));
        let plan = Plan::from_json(&v).unwrap();
        let m = build(Some(&plan), &["flux_capacitor".to_string()]);
        let layers = &m.rows[2];
        assert_eq!(layers.nodes.len(), 3);
        assert_eq!(layers.nodes[2].detail, vec!["mamba3 > dense_ffn"]);
        assert_eq!(m.layer_strip.last(), Some(&'C'));
        let kernels = &m.rows[3];
        assert!(kernels.nodes.iter().any(|n| n.id == "op:flux_capacitor"));
    }

    #[test]
    fn kernel_variants_of_one_op_become_one_node() {
        let plan = Plan::from_json(&json!({"ops": [
            {"op": "matmul", "kernel": "gemv_rows", "calls_per_decode_step": 0},
            {"op": "matmul", "kernel": "gemv", "calls_per_decode_step": 108},
            {"op": "matmul", "kernel": "nax", "calls_per_decode_step": 0},
            {"op": "logits", "kernel": "a"}, {"op": "logits", "kernel": "b"}
        ]}))
        .unwrap();
        let m = build(Some(&plan), &["matmul".to_string()]);
        let kernels = m.rows.iter().find(|r| r.nodes[0].kind == NodeKind::Op).unwrap();
        assert_eq!(kernels.nodes.len(), 1);
        assert!(kernels.nodes[0].detail[0].starts_with("gemv "), "{:?}", kernels.nodes[0].detail);
        assert!(kernels.nodes[0].detail[0].ends_with("(+2 variants)"));
        let sampler = &m.rows.last().unwrap().nodes[1];
        assert_eq!(sampler.ops, vec!["logits"]);
    }

    #[test]
    fn null_plan_falls_back_to_telemetry_ops() {
        assert!(Plan::from_json(&Value::Null).is_none());
        let m = build(None, &["matmul".into(), "argmax".into(), "kv_append".into()]);
        assert!(!m.from_plan);
        assert_eq!(m.layer_count, 0);
        // spine rows: queue, engine, kernels (matmul), kv/sampler/output; no layer row
        assert_eq!(m.rows.len(), 4);
        assert_eq!(m.rows[2].nodes[0].id, "op:matmul");
        assert_eq!(m.rows[3].nodes[1].ops, vec!["argmax"]);
    }

    #[test]
    fn layer_blocks_fall_back_to_kind() {
        let p = Plan::from_json(&json!({"layers": [{"index": 0, "kind": "attention"}, {"kind": "moe", "blocks": []}]})).unwrap();
        assert_eq!(p.layers[0].blocks, vec!["attention"]);
        assert_eq!(p.layers[1].blocks, vec!["moe"]);
    }
}
