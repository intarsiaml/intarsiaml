//! `intarsia plan show` for DeepSeek-V4.1: record the graph of a decode step (and of a prefill chunk) against a backend's
//! registry and describe it. The model does not execute yet (lanes 1b / 2), so nothing is loaded or admitted here: every
//! step is checked as an `OpCall` on shape-only tensors and resolved to the kernel the registry would pick.

use super::config::Dsv41Config;
use super::graph::{build, Buf, Dsv41Graph, GraphOpts, WeightShapes};
use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::Result;
use crate::registry::Registry;
use crate::state::SlotSpec;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Decode position the plan records at: odd, so the ratio-2 compressors complete a group in that step (they pool every
/// other step; the op list counts them as running).
pub const PLAN_POS: usize = 4095;

/// The plan JSON (the schema of `Model::plan` where it overlaps, plus `family`, `residual_streams`, `state_slots`,
/// `sequence`, `prefill` and `mtp`).
pub fn plan_json(model: &str, c: &Dsv41Config, w: &dyn WeightShapes, backend: &dyn Backend, cap: usize, prefill_rows: usize) -> Result<Value> {
    let info = backend.info();
    let reg = Registry::new(backend.name(), &info, backend.kernels());
    let act = DType::Bf16;
    let decode = build(c, w, GraphOpts { t: 1, pos0: PLAN_POS, act, cap })?;
    let prefill = build(c, w, GraphOpts { t: prefill_rows, pos0: 0, act, cap })?;

    // op -> (kernel, contract status, calls per decode step)
    let mut ops: BTreeMap<(usize, String), (String, u64)> = BTreeMap::new();
    let mut sequence = Vec::new();
    let mut layers = Vec::new();
    for st in &decode.stages {
        let mut steps = Vec::new();
        for s in &st.steps {
            let call = s.phantom_call()?;
            let kernel = reg.select(&call).map(|k| k.name.clone()).unwrap_or_else(|e| format!("NONE ({e})"));
            let e = ops.entry((s.op as usize, kernel.clone())).or_insert_with(|| (s.op.as_str().to_string(), 0));
            e.1 += 1;
            steps.push(json!({
                "op": s.op.as_str(),
                "what": s.what,
                "kernel": kernel,
                "inputs": s.inputs.iter().map(|a| arg_str(a.dtype, &a.shape, &a.buf)).collect::<Vec<_>>(),
                "outputs": s.outputs.iter().map(|a| arg_str(a.dtype, &a.shape, &a.buf)).collect::<Vec<_>>(),
            }));
        }
        if st.layer.is_some() {
            layers.push(json!({ "layer": st.layer, "blocks": st.blocks }));
        }
        sequence.push(json!({ "stage": st.name, "layer": st.layer, "blocks": st.blocks, "steps": steps }));
    }
    // the prefill chunk only needs to record cleanly (its shapes exercise the group / window / index rules); count per op
    let mut prefill_ops: BTreeMap<&'static str, u64> = BTreeMap::new();
    for st in &prefill.stages {
        for s in &st.steps {
            s.phantom_call()?;
            *prefill_ops.entry(s.op.as_str()).or_default() += 1;
        }
    }
    let status = |op: &str| {
        if super::ops::OPS.iter().any(|o| o.as_str() == op) {
            if STUB_OPS.contains(&op) { "stub (lane 1b)" } else { "dsv41 cpu ref" }
        } else {
            "existing"
        }
    };
    let ops: Vec<Value> = ops
        .into_iter()
        .map(|((_, kernel), (op, n))| json!({ "op": op, "kernel": kernel, "status": status(&op), "calls_per_decode_step": n }))
        .collect();
    Ok(json!({
        "model": model,
        "family": "deepseek_v41",
        "executes": false,
        "backend": backend.name(),
        "device": { "name": info.name, "mem_total": info.total_mem },
        "act_dtype": act.to_string(),
        "kv_dtype": act.to_string(),
        "residual_streams": decode.residual_streams,
        "decode_position": PLAN_POS,
        "cap_tokens": cap,
        "state_slots": slots_json(&decode, cap),
        "layers": layers,
        "ops": ops,
        "steps_per_decode_step": decode.stages.iter().map(|s| s.steps.len()).sum::<usize>(),
        "prefill": { "rows": prefill_rows, "steps": prefill.stages.iter().map(|s| s.steps.len()).sum::<usize>(), "ops": prefill_ops },
        "mtp": decode.mtp.iter().map(|m| json!({ "prefix": m.prefix, "experts": m.experts, "top_k": m.top_k, "expert_tensors_found": m.tensors, "executes": false })).collect::<Vec<_>>(),
        "sequence": sequence,
    }))
}

/// Ops whose cpu kernels are stubs (mirrors `intarsia_backend_cpu::dsv41_ref::STUBS`, which core cannot see).
const STUB_OPS: [&str; 2] = ["matmul_fp8_block", "moe_experts_exl3"];

fn arg_str(dt: DType, shape: &[usize], buf: &Buf) -> String {
    let at = match buf {
        Buf::Weight(n) => format!("w:{n}"),
        Buf::Ws(n) => format!("ws:{n}"),
        Buf::Slot { slot, plane, row0 } => format!("slot{slot}.{plane}@{row0}"),
        Buf::Const(n) => format!("const:{n}"),
        Buf::Host(n) => format!("host:{n}"),
    };
    format!("{dt}{shape:?} {at}")
}

fn slots_json(g: &Dsv41Graph, cap: usize) -> Vec<Value> {
    g.slots
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let bytes = s.bytes(cap);
            match s {
                SlotSpec::Ring { rows, width, planes, dtype, kind, .. } => json!({ "slot": i, "kind": kind, "rows": rows, "width": width, "planes": planes, "dtype": dtype.to_string(), "bytes": bytes }),
                SlotSpec::Shared { ratio, width, dtype, writer, readers, kind } => json!({ "slot": i, "kind": kind, "ratio": ratio, "width": width, "dtype": dtype.to_string(), "writer": writer, "readers": readers, "bytes": bytes }),
                other => json!({ "slot": i, "kind": format!("{other:?}"), "bytes": bytes }),
            }
        })
        .collect()
}

/// Plain-text rendering of [`plan_json`] (`intarsia plan show` without `--json`).
pub fn render(p: &Value) -> String {
    use std::fmt::Write;
    let s = |v: &Value| v.as_str().unwrap_or("?").to_string();
    let mut o = String::new();
    let _ = writeln!(o, "model {}  family deepseek_v41  backend {}  device {}  (graph only: the model does not execute yet)", s(&p["model"]), s(&p["backend"]), s(&p["device"]["name"]));
    let _ = writeln!(
        o,
        "act {}  residual streams {} (hyper-connections)  decode step at position {}  {} steps/decode step  prefill {} rows: {} steps",
        s(&p["act_dtype"]),
        p["residual_streams"],
        p["decode_position"],
        p["steps_per_decode_step"],
        p["prefill"]["rows"],
        p["prefill"]["steps"]
    );
    let slots = p["state_slots"].as_array().cloned().unwrap_or_default();
    let total: u64 = slots.iter().filter_map(|x| x["bytes"].as_u64()).sum();
    let _ = writeln!(o, "\nstate slots ({}, {} MiB at {} tokens):", slots.len(), total >> 20, p["cap_tokens"]);
    let mut i = 0;
    while i < slots.len() {
        let key = |x: &Value| format!("{} {} {} {}", s(&x["kind"]), x["rows"], x["width"], x["planes"]);
        let mut j = i;
        while j + 1 < slots.len() && key(&slots[j + 1]) == key(&slots[i]) && slots[i]["kind"] != "compressed_kv" && slots[i]["kind"] != "index_k" {
            j += 1;
        }
        let x = &slots[i];
        let what = match x["kind"].as_str() {
            Some("compressed_kv") | Some("index_k") => format!("ratio {} width {}  written by layer {}, read by {}", x["ratio"], x["width"], x["writer"], x["readers"]),
            _ => format!("{} rows x {} x {} planes {}", x["rows"], x["width"], x["planes"], s(&x["dtype"])),
        };
        let _ = writeln!(o, "  {:>3}..{:<3} {:<14} {what}", i, j, s(&x["kind"]));
        i = j + 1;
    }
    let layers = p["layers"].as_array().cloned().unwrap_or_default();
    let _ = writeln!(o, "\nlayers ({}):", layers.len());
    let key = |l: &Value| l["blocks"].as_array().map(|b| b.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(" + ")).unwrap_or_default();
    let mut i = 0;
    while i < layers.len() {
        let mut j = i;
        while j + 1 < layers.len() && key(&layers[j + 1]) == key(&layers[i]) {
            j += 1;
        }
        let _ = writeln!(o, "  {:>3}..{:<3} {}", i, j, key(&layers[i]));
        i = j + 1;
    }
    let _ = writeln!(o, "\n{:<17} {:<28} {:<15} {:>10}", "op", "kernel", "status", "calls/step");
    for op in p["ops"].as_array().cloned().unwrap_or_default() {
        let _ = writeln!(o, "{:<17} {:<28} {:<15} {:>10}", s(&op["op"]), s(&op["kernel"]), s(&op["status"]), op["calls_per_decode_step"]);
    }
    let _ = writeln!(o, "\nper-layer op sequence (decode step):");
    for st in p["sequence"].as_array().cloned().unwrap_or_default() {
        let names: Vec<String> = st["steps"].as_array().cloned().unwrap_or_default().iter().map(|x| s(&x["op"])).collect();
        let _ = writeln!(o, "  {:<9} ({:>2}) {}", s(&st["stage"]), names.len(), names.join(" "));
    }
    for m in p["mtp"].as_array().cloned().unwrap_or_default() {
        let _ = writeln!(o, "mtp block {}: {} experts top {} ({} found), parsed, not executed", s(&m["prefix"]), m["experts"], m["top_k"], m["expert_tensors_found"]);
    }
    o
}
