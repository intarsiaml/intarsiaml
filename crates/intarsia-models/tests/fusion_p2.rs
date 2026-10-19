//! Core-op fusion phase 2 at the model level (docs/results/2026-10-19-core-op-fusion-p2.md): the call sites of
//! `GateSplitQkNormRopeKv` (P4a, the gated attention's split folded into the prologue) and `GdnConvDelta` (P3, the
//! GatedDeltaNet conv folded into the delta rule) run, give the bits of the unfused chain (logits, and the recurrent
//! state: conv state and delta-rule state, byte for byte), also across a snapshot / restore (the prefix-reuse path),
//! and a backend without the new ops runs exactly the unfused chain.
//!
//! The tiny hybrid fixture (`tests/fixtures/tiny_qwen3_5`: GatedDeltaNet layers and 2 gated full-attention layers) runs
//! on two cpu backends: the plain one (it registers the fused ops: the oracle is the composition of the unfused ops) and
//! one that drops the two new ops (as CUDA and Metal do).

use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::quant::Fp8Act;
use intarsia_core::recipe::ModelDims;
use intarsia_core::state::{SequenceState, StateSlot};
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{DType, HostTensor, KernelImpl, Op, OpCall, Result, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const NEW_OPS: [Op; 2] = [Op::GateSplitQkNormRopeKv, Op::GdnConvDelta];

/// The cpu backend, optionally without the phase-2 ops, recording the op of every launch.
struct Rec {
    inner: Arc<dyn Backend>,
    drop_new: bool,
    ops: Mutex<Vec<Op>>,
}

impl Backend for Rec {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn info(&self) -> DeviceInfo {
        self.inner.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> Result<Buffer> {
        self.inner.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> Result<()> {
        self.inner.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> Result<()> {
        self.inner.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> Result<()> {
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.inner.kernels().into_iter().filter(|k| !(self.drop_new && NEW_OPS.contains(&k.op))).collect()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        self.ops.lock().unwrap().push(call.op);
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.inner.synchronize(s)
    }
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn load_with(backend: Arc<dyn Backend>, w: &dyn WeightSource, act: DType, fp8_act: Option<Fp8Act>) -> Model {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let opts =
        LoadOptions { max_seq: 64, prefill_chunk: 8, act_dtype: Some(act), kv_dtype: Some(act), fp8_act, ..Default::default() };
    Model::load(backend, &recipe, &dims, w, &opts, Arc::new(Telemetry::new())).unwrap()
}

fn load(backend: Arc<dyn Backend>, act: DType) -> Model {
    load_with(backend, &SafeTensors::open_dir(&fixture_dir()).unwrap(), act, None)
}

/// The fixture with every full-attention projection replaced by its fp8 quantisation (tests/fusion_p1.rs's), so the
/// phase-1 attention folds (P5: input norm + k / v, P4b: output gate + o_proj) run in the same blocks as P4a.
fn fp8_weights(input_scale: f32) -> MemWeights {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let mut w = MemWeights::new();
    for name in src.names() {
        let info = src.info(&name).unwrap();
        let t = HostTensor::new(info.dtype, info.shape.clone(), src.bytes(&name).unwrap().to_vec()).unwrap();
        if !(name.contains(".self_attn.") && name.ends_with("_proj.weight")) {
            w.insert(&name, t);
            continue;
        }
        let v = t.to_f32();
        let scale = v.iter().fold(0f32, |m, x| m.max(x.abs())) / 448.0;
        let q: Vec<f32> = v.iter().map(|x| x / scale).collect();
        w.insert(&name, HostTensor::from_f32_as(DType::Fp8E4m3, &info.shape, &q));
        let base = name.trim_end_matches(".weight");
        w.insert(&format!("{base}.weight_scale"), HostTensor::from_f32(&[], &[scale]));
        w.insert(&format!("{base}.input_scale"), HostTensor::from_f32(&[], &[input_scale]));
    }
    w
}

/// Every recurrent slot's bytes (conv state, then the delta-rule state, per GatedDeltaNet layer).
fn recurrent_bytes(b: &dyn Backend, st: &SequenceState) -> Vec<u8> {
    let mut out = Vec::new();
    for s in &st.slots {
        if let StateSlot::Recurrent(r) = s {
            let mut v = vec![0u8; r.bytes];
            b.download(&r.buf, 0, &mut v, 0).unwrap();
            out.extend(v);
        }
    }
    out
}

struct Run {
    rows: Vec<Vec<f32>>,
    /// recurrent state after the prefill, after the first decode run, and after the restored second one
    states: Vec<Vec<u8>>,
    /// the ops of one decode step
    step_ops: Vec<Op>,
}

/// A 12-token prefill (chunks of 8 and 4), a snapshot, 5 decode steps, a roll back to the snapshot (the prefix-reuse
/// restore), then 5 other decode steps: every logit row, and the recurrent state at each of the three points.
fn run(m: &Model, rec: &Rec) -> Run {
    let b: &dyn Backend = &*m.backend;
    let ids: Vec<u32> = (0..12u32).map(|i| (i * 7 + 3) % 96).collect();
    let mut st = m.new_state().unwrap();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, &ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    let mut states = vec![recurrent_bytes(b, &st)];
    st.snapshot(b, 4).unwrap();
    let mut step_ops = Vec::new();
    for (round, mul) in [(0u32, 11u32), (1, 5)] {
        if round == 1 {
            st.truncate(12, b).unwrap();
            assert_eq!(recurrent_bytes(b, &st), states[0], "the restore brings back the snapshot's state");
        }
        for step in 0..5u32 {
            rec.ops.lock().unwrap().clear();
            let mut cb = |_p: usize, r: &[f32]| {
                rows.push(r.to_vec());
                Ok(())
            };
            m.forward(&mut st, &[(step * mul + 5 + round) % 96], HeadMode::AllLogits, Some(&mut cb)).unwrap();
            step_ops = rec.ops.lock().unwrap().clone();
        }
        states.push(recurrent_bytes(b, &st));
    }
    Run { rows, states, step_ops }
}

fn count(ops: &[Op], op: Op) -> usize {
    ops.iter().filter(|&&o| o == op).count()
}

#[test]
fn gate_split_and_gdn_conv_folds_are_bitwise_the_unfused_chain() {
    for act in [DType::F32, DType::Bf16] {
        let fused = Arc::new(Rec { inner: intarsia_backend_cpu::CpuBackend::arc(), drop_new: false, ops: Mutex::new(vec![]) });
        let plain = Arc::new(Rec { inner: intarsia_backend_cpu::CpuBackend::arc(), drop_new: true, ops: Mutex::new(vec![]) });
        let (mf, mp) = (load(fused.clone(), act), load(plain.clone(), act));
        let (f, p) = (run(&mf, &fused), run(&mp, &plain));
        assert_eq!(f.rows.len(), 22, "{act}");
        for (i, (a, b)) in f.rows.iter().zip(&p.rows).enumerate() {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{act}: logit row {i} differs with the folds on");
        }
        assert!(!f.states[0].is_empty(), "{act}: the fixture has recurrent layers");
        for (i, (a, b)) in f.states.iter().zip(&p.states).enumerate() {
            assert!(a == b, "{act}: recurrent state {i} (conv state + delta state) differs with the folds on");
        }
        assert!(f.states[1] != f.states[2], "{act}: the two decode runs after the restore took different tokens");
        let (of, uf) = (&f.step_ops, &p.step_ops);
        let gdn = count(uf, Op::GdnConv);
        assert!(gdn > 0, "{act}: {uf:?}");
        // P4a replaces GateSplit + QkNormRopeKv, so it runs where QkNormRopeKv ran: both gated layers at f32. At bf16 the
        // fixture's norm weights are not the activation dtype, which QkNormRopeKv's check refuses (it falls back to the six
        // unfused ops), and P4a's check is that check: it refuses too, and the chain is the same on both backends.
        let p4a = count(uf, Op::QkNormRopeKv);
        if act == DType::F32 {
            assert_eq!(p4a, 2, "{act}: {uf:?}");
        }
        // the folds in place of their chains: the gated attention layers, every GatedDeltaNet layer
        assert_eq!((count(of, Op::GateSplitQkNormRopeKv), count(of, Op::GdnConvDelta)), (p4a, gdn), "{act}: {of:?}");
        assert_eq!(count(of, Op::GateSplit), 2 - p4a, "{act}");
        for op in [Op::QkNormRopeKv, Op::GdnConv, Op::GdnDeltaRule] {
            assert_eq!(count(of, op), 0, "{act}: {op} with the folds on");
        }
        // the backend without the new ops: none of them, the unfused chain in their place
        for op in NEW_OPS {
            assert_eq!(count(uf, op), 0, "{act}: {op} on the backend without it");
        }
        assert_eq!((count(uf, Op::GateSplit), count(uf, Op::GdnDeltaRule)), (2, gdn), "{act}");
        // one launch fewer per fold, nothing else moves
        assert_eq!(uf.len(), of.len() + p4a + gdn, "{act}: launches per decode step {} (folds on) vs {} (off)", of.len(), uf.len());
    }
}

/// P4a in the same attention blocks as the phase-1 folds (fp8 attention projections, W8A16 and W8A8): P5 projects k / v
/// with the input norm before P4a reads them (`kv_done`), and P4b reads the gate P4a writes. Bitwise against the backend
/// without the phase-2 ops only (the phase-1 folds run on both sides).
#[test]
fn gate_split_fold_composes_with_the_phase1_attention_folds() {
    for mode in [Fp8Act::None, Fp8Act::Static] {
        let w = fp8_weights(0.02);
        let fused = Arc::new(Rec { inner: intarsia_backend_cpu::CpuBackend::arc(), drop_new: false, ops: Mutex::new(vec![]) });
        let plain = Arc::new(Rec { inner: intarsia_backend_cpu::CpuBackend::arc(), drop_new: true, ops: Mutex::new(vec![]) });
        let (mf, mp) = (load_with(fused.clone(), &w, DType::F32, Some(mode)), load_with(plain.clone(), &w, DType::F32, Some(mode)));
        let (f, p) = (run(&mf, &fused), run(&mp, &plain));
        for (i, (a, b)) in f.rows.iter().zip(&p.rows).enumerate() {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{mode:?}: logit row {i} differs with P4a on");
        }
        assert!(f.states == p.states, "{mode:?}: recurrent state differs");
        let (of, uf) = (&f.step_ops, &p.step_ops);
        // all three attention folds in both gated layers
        for op in [Op::GateSplitQkNormRopeKv, Op::RmsNormAddMatmulFp8Pair, Op::SigmoidMulMatmulFp8] {
            assert_eq!(count(of, op), 2, "{mode:?}: {op}: {of:?}");
        }
        assert_eq!((count(uf, Op::RmsNormAddMatmulFp8Pair), count(uf, Op::SigmoidMulMatmulFp8)), (2, 2), "{mode:?}: {uf:?}");
        assert_eq!((count(uf, Op::GateSplit), count(uf, Op::QkNormRopeKv)), (2, 2), "{mode:?}");
    }
}
