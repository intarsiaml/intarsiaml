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
use intarsia_core::recipe::ModelDims;
use intarsia_core::state::{SequenceState, StateSlot};
use intarsia_core::{DType, KernelImpl, Op, OpCall, Result, Telemetry};
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

fn load(backend: Arc<dyn Backend>, act: DType) -> Model {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let w = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let opts = LoadOptions { max_seq: 64, prefill_chunk: 8, act_dtype: Some(act), kv_dtype: Some(act), ..Default::default() };
    Model::load(backend, &recipe, &dims, &w, &opts, Arc::new(Telemetry::new())).unwrap()
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
