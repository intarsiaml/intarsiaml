//! Core-op fusion phase 1 at the model level (docs/results/2026-10-19-core-op-fusion-p1.md): the call sites of
//! `RmsNormAddMatmulFp8Pair` (P5, attention input norm + fp8 k / v) and `SigmoidMulMatmulFp8` (P4b, output gate + fp8
//! o_proj) run, give the bits of the unfused chain, and a backend without the new ops runs exactly the unfused chain.
//!
//! The tiny hybrid fixture (`tests/fixtures/tiny_qwen3_5`, 2 gated full-attention layers) is loaded with its attention
//! q / k / v / o projections quantised to fp8 (per-tensor scale, W8A16 and W8A8), on two cpu backends: the plain one
//! (it registers the fused ops: the oracle is the composition of the unfused ops) and one that drops the three new ops
//! (as CUDA and Metal do). Every logit row of a prefill + decode run must be bitwise equal, and the recorded launches must
//! be the expected ones: the fused ops in place of `RmsNormAdd` + `MatmulFp8Pair` and `SigmoidMul` + `MatmulFp8`, and none
//! of them on the backend without. (P2a, `SharedDownCombine`, has no MoE layer here; its block-level test is
//! `intarsia-backend-cpu/tests/quant_hf.rs::moe_block_fused_path_is_bitwise_the_unfused_one`.)

use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::quant::Fp8Act;
use intarsia_core::recipe::ModelDims;
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{DType, HostTensor, KernelImpl, Op, OpCall, Result, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const NEW_OPS: [Op; 3] = [Op::RmsNormAddMatmulFp8Pair, Op::SigmoidMulMatmulFp8, Op::SharedDownCombine];

/// The cpu backend, optionally without the phase-1 ops, recording the op of every launch.
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

/// The fixture with every full-attention projection replaced by its fp8 quantisation (scale = max|w| / 448, round to
/// nearest e4m3) and a static `input_scale` (used in W8A8, ignored in W8A16).
fn fp8_weights(input_scale: f32) -> MemWeights {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let mut w = MemWeights::new();
    for name in src.names() {
        let info = src.info(&name).unwrap();
        let t = HostTensor::new(info.dtype, info.shape.clone(), src.bytes(&name).unwrap().to_vec()).unwrap();
        let proj = name.contains(".self_attn.") && name.ends_with("_proj.weight");
        if !proj {
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

fn load(backend: Arc<dyn Backend>, w: &MemWeights, act: DType, fp8_act: Fp8Act) -> Model {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let opts = LoadOptions {
        max_seq: 64,
        prefill_chunk: 8,
        act_dtype: Some(act),
        kv_dtype: Some(act),
        fp8_act: Some(fp8_act),
        ..Default::default()
    };
    Model::load(backend, &recipe, &dims, w, &opts, Arc::new(Telemetry::new())).unwrap()
}

/// Logit rows of a 12-token prefill (two chunks) followed by 6 single-token decode steps, and the ops of the last step.
fn run(m: &Model, rec: &Rec) -> (Vec<Vec<f32>>, Vec<Op>) {
    let ids: Vec<u32> = (0..12u32).map(|i| (i * 7 + 3) % 96).collect();
    let mut st = m.new_state().unwrap();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, &ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    let mut last = Vec::new();
    for step in 0..6u32 {
        rec.ops.lock().unwrap().clear();
        let mut cb = |_p: usize, r: &[f32]| {
            rows.push(r.to_vec());
            Ok(())
        };
        m.forward(&mut st, &[(step * 11 + 5) % 96], HeadMode::AllLogits, Some(&mut cb)).unwrap();
        last = rec.ops.lock().unwrap().clone();
    }
    (rows, last)
}

fn count(ops: &[Op], op: Op) -> usize {
    ops.iter().filter(|&&o| o == op).count()
}

#[test]
fn fp8_attention_folds_are_bitwise_the_unfused_chain() {
    for (act, mode) in [(DType::F32, Fp8Act::None), (DType::Bf16, Fp8Act::None), (DType::Bf16, Fp8Act::Static)] {
        let w = fp8_weights(0.02);
        let label = format!("act {act} fp8 activations {mode:?}");
        let fused = Arc::new(Rec { inner: intarsia_backend_cpu::CpuBackend::arc(), drop_new: false, ops: Mutex::new(vec![]) });
        let plain = Arc::new(Rec { inner: intarsia_backend_cpu::CpuBackend::arc(), drop_new: true, ops: Mutex::new(vec![]) });
        let (mf, mp) = (load(fused.clone(), &w, act, mode), load(plain.clone(), &w, act, mode));
        let (rf, of) = run(&mf, &fused);
        let (rp, uf) = run(&mp, &plain);
        assert_eq!(rf.len(), 18, "{label}");
        for (i, (a, b)) in rf.iter().zip(&rp).enumerate() {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{label}: logit row {i} differs with the folds on");
        }
        // 2 full-attention layers, both after another layer (the norm in front of them is a residual add + norm)
        assert_eq!((count(&of, Op::RmsNormAddMatmulFp8Pair), count(&of, Op::SigmoidMulMatmulFp8)), (2, 2), "{label}: {of:?}");
        assert_eq!(count(&of, Op::SigmoidMul), 0, "{label}");
        // the backend without the new ops: none of them, the unfused chain in their place
        for op in NEW_OPS {
            assert_eq!(count(&uf, op), 0, "{label}: {op} on the backend without it");
        }
        assert_eq!(count(&uf, Op::SigmoidMul), 2, "{label}");
        assert_eq!(count(&uf, Op::MatmulFp8Pair), count(&of, Op::MatmulFp8Pair) + 2, "{label}: k / v pairs");
        assert_eq!(count(&uf, Op::RmsNormAdd), count(&of, Op::RmsNormAdd) + 2, "{label}: input norms");
        assert_eq!(count(&uf, Op::MatmulFp8), count(&of, Op::MatmulFp8) + 2, "{label}: o_proj");
        // one launch fewer per fold, nothing else moves
        assert_eq!(uf.len(), of.len() + 4, "{label}: launches per decode step {} (folds on) vs {} (off)", of.len(), uf.len());
    }
}
