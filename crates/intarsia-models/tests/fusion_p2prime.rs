//! Core-op fusion P2' at the model level (docs/results/2026-10-19-p2prime-moeblock.md): the call site of `MoeBlock` in
//! `MoeWeights::forward_folded` (the whole MoE block after the router logits as one op) runs, gives the bits of the
//! unfused chain, and a backend without the op runs exactly the chain it ran before.
//!
//! The tiny hybrid fixture (`tests/fixtures/tiny_qwen3_5`: 6 layers, 4 GatedDeltaNet-conv-attention and 2 gated full
//! attention) is loaded with the `qwen3_5_moe` recipe: every layer's dense FFN is replaced by a sparse MoE block (8 NVFP4
//! experts, top-2, a shared NVFP4 expert, a bf16 router and shared-expert gate; random weights). Three cpu backends:
//! the plain one (it registers every fused op: the oracle of `MoeBlock` is the composition of the unfused ops), one without
//! `MoeBlock` (as CUDA and Metal are today: the P2a chain), and one without `MoeBlock`, `SharedDownCombine` and
//! `MatmulNvfp4Glu` (the chain before any MoE fusion). Every logit row of a prefill + decode run must be bitwise equal
//! across the three, and the recorded launches must be the expected ones.

use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::quant::{nvfp4, Fp8Act};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{DType, HostTensor, KernelImpl, Op, OpCall, Result, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const E: usize = 8;
const TOP_K: usize = 2;
const I: usize = 32;
const IS: usize = 32;
const H: usize = 64;
const LAYERS: usize = 6;

/// The cpu backend without the ops in `drop`, recording the op of every launch.
struct Rec {
    inner: Arc<dyn Backend>,
    drop: Vec<Op>,
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
        self.inner.kernels().into_iter().filter(|k| !self.drop.contains(&k.op)).collect()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        self.ops.lock().unwrap().push(call.op);
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.inner.synchronize(s)
    }
}

fn rec(drop: &[Op]) -> Arc<Rec> {
    Arc::new(Rec { inner: intarsia_backend_cpu::CpuBackend::arc(), drop: drop.to_vec(), ops: Mutex::new(vec![]) })
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

/// `{prefix}.weight` / `.weight_scale` / `.weight_scale_2` of a random NVFP4 `[n, k]` matrix.
fn put_nvfp4(w: &mut MemWeights, rng: &mut SplitMix64, prefix: &str, n: usize, k: usize) {
    let q = nvfp4::random_rows(rng, n, k);
    w.insert(&format!("{prefix}.weight"), q.parts[0].clone());
    w.insert(&format!("{prefix}.weight_scale"), q.parts[1].clone());
    w.insert(&format!("{prefix}.weight_scale_2"), q.parts[2].clone());
}

fn bf16(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), scale))
}

/// The fixture under the `qwen3_5_moe` recipe's names (`model.language_model.*`) with every layer's dense FFN replaced by an
/// NVFP4 MoE block.
fn moe_weights() -> MemWeights {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let mut w = MemWeights::new();
    for name in src.names() {
        if name.contains(".mlp.") {
            continue;
        }
        let info = src.info(&name).unwrap();
        let t = HostTensor::new(info.dtype, info.shape.clone(), src.bytes(&name).unwrap().to_vec()).unwrap();
        let renamed = if let Some(rest) = name.strip_prefix("model.") { format!("model.language_model.{rest}") } else { name.clone() };
        w.insert(&renamed, t);
    }
    let mut rng = SplitMix64::new(0x70_2f);
    for l in 0..LAYERS {
        let p = format!("model.language_model.layers.{l}.mlp");
        w.insert(&format!("{p}.gate.weight"), bf16(&mut rng, &[E, H], 0.3));
        w.insert(&format!("{p}.shared_expert_gate.weight"), bf16(&mut rng, &[1, H], 0.3));
        for x in 0..E {
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.gate_proj"), I, H);
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.up_proj"), I, H);
            put_nvfp4(&mut w, &mut rng, &format!("{p}.experts.{x}.down_proj"), H, I);
        }
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.gate_proj"), IS, H);
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.up_proj"), IS, H);
        put_nvfp4(&mut w, &mut rng, &format!("{p}.shared_expert.down_proj"), H, IS);
    }
    w
}

fn load(backend: Arc<dyn Backend>, w: &MemWeights, act: DType) -> Model {
    load_on(vec![backend], w, act)
}

/// `load` on one backend, or tensor-parallel over several (`HostCommGroup`).
fn load_on(backends: Vec<Arc<dyn Backend>>, w: &MemWeights, act: DType) -> Model {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_moe.toml")).unwrap();
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    cfg["architectures"] = serde_json::json!(["Qwen3_5MoeForCausalLM"]);
    cfg["num_experts"] = E.into();
    cfg["num_experts_per_tok"] = TOP_K.into();
    cfg["moe_intermediate_size"] = I.into();
    cfg["shared_expert_intermediate_size"] = IS.into();
    cfg["norm_topk_prob"] = true.into();
    cfg.as_object_mut().unwrap().remove("intermediate_size");
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert_eq!(dims.moe.as_ref().map(|m| (m.experts, m.top_k, m.inter, m.shared_inter)), Some((E, TOP_K, I, IS)));
    let opts = LoadOptions {
        max_seq: 64,
        prefill_chunk: 8,
        act_dtype: Some(act),
        kv_dtype: Some(act),
        fp8_act: Some(Fp8Act::None),
        ..Default::default()
    };
    if backends.len() == 1 {
        return Model::load(backends[0].clone(), &recipe, &dims, w, &opts, Arc::new(Telemetry::new())).unwrap();
    }
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    Model::load_tp(backends, comms, &recipe, &dims, w, &opts, Arc::new(Telemetry::new())).unwrap()
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
fn moe_block_is_bitwise_the_unfused_chain() {
    for act in [DType::F32, DType::Bf16] {
        let w = moe_weights();
        let label = format!("act {act}");
        let fused = rec(&[]);
        let p2a = rec(&[Op::MoeBlock]);
        let plain = rec(&[Op::MoeBlock, Op::SharedDownCombine, Op::MatmulNvfp4Glu]);
        let (mf, m2, mp) = (load(fused.clone(), &w, act), load(p2a.clone(), &w, act), load(plain.clone(), &w, act));
        let (rf, of) = run(&mf, &fused);
        let (r2, o2) = run(&m2, &p2a);
        let (rp, op) = run(&mp, &plain);
        assert_eq!(rf.len(), 18, "{label}");
        for (i, ((a, b), c)) in rf.iter().zip(&r2).zip(&rp).enumerate() {
            assert!(a.iter().all(|x| x.is_finite()), "{label}: logit row {i} is not finite");
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{label}: logit row {i} differs from the P2a chain");
            assert!(a.iter().zip(c).all(|(x, y)| x.to_bits() == y.to_bits()), "{label}: logit row {i} differs from the unfused chain");
        }
        // every layer is a MoE layer: one MoeBlock per layer per decode step, and none of the ops it replaces
        assert_eq!(count(&of, Op::MoeBlock), LAYERS, "{label}: {of:?}");
        for op in [Op::MoeTopk, Op::MoeExperts, Op::MatmulNvfp4Glu, Op::SharedDownCombine, Op::MoeCombine, Op::SiluGlu] {
            assert_eq!(count(&of, op), 0, "{label}: {op} next to MoeBlock");
        }
        // the backend without it runs the chain it ran before: the P2a one ...
        assert_eq!(count(&o2, Op::MoeBlock), 0, "{label}");
        for op in [Op::MoeTopk, Op::MoeExperts, Op::MatmulNvfp4Glu, Op::SharedDownCombine] {
            assert_eq!(count(&o2, op), LAYERS, "{label}: {op} on the backend without MoeBlock");
        }
        assert_eq!((count(&o2, Op::MoeCombine), count(&o2, Op::SiluGlu)), (0, 0), "{label}");
        // ... and, without the P2a folds either, the plain one: separate gate / up / down projections, silu_glu, combine
        assert_eq!(count(&op, Op::MoeBlock) + count(&op, Op::SharedDownCombine) + count(&op, Op::MatmulNvfp4Glu), 0, "{label}");
        for (what, got, want) in [
            ("MoeTopk", count(&op, Op::MoeTopk), LAYERS),
            ("MoeExperts", count(&op, Op::MoeExperts), LAYERS),
            ("SiluGlu", count(&op, Op::SiluGlu), LAYERS),
            ("MoeCombine", count(&op, Op::MoeCombine), LAYERS),
            ("shared gate / up / down MatmulNvfp4", count(&op, Op::MatmulNvfp4), 3 * LAYERS),
        ] {
            assert_eq!(got, want, "{label}: {what} on the unfused chain");
        }
        // 1 launch against the 4 (topk, experts, glu, down+combine) the P2a chain runs per layer
        assert_eq!(o2.len() - of.len(), 3 * LAYERS, "{label}: launches per decode step {} (MoeBlock) vs {} (P2a chain)", of.len(), o2.len());
    }
}

/// TP > 1: the fold is off. Two layers of protection, both pinned here:
/// * `Model::fused_ok` refuses every fused op when `tp.world > 1` (the routed and shared outputs are partial sums before the
///   all-reduce, so a fused combine would be wrong under a partition);
/// * a TP load of this model family is refused outright today (GatedDeltaNet blocks are not sharded), so no TP plan can
///   hold `MoeBlock` at all. If that changes, this test fails and the TP2 run below must be turned into a launch count.
#[test]
fn moe_block_is_off_under_tp2() {
    let w = moe_weights();
    let (r0, r1) = (rec(&[]), rec(&[]));
    assert!(r0.kernels().iter().any(|k| k.op == Op::MoeBlock), "the cpu backend must offer MoeBlock for this test to mean anything");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        load_on(vec![r0.clone() as Arc<dyn Backend>, r1.clone() as Arc<dyn Backend>], &w, DType::F32)
    }));
    assert!(r.is_err(), "a TP2 load of the hybrid MoE model now succeeds: replace this with a MoeBlock launch count of 0 per rank");
    assert!(r0.ops.lock().unwrap().iter().all(|&o| o != Op::MoeBlock) && r1.ops.lock().unwrap().iter().all(|&o| o != Op::MoeBlock));
}
