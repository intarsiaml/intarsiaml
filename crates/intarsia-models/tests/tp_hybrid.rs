//! Tensor parallelism of the Qwen3.5/3.6 hybrid MoE family (docs/design/35b-tp2.md) on the cpu backend, the Mac gate for
//! the 35B sharding: two shards joined by the host-staged communicator must compute what one unsharded model computes
//! (summation order aside), with the 35B's weight formats where it has them.
//!
//! The model is the tiny hybrid fixture (`tests/fixtures/tiny_qwen3_5`: 4 GatedDeltaNet and 2 gated full-attention layers)
//! under the `qwen3_5_moe` recipe, every dense FFN replaced by an MoE block (8 NVFP4 experts, top-2, an NVFP4 shared
//! expert, bf16 router and shared-expert gate), and the attention q/k/v/o and GDN in_qkv / in_z / out projections stored
//! as FP8 e4m3 with a per-tensor scale, as in the 35B checkpoint. So every sharded loader the 35B uses runs here: FP8
//! rows / columns / q|k|v row groups, NVFP4 rows and columns (whole 16-element blocks), stacked experts, the conv taps,
//! the 1-D A_log / dt_bias slices.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::dtype::f32_to_fp8_e4m3;
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::quant::{nvfp4, Fp8Act};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{DType, HostTensor, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// `INTARSIA_TP_TEST_BUG` is read at load: every test holds this while it runs so none sees another's setting.
static ENV: Mutex<()> = Mutex::new(());

fn env() -> MutexGuard<'static, ()> {
    ENV.lock().unwrap_or_else(|e| e.into_inner())
}

const E: usize = 8;
const TOP_K: usize = 2;
const I: usize = 32;
const IS: usize = 32;
const H: usize = 64;
const LAYERS: usize = 6;
const V: u32 = 96;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn put_nvfp4(w: &mut MemWeights, rng: &mut SplitMix64, prefix: &str, n: usize, k: usize) {
    let q = nvfp4::random_rows(rng, n, k);
    w.insert(&format!("{prefix}.weight"), q.parts[0].clone());
    w.insert(&format!("{prefix}.weight_scale"), q.parts[1].clone());
    w.insert(&format!("{prefix}.weight_scale_2"), q.parts[2].clone());
}

fn bf16(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), scale))
}

/// `{prefix}.weight` as e4m3 with a per-tensor `{prefix}.weight_scale` (amax / 448).
fn to_fp8(w: &mut MemWeights, prefix: &str) {
    let name = format!("{prefix}.weight");
    let info = w.info(&name).unwrap_or_else(|| panic!("{name} missing"));
    let t = HostTensor::new(info.dtype, info.shape.clone(), w.bytes(&name).unwrap().to_vec()).unwrap();
    let v = t.to_f32();
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let ws = if amax > 0.0 { amax / 448.0 } else { 1.0 };
    let data: Vec<u8> = v.iter().map(|x| f32_to_fp8_e4m3(x / ws)).collect();
    w.insert(&name, HostTensor::new(DType::Fp8E4m3, info.shape, data).unwrap());
    w.insert(&format!("{prefix}.weight_scale"), HostTensor::from_f32(&[], &[ws]));
}

fn weights() -> MemWeights {
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
    let mut rng = SplitMix64::new(0x7b_35);
    for l in 0..LAYERS {
        let p = format!("model.language_model.layers.{l}");
        let attn = [".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"];
        let gdn = [".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.out_proj"];
        for s in attn.iter().chain(&gdn) {
            if w.info(&format!("{p}{s}.weight")).is_some() {
                to_fp8(&mut w, &format!("{p}{s}"));
            }
        }
        let p = format!("{p}.mlp");
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

fn dims() -> (intarsia_core::recipe::Recipe, ModelDims) {
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
    (recipe, dims)
}

fn opts(act: DType) -> LoadOptions {
    LoadOptions { max_seq: 64, prefill_chunk: 8, act_dtype: Some(act), kv_dtype: Some(act), fp8_act: Some(Fp8Act::None), ..Default::default() }
}

fn load(world: usize, w: &MemWeights, act: DType) -> Model {
    let (recipe, dims) = dims();
    let backends: Vec<Arc<dyn Backend>> = (0..world).map(|_| CpuBackend::arc()).collect();
    if world == 1 {
        return Model::load(backends[0].clone(), &recipe, &dims, w, &opts(act), Arc::new(Telemetry::new())).unwrap();
    }
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    Model::load_tp(backends, comms, &recipe, &dims, w, &opts(act), Arc::new(Telemetry::new())).unwrap()
}

fn prompt() -> Vec<u32> {
    (0..12u32).map(|i| (i * 7 + 3) % V).collect()
}

/// Logit rows of a 12-token prefill (two chunks of 8 and 4) followed by 6 single-token decode steps.
fn run(m: &Model) -> Vec<Vec<f32>> {
    let mut st = m.new_state().unwrap();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, &prompt(), HeadMode::AllLogits, Some(&mut cb)).unwrap();
    for step in 0..6u32 {
        let mut cb = |_p: usize, r: &[f32]| {
            rows.push(r.to_vec());
            Ok(())
        };
        m.forward(&mut st, &[(step * 11 + 5) % V], HeadMode::AllLogits, Some(&mut cb)).unwrap();
    }
    rows
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
fn hybrid_moe_tp2_matches_tp1_f32() {
    let _g = env();
    let w = weights();
    let (one, two) = (load(1, &w, DType::F32), load(2, &w, DType::F32));
    assert_eq!(two.tp_world(), 2);
    let (a, b) = (run(&two), run(&one));
    assert_eq!((a.len(), b.len()), (18, 18));
    let mut worst = 0f64;
    for (p, (x, y)) in a.iter().zip(&b).enumerate() {
        let r = rel_l2(x, y);
        worst = worst.max(r);
        assert!(r < 1e-5, "row {p}: rel_l2 {r}");
        assert_eq!(argmax(x), argmax(y), "row {p}");
    }
    eprintln!("hybrid MoE tp=2 vs tp=1 (f32): worst rel_l2 {worst:.2e}");
}

/// Row-local fusion under TP (`LoadOptions::tp_fusion`): the fused decode ops run on each rank's shard and the result is
/// still tp=1's up to summation order. Rank 0's telemetry shows which fused ops ran; the GatedDeltaNet out-projection
/// fold (`gated_rmsnorm_matmul_fp8`, it would swallow the all-reduce) must not be among them.
#[test]
fn hybrid_moe_tp2_row_local_fusion_matches_tp1_f32() {
    let _g = env();
    let w = weights();
    let one = load(1, &w, DType::F32);
    let (recipe, dims) = dims();
    let backends: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    let tel = Arc::new(Telemetry::new());
    let fused =
        Model::load_tp(backends, comms, &recipe, &dims, &w, &LoadOptions { tp_fusion: true, ..opts(DType::F32) }, tel.clone())
            .unwrap();
    let (a, b) = (run(&fused), run(&one));
    let mut worst = 0f64;
    for (p, (x, y)) in a.iter().zip(&b).enumerate() {
        let r = rel_l2(x, y);
        worst = worst.max(r);
        assert!(r < 1e-5, "row {p}: rel_l2 {r}");
        assert_eq!(argmax(x), argmax(y), "row {p}");
    }
    let ops = tel.snapshot().ops;
    let ran: Vec<&str> = ["matmul_pair", "matmul_fp8_pair", "rmsnorm_add_matmul_pair", "rmsnorm_add_matmul_fp8_pair", "gdn_conv_delta",
        "gated_rmsnorm", "qk_norm_rope_kv", "gate_split_qk_norm_rope_kv", "matmul_nvfp4_glu", "moe_block"]
        .into_iter()
        .filter(|k| ops.get(*k).is_some_and(|s| s.calls > 0))
        .collect();
    eprintln!("hybrid MoE tp=2 fused vs tp=1 (f32): worst rel_l2 {worst:.2e}; fused ops on rank 0: {ran:?}");
    assert!(!ran.is_empty(), "no fused op ran under tp_fusion");
    assert!(ops.get("gated_rmsnorm_matmul_fp8").is_none_or(|s| s.calls == 0), "the out-projection fold ran under TP");
}

/// The same comparison separates a 1% partial-scale bug on rank 1 (`INTARSIA_TP_TEST_BUG=scale:1.01`, every all-reduce of
/// attention, GatedDeltaNet and MoE) from summation order by orders of magnitude: the guard the GPU gates lack
/// (docs/accuracy.md, "TP self-golden budget").
#[test]
fn hybrid_moe_tp2_catches_a_one_percent_partial_scale_bug() {
    let _g = env();
    let w = weights();
    let one = load(1, &w, DType::F32);
    std::env::set_var("INTARSIA_TP_TEST_BUG", "scale:1.01");
    let bad = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load(2, &w, DType::F32)));
    std::env::remove_var("INTARSIA_TP_TEST_BUG");
    let bad = bad.unwrap();
    let worst = run(&bad).iter().zip(&run(&one)).map(|(x, y)| rel_l2(x, y)).fold(0f64, f64::max);
    eprintln!("hybrid MoE tp=2 with scale:1.01 on rank 1 vs tp=1 (f32): worst rel_l2 {worst:.2e}");
    assert!(worst > 1e-4, "a 1% bug moved the logits by only {worst:.2e} (clean bound 1e-5)");
}

/// The 35B split per rank: half the attention / KV / GDN heads (and recurrent state), half of every expert's width, and
/// only the slices on the device.
#[test]
fn hybrid_moe_tp2_rank_shapes_and_bytes() {
    let _g = env();
    let w = weights();
    let (one, two) = (load(1, &w, DType::F32), load(2, &w, DType::F32));
    let (r0, r1) = (two.rank_model(0).unwrap(), two.rank_model(1).unwrap());
    for r in [r0, r1] {
        let l = r.ldims.linear.unwrap();
        let g = one.ldims.linear.unwrap();
        assert_eq!((l.k_heads, l.v_heads), (g.k_heads / 2, g.v_heads / 2));
        let (m, gm) = (r.ldims.moe.as_ref().unwrap(), one.ldims.moe.as_ref().unwrap());
        assert_eq!((m.inter, m.shared_inter, m.experts, m.top_k), (I / 2, IS / 2, E, TOP_K));
        assert_eq!((gm.inter, gm.shared_inter), (I, IS));
        assert_eq!((r.ldims.n_heads, r.ldims.n_kv_heads), (one.dims.n_heads / 2, one.dims.n_kv_heads / 2));
        assert_eq!(r.dims.hidden, one.dims.hidden);
    }
    // rank 1 holds no head; both hold less than the whole model, and the two together hold the whole model once plus the
    // replicated parts (embedding, norms, router, shared-expert gate, gated-norm weights)
    assert!(r1.weights_bytes < r0.weights_bytes, "{} vs {}", r1.weights_bytes, r0.weights_bytes);
    assert!(r0.weights_bytes < one.weights_bytes, "{} vs {}", r0.weights_bytes, one.weights_bytes);
    // recurrent state follows the heads: per rank half of tp=1's (up to the 256-byte alignment of the delta-rule state
    // inside each of the 4 GatedDeltaNet slots)
    let (s1, s2) = (one.new_state().unwrap(), two.new_state().unwrap());
    assert_eq!(s2.peers.len(), 1);
    let (half, whole) = (s2.recurrent_bytes(), s1.recurrent_bytes());
    assert!(half * 2 >= whole - 4 * 256 && half * 2 <= whole + 4 * 256, "{half} per rank vs {whole}");
    assert_eq!(s2.peers[0].st.recurrent_bytes(), s2.recurrent_bytes());
}

/// Prefix reuse under TP2: a snapshot of the recurrent state on both ranks, a rollback to it and the same continuation must
/// reproduce the first continuation bit for bit (bf16 activations, the serving dtype).
#[test]
fn hybrid_moe_tp2_snapshot_rollback_is_exact() {
    let _g = env();
    let w = weights();
    let m = load(2, &w, DType::Bf16);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &prompt(), HeadMode::LastLogits, None).unwrap();
    st.snapshot(&*m.backend, 4).unwrap();
    assert_eq!(st.restorable_len(12), 12);
    let cont: Vec<u32> = vec![17, 4, 61, 33];
    let mut first = Vec::new();
    let mut cb = |_p: usize, r: &[f32]| {
        first.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, &cont, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    st.truncate(12, &*m.backend).unwrap();
    assert_eq!(st.peers[0].st.len, 12, "the peer rank must roll back with the primary");
    let mut again = Vec::new();
    let mut cb = |_p: usize, r: &[f32]| {
        again.push(r.to_vec());
        Ok(())
    };
    m.forward(&mut st, &cont, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    assert_eq!(first.len(), 4);
    for (i, (a, b)) in first.iter().zip(&again).enumerate() {
        assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "row {i} differs after the rollback");
    }
}

/// Rank 0 here, rank 1 behind a real (localhost) TCP connection running `tp_remote::serve` on another thread, the two
/// joined by the host communicator: the cross-process protocol must give the in-process tp=2 bits exactly (same shards,
/// same sums), through prefill chunks, decode, a snapshot and a rollback. Returns the rank-0 model and the server thread.
fn load_remote(w: &MemWeights, act: DType) -> (Model, std::thread::JoinHandle<u64>) {
    use intarsia_core::model::TpRank;
    use intarsia_core::tp_remote;
    let (recipe, dims) = dims();
    let backends: Vec<Arc<dyn Backend>> = (0..2).map(|_| CpuBackend::arc()).collect();
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (b1, c1, r1, d1, w1) = (backends[1].clone(), comms[1].clone(), recipe.clone(), dims.clone(), weights_clone(w));
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut m = Model::load_shard(b1, &r1, &d1, &w1, &opts(act), Arc::new(Telemetry::new()), TpRank { rank: 1, world: 2, comm: Some(c1) })
            .unwrap();
        let cap = tp_remote::handshake(&mut s, "fixture", m.plan.kv_cap_tokens).unwrap();
        m.clamp_kv_cap(cap).unwrap();
        tp_remote::serve(&m, s).unwrap()
    });
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    let tp = TpRank { rank: 0, world: 2, comm: Some(comms[0].clone()) };
    let mut m = Model::load_shard(backends[0].clone(), &recipe, &dims, w, &opts(act), Arc::new(Telemetry::new()), tp).unwrap();
    let cap = tp_remote::handshake(&mut s, "fixture", m.plan.kv_cap_tokens).unwrap();
    m.clamp_kv_cap(cap).unwrap();
    m.attach_remote(tp_remote::RemotePeer::new(s, comms[0].clone()).unwrap()).unwrap();
    (m, server)
}

fn weights_clone(w: &MemWeights) -> MemWeights {
    let mut c = MemWeights::new();
    for n in w.names() {
        let i = w.info(&n).unwrap();
        c.insert(&n, HostTensor::new(i.dtype, i.shape, w.bytes(&n).unwrap().to_vec()).unwrap());
    }
    c
}

#[test]
fn hybrid_moe_tp2_remote_rank_is_bitwise_the_in_process_group() {
    let _g = env();
    let w = weights();
    for act in [DType::F32, DType::Bf16] {
        let local = load(2, &w, act);
        let (remote, server) = load_remote(&w, act);
        let (a, b) = (run(&remote), run(&local));
        assert_eq!(a.len(), 18);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "act {act}: row {i} differs from the in-process group");
        }
        // prefix reuse through the connection: snapshot, continue, roll back, continue again
        let mut st = remote.new_state().unwrap();
        remote.forward(&mut st, &prompt(), HeadMode::LastLogits, None).unwrap();
        st.snapshot(&*remote.backend, 4).unwrap();
        let cont: Vec<u32> = vec![17, 4, 61, 33];
        let mut rows = [Vec::new(), Vec::new()];
        for r in rows.iter_mut() {
            let mut cb = |_p: usize, x: &[f32]| {
                r.push(x.to_vec());
                Ok(())
            };
            remote.forward(&mut st, &cont, HeadMode::AllLogits, Some(&mut cb)).unwrap();
            st.truncate(12, &*remote.backend).unwrap();
        }
        assert!(rows[0].iter().flatten().zip(rows[1].iter().flatten()).all(|(p, q)| p.to_bits() == q.to_bits()), "act {act}: rollback");
        // a snapshot right after a rollback, keeping one: the remote rank must have rolled back before it snapshots
        // (else its only snapshot is past the rollback point and the next rollback cannot restore)
        let mut again = Vec::new();
        for _ in 0..2 {
            st.snapshot(&*remote.backend, 1).unwrap();
            let mut r = Vec::new();
            let mut cb = |_p: usize, x: &[f32]| {
                r.push(x.to_vec());
                Ok(())
            };
            remote.forward(&mut st, &cont, HeadMode::AllLogits, Some(&mut cb)).unwrap();
            st.truncate(12, &*remote.backend).unwrap();
            again.push(r);
        }
        for r in &again {
            assert!(r.iter().flatten().zip(rows[0].iter().flatten()).all(|(p, q)| p.to_bits() == q.to_bits()), "act {act}: snapshot after rollback");
        }
        drop(st);
        drop(remote); // closes the connection: the server loop returns
        assert_eq!(server.join().unwrap(), 7 + 5, "act {act}: forward passes served (7 in run, 5 here)");
    }
}

/// tp=1 is untouched by the sharding code: the plain load and a `load_shard` at world 1 give the same bits.
#[test]
fn hybrid_moe_tp1_unchanged_by_shard_path() {
    let _g = env();
    let w = weights();
    let (recipe, dims) = dims();
    let a = load(1, &w, DType::Bf16);
    let b = Model::load_shard(
        CpuBackend::arc(),
        &recipe,
        &dims,
        &w,
        &opts(DType::Bf16),
        Arc::new(Telemetry::new()),
        intarsia_core::model::TpRank::single(),
    )
    .unwrap();
    let (ra, rb) = (run(&a), run(&b));
    for (i, (x, y)) in ra.iter().zip(&rb).enumerate() {
        assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "row {i}");
    }
}
