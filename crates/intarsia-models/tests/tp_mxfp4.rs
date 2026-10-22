//! Tensor parallelism of the hybrid decoder with MXFP4 projections (Qwen3.8-27B's layout, recipes/qwen3_5_vl.toml) on
//! the cpu backend, the Mac gate for lane T2's MXFP4 sharding: two shards joined by the host-staged communicator must
//! compute what the unsharded model computes, up to the order of the f32 sums.
//!
//! The model is the tiny hybrid fixture (`tests/fixtures/tiny_qwen3_5`: 4 GatedDeltaNet and 2 gated full-attention layers,
//! each with a dense FFN) in the `model.language_model.*` layout, every projection quantised to MXFP4 as in the 27B
//! (attention q / k / v / o, GDN in_proj_qkv / z / a / b and out_proj, FFN gate / up / down). The fixture's FFN is 96 wide,
//! which at tp=2 would cut down_proj's 32-element groups (48 columns per rank), so the FFN is replaced by a random one of
//! width 128 (64 per rank, two whole groups). Every sharded MXFP4 loader the 27B uses runs here: rows, the q|k|v row
//! groups, columns.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::quant::{mxfp4::Mxfp4, Mxfp4Act, QuantCfg, QuantFormat};
use intarsia_core::recipe::ModelDims;
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, HostTensor, Telemetry};
use intarsia_core::weights::WeightSource;
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// `INTARSIA_TP_TEST_BUG` is read at load: every test holds this while it runs so none sees another's setting.
static ENV: Mutex<()> = Mutex::new(());

fn env() -> MutexGuard<'static, ()> {
    ENV.lock().unwrap_or_else(|e| e.into_inner())
}

const I: usize = 128;
const H: usize = 64;
const V: u32 = 96;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

const PROJ: [&str; 9] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
];

fn put_mx(w: &mut MemWeights, module: &str, t: &HostTensor) {
    let q = Mxfp4.quantize(t, &QuantCfg::default()).expect("K % 32 == 0");
    w.insert(&format!("{module}.weight"), q.parts[0].clone());
    w.insert(&format!("{module}.weight_scale"), q.parts[1].clone());
}

/// The MXFP4 checkpoint (VL layout, FFN width `I`).
fn weights() -> MemWeights {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    let mut w = MemWeights::new();
    let mut nq = 0;
    for name in src.names() {
        if name.contains(".mlp.") {
            continue;
        }
        let info = src.info(&name).unwrap();
        let t = HostTensor::new(info.dtype, info.shape.clone(), src.bytes(&name).unwrap().to_vec()).unwrap();
        let vl = name.strip_prefix("model.").map(|r| format!("model.language_model.{r}")).unwrap_or(name.clone());
        let module = vl.strip_suffix(".weight").unwrap_or(&vl).to_string();
        if PROJ.iter().any(|p| module.ends_with(p)) {
            put_mx(&mut w, &module, &t);
            nq += 1;
        } else {
            w.insert(&vl, t);
        }
    }
    assert_eq!(nq, 4 * 5 + 2 * 4, "every attention and GDN projection is MXFP4");
    let mut rng = SplitMix64::new(0x7227);
    for l in 0..6 {
        let p = format!("model.language_model.layers.{l}.mlp");
        let g = |rng: &mut SplitMix64, n: usize, k: usize| HostTensor::from_f32(&[n, k], &rng.normal_vec(n * k, 0.15));
        put_mx(&mut w, &format!("{p}.gate_proj"), &g(&mut rng, I, H));
        put_mx(&mut w, &format!("{p}.up_proj"), &g(&mut rng, I, H));
        put_mx(&mut w, &format!("{p}.down_proj"), &g(&mut rng, H, I));
    }
    w
}

fn dims() -> (intarsia_core::recipe::Recipe, ModelDims) {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_vl.toml")).unwrap();
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    cfg["intermediate_size"] = I.into();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    (recipe, dims)
}

fn opts(act: Mxfp4Act, fusion: bool) -> LoadOptions {
    LoadOptions {
        max_seq: 64,
        prefill_chunk: 8,
        act_dtype: Some(DType::F32),
        kv_dtype: Some(DType::F32),
        mxfp4_act: Some(act),
        tp_fusion: fusion,
        ..Default::default()
    }
}

fn load(world: usize, w: &MemWeights, act: Mxfp4Act, fusion: bool, tel: Arc<Telemetry>) -> Model {
    let (recipe, dims) = dims();
    let backends: Vec<Arc<dyn Backend>> = (0..world).map(|_| CpuBackend::arc()).collect();
    if world == 1 {
        return Model::load(backends[0].clone(), &recipe, &dims, w, &opts(act, fusion), tel).unwrap();
    }
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    Model::load_tp(backends, comms, &recipe, &dims, w, &opts(act, fusion), tel).unwrap()
}

fn prompt() -> Vec<u32> {
    (0..12u32).map(|i| (i * 7 + 3) % V).collect()
}

/// Logit rows of a 12-token prefill (chunks of 8 and 4), then 6 single-token decode steps.
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
    (num / den.max(1e-30)).sqrt()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

/// Worst rel l2 of `a` against `b` over all rows, asserting finiteness and equal greedy tokens.
fn compare(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) -> f64 {
    assert_eq!((a.len(), b.len()), (18, 18), "{what}");
    let mut worst = 0f64;
    for (p, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(x.iter().all(|v| v.is_finite()), "{what}: row {p} not finite");
        worst = worst.max(rel_l2(x, y));
        assert_eq!(argmax(x), argmax(y), "{what}: row {p} greedy token");
    }
    worst
}

/// W4A16: the MXFP4 matmuls see the same 16-bit activations on every rank, so tp=2 is tp=1 up to the f32 sum order.
#[test]
fn mxfp4_hybrid_tp2_matches_tp1_w4a16() {
    let _g = env();
    let w = weights();
    let t = || Arc::new(Telemetry::new());
    let (one, two) = (load(1, &w, Mxfp4Act::None, false, t()), load(2, &w, Mxfp4Act::None, false, t()));
    assert_eq!(two.tp_world(), 2);
    let worst = compare(&run(&two), &run(&one), "W4A16 tp=2 vs tp=1");
    eprintln!("MXFP4 hybrid tp=2 vs tp=1, W4A16 (f32 activations): worst rel_l2 {worst:.2e}");
    assert!(worst < 1e-5, "W4A16 tp=2 vs tp=1: worst rel l2 {worst:.3e}");
}

/// W4A4 (the 27B's declared scheme): each rank QDQs its own K slice of whole 32-groups, which equals the slice of the full
/// QDQ, so the only difference left is again the f32 sum order (which can move an activation across an e2m1 rounding
/// boundary in a later layer: a looser bound than W4A16, still far below a real bug, see the next test).
#[test]
fn mxfp4_hybrid_tp2_matches_tp1_w4a4() {
    let _g = env();
    let w = weights();
    let t = || Arc::new(Telemetry::new());
    let (one, two) = (load(1, &w, Mxfp4Act::Dynamic, false, t()), load(2, &w, Mxfp4Act::Dynamic, false, t()));
    let rows = run(&two);
    let worst = compare(&rows, &run(&one), "W4A4 tp=2 vs tp=1");
    // the activation QDQ ran on the shards: W4A4 is not W4A16
    let w4a16 = run(&load(2, &w, Mxfp4Act::None, false, t()));
    let qdq = rows.iter().zip(&w4a16).map(|(x, y)| rel_l2(x, y)).fold(0f64, f64::max);
    eprintln!("MXFP4 hybrid tp=2 vs tp=1, W4A4 (f32 activations): worst rel_l2 {worst:.2e}; tp=2 W4A4 vs W4A16 {qdq:.2e}");
    assert!(worst < 1e-3, "W4A4 tp=2 vs tp=1: worst rel l2 {worst:.3e}");
    assert!(qdq > 1e-4, "tp=2 W4A4 equals W4A16: the activation QDQ did not run");
}

/// The comparison separates a 1 % partial-scale bug on rank 1 (every all-reduce: attention o, GDN out, FFN down) from
/// the sum order by orders of magnitude.
#[test]
fn mxfp4_hybrid_tp2_catches_a_one_percent_partial_scale_bug() {
    let _g = env();
    let w = weights();
    let t = || Arc::new(Telemetry::new());
    let one = load(1, &w, Mxfp4Act::None, false, t());
    std::env::set_var("INTARSIA_TP_TEST_BUG", "scale:1.01");
    let bad = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load(2, &w, Mxfp4Act::None, false, t())));
    std::env::remove_var("INTARSIA_TP_TEST_BUG");
    let bad = bad.unwrap();
    let worst = run(&bad).iter().zip(&run(&one)).map(|(x, y)| rel_l2(x, y)).fold(0f64, f64::max);
    eprintln!("MXFP4 hybrid tp=2 with scale:1.01 on rank 1 vs tp=1: worst rel_l2 {worst:.2e}");
    assert!(worst > 1e-4, "a 1% bug moved the logits by only {worst:.2e} (clean W4A16 bound 1e-5)");
}

/// Row-local fusion under TP: the fused MXFP4 pair (`MatmulMxfp4Pair`: GDN qkv|z and b|a, attention k|v, FFN gate|up)
/// runs on each rank's shard, and the result is still tp=1's up to the sum order.
#[test]
fn mxfp4_hybrid_tp2_row_local_fusion_matches_tp1() {
    let _g = env();
    let w = weights();
    for act in [Mxfp4Act::None, Mxfp4Act::Dynamic] {
        let one = load(1, &w, act, false, Arc::new(Telemetry::new()));
        let tel = Arc::new(Telemetry::new());
        let fused = load(2, &w, act, true, tel.clone());
        let worst = compare(&run(&fused), &run(&one), "fused tp=2 vs tp=1");
        let ops = tel.snapshot().ops;
        let pair = ops.get("matmul_mxfp4_pair").map_or(0, |s| s.calls);
        eprintln!("MXFP4 hybrid tp=2 fused vs tp=1, act {}: worst rel_l2 {worst:.2e}; matmul_mxfp4_pair calls on rank 0 {pair}", act.name());
        assert!(worst < if act == Mxfp4Act::None { 1e-5 } else { 1e-3 }, "act {}: {worst:.3e}", act.name());
        assert!(pair > 0, "the MXFP4 pair did not run under tp_fusion");
    }
}

/// The split per rank: half the attention / KV / GDN heads and half the FFN, only the slices on the device, rank 1 without
/// the head; and tp=1 is untouched by the shard path (the plain load and `load_shard` at world 1 give the same bits).
#[test]
fn mxfp4_hybrid_tp2_rank_shapes_bytes_and_tp1_unchanged() {
    let _g = env();
    let w = weights();
    let t = || Arc::new(Telemetry::new());
    let (one, two) = (load(1, &w, Mxfp4Act::Dynamic, false, t()), load(2, &w, Mxfp4Act::Dynamic, false, t()));
    let (r0, r1) = (two.rank_model(0).unwrap(), two.rank_model(1).unwrap());
    for r in [r0, r1] {
        let (l, g) = (r.ldims.linear.unwrap(), one.ldims.linear.unwrap());
        assert_eq!((l.k_heads, l.v_heads), (g.k_heads / 2, g.v_heads / 2));
        assert_eq!((r.ldims.n_heads, r.ldims.n_kv_heads), (one.dims.n_heads / 2, one.dims.n_kv_heads / 2));
        assert_eq!(r.ldims.intermediate, I / 2);
    }
    assert!(r1.weights_bytes < r0.weights_bytes, "{} vs {}", r1.weights_bytes, r0.weights_bytes);
    assert!(r0.weights_bytes < one.weights_bytes, "{} vs {}", r0.weights_bytes, one.weights_bytes);

    let (recipe, dims) = dims();
    let b = Model::load_shard(
        CpuBackend::arc(),
        &recipe,
        &dims,
        &w,
        &opts(Mxfp4Act::Dynamic, false),
        Arc::new(Telemetry::new()),
        intarsia_core::model::TpRank::single(),
    )
    .unwrap();
    for (i, (x, y)) in run(&one).iter().zip(&run(&b)).enumerate() {
        assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "tp=1 row {i}");
    }
}

/// Qwen3.8-27B's own dimensions split two ways with whole 32-element groups on every row-parallel input (o_proj 6144,
/// GDN out_proj 6144, down_proj 17408 columns), and the GDN q|k|v groups / heads divide.
#[test]
fn qwen38_27b_dims_split_two_ways_on_whole_mxfp4_groups() {
    let _g = env();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5_vl.toml")).unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_27b_quark_mxfp4");
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let s = d.sharded(2).unwrap();
    let (l, ls) = (d.linear.unwrap(), s.linear.unwrap());
    assert_eq!((s.n_heads, s.n_kv_heads, s.intermediate), (12, 2, 8704));
    assert_eq!((ls.k_heads, ls.v_heads), (8, 24));
    assert_eq!(l.v_heads / l.k_heads, ls.v_heads / ls.k_heads, "v heads per k head unchanged");
    let qdim = s.n_heads * s.head_dim;
    let vdim = ls.v_heads * ls.v_head_dim;
    for (what, k) in [("o_proj", qdim), ("out_proj", vdim), ("down_proj", s.intermediate)] {
        assert_eq!(k % 32, 0, "{what}: {k} columns per rank");
    }
    assert_eq!((qdim, vdim), (3072, 3072));
    // in_proj_qkv groups q | k | v per rank
    assert_eq!((ls.k_heads * ls.k_head_dim, ls.v_heads * ls.v_head_dim), (1024, 3072));
}
