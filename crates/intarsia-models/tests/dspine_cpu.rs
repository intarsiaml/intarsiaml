//! `--spec dflash` with a DSpine draft (`crates/intarsia-core/src/model/dspine.rs`) on the cpu backend, bf16 activations.
//!
//! Target: the tiny hybrid `Qwen3_5ForCausalLM` of `tests/fixtures/tiny_qwen3_5` (6 layers), loaded at bf16 activations (the code
//! bridge rounds at bf16 points, and Tencent's reference refuses any other activation dtype). Draft: `tests/fixtures/tiny_dspine`,
//! a random 3-layer DSpine checkpoint (4 / 2 heads of 16, code 16, rank 8, K = 4, block 6, window 8, taps 1 / 3 / 5) written by
//! `tools/dspine/dspine_mlx_fixture.py` together with what Tencent's own MLX reference (`dspine_mlx`) computes for it.
//!
//! * `dspine_draft_matches_the_mlx_reference`: per-layer bridge rows, final hidden rows, the last write's stash, candidates and
//!   drafts against the reference for four cases (window a no-op / one row past it / every row past it, chunked ingest, a served
//!   shorter block), through the real loader.
//! * the tail on the reference's own stash and tables (bitwise drafts), the tables our matmul build produces, the walk against
//!   the full-table-plus-chain formulation, a shorter block as a prefix of the longer one.
//! * greedy ids with the draft equal `--spec off` on the hybrid target.

mod common;
use common::ByteCodec;
use intarsia_backend_cpu::CpuBackend;
use intarsia_core::model::{HeadMode, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{DType, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Telemetry, Engine};
use intarsia_models::SafeTensors;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const H: usize = 64;
const V: usize = 96;
const R: usize = 8;
const K: usize = 4;
const L: usize = 3;
const BLOCK: usize = 6;
const WINDOW: usize = 8;

fn target_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}
fn draft_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_dspine")
}

/// name -> (shape, values) of a safetensors file with f32 / i32 tensors.
fn read_f32(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
    let data = std::fs::read(path).unwrap();
    let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&data[8..8 + hlen]).unwrap();
    let mut out = BTreeMap::new();
    for (name, v) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let off = v["data_offsets"].as_array().unwrap();
        let (b, e) = (8 + hlen + off[0].as_u64().unwrap() as usize, 8 + hlen + off[1].as_u64().unwrap() as usize);
        let vals: Vec<f32> = match v["dtype"].as_str().unwrap() {
            "F32" => data[b..e].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "I32" => data[b..e].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            other => panic!("dtype {other}"),
        };
        out.insert(name.clone(), (shape, vals));
    }
    out
}

fn target_spec() -> (intarsia_core::recipe::Recipe, ModelDims) {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(target_dir().join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert_eq!((dims.hidden, dims.vocab, dims.n_layers), (H, V, 6));
    (recipe, dims)
}

fn opts(dir: Option<&Path>, chunk: usize) -> LoadOptions {
    LoadOptions {
        max_seq: 512,
        prefill_chunk: chunk,
        act_dtype: Some(DType::Bf16),
        kv_dtype: Some(DType::Bf16),
        profile_every: 0,
        dflash: dir.map(|d| intarsia_models::load::open_dflash(Some(d), false).unwrap()),
        ..Default::default()
    }
}

/// The hybrid target (bf16) with the DSpine draft (or without); verify passes for `--spec-k k`.
fn model(draft: bool, k: Option<usize>, chunk: usize) -> (Model, Arc<Telemetry>) {
    let (recipe, dims) = target_spec();
    let src = SafeTensors::open_dir(&target_dir()).unwrap();
    let tel = Arc::new(Telemetry::new());
    let dd = draft_dir();
    let mut m = Model::load(CpuBackend::arc(), &recipe, &dims, &src, &opts(draft.then_some(&*dd), chunk), tel.clone()).unwrap();
    if let Some(k) = k {
        m.enable_verify(k + 1).unwrap();
        m.configure_plan(&Default::default(), None).unwrap();
    }
    (m, tel)
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Worst per-row relative L2 of `[rows, w]` values.
fn worst_rows(got: &[f32], want: &[f32], w: usize) -> f64 {
    got.chunks(w).zip(want.chunks(w)).map(|(g, x)| rel_l2(g, x)).fold(0.0, f64::max)
}

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| (x.to_bits() >> 16) as u16).collect()
}

const CASES: [(&str, usize); 4] = [("a", 6), ("b", 6), ("c", 6), ("d", 3)];

/// Tolerances of the engine's bf16 pipeline against MLX's: both do f32 math with bf16 stores; the extra roundings are the
/// engine's `fc` partial sums (split in three) and the matmul accumulation order.
///
/// 3e-2 is measured noise, not slack for a bug: the same MLX reference run on its Metal GPU against its own CPU golden gives
/// 1.0e-2 to 1.3e-2 per row (cases a, b, c), and this engine against MLX 1.3e-2 to 2.0e-2. The design's 2e-3 is out of reach even
/// for the reference across devices. For the same reason the end-to-end candidates and drafts are gated on an agreement rate
/// ([`MIN_CAND_AGREE`], [`MIN_DRAFT_AGREE`]), not bitwise: the fixture dumper keeps every top-K and tail decision at least 1.0 %
/// (relative) from a tie, which is inside that noise, so a harmless change of accumulation order may flip one row. The bitwise gate
/// is the tail on the reference's own inputs (`dspine_tail_is_the_reference_on_its_own_inputs`).
const HIDDEN_TOL: f64 = 3e-2;
/// Least share of rows whose candidate set equals the reference's, and of drafted ids equal to its drafts (17/17 on the committed seed).
const MIN_CAND_AGREE: f64 = 0.85;
const MIN_DRAFT_AGREE: f64 = 0.75;

/// Run one case through the loader and the draft: returns the model with its last pass recorded, and the drafts.
fn run_case(gold: &BTreeMap<String, (Vec<usize>, Vec<f32>)>, name: &str, t: usize) -> (Model, Vec<u32>) {
    let (m, _) = model(true, Some(BLOCK - 1), 64);
    let mut st = m.new_state().unwrap();
    let feat = &gold[&format!("{name}.feat")].1;
    let chunks: Vec<usize> = gold[&format!("{name}.chunks")].1.iter().map(|&x| x as usize).collect();
    let mut done = 0;
    for n in chunks {
        m.dflash_ingest_host(&mut st, &feat[done * 3 * H..(done + n) * 3 * H], n).unwrap();
        done += n;
    }
    m.dspine_trace(true).unwrap();
    let ids: Vec<u32> = gold[&format!("{name}.ids")].1.iter().map(|&x| x as u32).collect();
    let drafts = m.dflash_draft(&mut st, ids[0], t - 1).unwrap();
    (m, drafts)
}

#[test]
fn dspine_draft_matches_the_mlx_reference() {
    let gold = read_f32(&draft_dir().join("golden.safetensors"));
    let (mut worst_layer, mut worst_hidden, mut worst_stash, mut agree, mut total) = (0f64, 0f64, 0f64, 0usize, 0usize);
    let (mut cand_agree, mut cand_total) = (0usize, 0usize);
    for (name, t) in CASES {
        let (m, drafts) = run_case(&gold, name, t);
        // per-layer rows: the post-write residual stream and the next layer's normed input
        let layers = m.dspine_layers().unwrap();
        assert_eq!(layers.len(), L);
        for (l, [out, normed, raw]) in layers.iter().enumerate() {
            let (eo, en) = (worst_rows(out, &gold[&format!("{name}.layer{l}.out")].1, H), worst_rows(normed, &gold[&format!("{name}.layer{l}.normed")].1, H));
            let er = worst_rows(raw, &gold[&format!("{name}.layer{l}.raw")].1, H);
            eprintln!("mlx parity: case {name} layer {l}: raw (before the write) {er:.2e} out {eo:.2e} normed {en:.2e}");
            worst_layer = worst_layer.max(eo).max(en).max(er);
        }
        let hid = m.dflash_last_hidden(t).unwrap();
        let eh = worst_rows(&hid, &gold[&format!("{name}.hidden")].1, H);
        let pass = m.dspine_last_pass().unwrap();
        let er = worst_rows(&pass.raw, &gold[&format!("{name}.raw")].1, H);
        // packed rows 1 .. (row 0's code is the anchor's: its sender half is replaced by the TSV row, and it is never read)
        let ep = worst_rows(&pass.packed[3 * R..], &gold[&format!("{name}.packed")].1[3 * R..], 3 * R);
        let es = pass.rms.iter().zip(&gold[&format!("{name}.rms")].1).map(|(a, b)| ((a - b) / b).abs() as f64).fold(0.0, f64::max);
        eprintln!("mlx parity: case {name}: hidden {eh:.2e} raw {er:.2e} packed {ep:.2e} rms rel {es:.2e}");
        worst_hidden = worst_hidden.max(eh);
        worst_stash = worst_stash.max(er).max(ep).max(es);
        let want_cand: Vec<Vec<usize>> = gold[&format!("{name}.cand")].1.chunks(K).map(|c| c.iter().map(|&x| x as usize).collect()).collect();
        let want_drafts: Vec<u32> = gold[&format!("{name}.drafts")].1.iter().map(|&x| x as u32).collect();
        let am: Vec<u32> = gold[&format!("{name}.argmax")].1.iter().map(|&x| x as u32).collect();
        eprintln!("mlx parity: case {name}: drafts {drafts:?} reference {want_drafts:?} (argmax {am:?}); candidates equal: {}", pass.cand == want_cand);
        total += want_drafts.len();
        agree += drafts.iter().zip(&want_drafts).filter(|(a, b)| a == b).count();
        assert_eq!(pass.cand.len(), want_cand.len(), "case {name}: candidate rows");
        assert_eq!(drafts.len(), want_drafts.len(), "case {name}: drafts");
        cand_total += want_cand.len();
        cand_agree += pass.cand.iter().zip(&want_cand).filter(|(a, b)| a == b).count();
    }
    eprintln!(
        "mlx parity: worst layer row {worst_layer:.2e}, hidden {worst_hidden:.2e}, stash {worst_stash:.2e}; candidate rows {cand_agree}/{cand_total}, drafts {agree}/{total}"
    );
    assert!(worst_layer < HIDDEN_TOL && worst_hidden < HIDDEN_TOL && worst_stash < HIDDEN_TOL);
    assert!(cand_agree as f64 >= MIN_CAND_AGREE * cand_total as f64, "candidate sets agree on {cand_agree}/{cand_total} rows");
    assert!(agree as f64 >= MIN_DRAFT_AGREE * total as f64, "drafts agree on {agree}/{total}");
}

/// The tail is the reference's on the reference's own inputs: its stash, candidates and tables give its drafts bitwise, and the
/// chain over the full `[n, K, K]` table equals the sequential walk. The tables our matmul build produces agree with its.
#[test]
fn dspine_tail_is_the_reference_on_its_own_inputs() {
    let gold = read_f32(&draft_dir().join("golden.safetensors"));
    let (m, _) = model(true, Some(BLOCK - 1), 64);
    m.dspine_prepare().unwrap();
    let (tg, tsv) = m.dspine_tables().unwrap();
    // our tables against the reference's: a matmul in a different accumulation order, so a few elements differ by an ulp
    for (got, want, what) in [(&tg, &gold["tg"].1, "TG"), (&tsv, &gold["tsv"].1, "TSV")] {
        let wb = bits(want);
        let same = got.iter().zip(&wb).filter(|(a, b)| a == b).count();
        let off1 = got.iter().zip(&wb).filter(|(a, b)| a != b && (**a as i32 - **b as i32).abs() <= 1).count();
        eprintln!("tables: {what} {}/{} bitwise, {} one ulp, {} other", same, got.len(), off1, got.len() - same - off1);
        assert!(same as f64 >= 0.99 * got.len() as f64 && same + off1 >= got.len() * 999 / 1000, "{what} agreement");
    }
    for (name, t) in CASES {
        let n = t - 1;
        let raw: Vec<u16> = bits(&gold[&format!("{name}.raw")].1[H..]);
        let rms: Vec<f32> = gold[&format!("{name}.rms")].1[1..].to_vec();
        let packed = &gold[&format!("{name}.packed")].1;
        let q: Vec<f32> = (1..t).flat_map(|r| packed[r * 3 * R..r * 3 * R + R].to_vec()).collect();
        let cand: Vec<Vec<usize>> = gold[&format!("{name}.cand")].1.chunks(K).map(|c| c.iter().map(|&x| x as usize).collect()).collect();
        let anchor = gold[&format!("{name}.ids")].1[0] as u32;
        let (walk, chain) = m.dspine_tail_host(&raw, &rms, &q, &cand, anchor).unwrap();
        let want: Vec<u32> = gold[&format!("{name}.drafts")].1.iter().map(|&x| x as u32).collect();
        assert_eq!(walk.len(), n);
        assert_eq!(walk, chain, "case {name}: the sequential walk equals the table + chain");
        assert_eq!(walk, want, "case {name}: the tail on the reference's own inputs");
    }
}

/// A shorter block is the prefix of a longer one (the block is causal): the engine runs `min(block, k + 1)` rows and relies on it.
#[test]
fn shorter_block_is_a_prefix() {
    let gold = read_f32(&draft_dir().join("golden.safetensors"));
    let (long, dl) = run_case(&gold, "c", 6);
    let (short, ds) = run_case(&gold, "c", 3);
    let (hl, hs) = (long.dflash_last_hidden(6).unwrap(), short.dflash_last_hidden(3).unwrap());
    assert_eq!(bits(&hl[..3 * H]), bits(&hs[..3 * H]), "the first 3 hidden rows");
    let (pl, ps) = (long.dspine_last_pass().unwrap(), short.dspine_last_pass().unwrap());
    assert_eq!(bits(&pl.raw[..3 * H]), bits(&ps.raw[..3 * H]));
    assert_eq!(pl.cand[..2], ps.cand[..]);
    assert_eq!(dl[..2], ds[..]);
}

// ------------------------------------------------------------------------------------------------------------------
// engine: token identity on the hybrid target
// ------------------------------------------------------------------------------------------------------------------

fn engine(k: Option<usize>) -> (LocalEngine, Arc<Telemetry>) {
    let (m, tel) = model(k.is_some(), Some(k.map_or(5, |k| k)), 8);
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Dflash, k, k_min: None },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny-dspine".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone()).unwrap();
    (e, tel)
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    reason: FinishReason,
    completion_tokens: usize,
}

fn run(e: &LocalEngine, req: GenRequest) -> Out {
    let (mut ids, mut done) = (vec![], None);
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Done { finish_reason, completion_tokens, .. } => done = Some((finish_reason, completion_tokens)),
        GenEvent::Error { message } => panic!("engine error: {message}"),
    })
    .unwrap();
    let (reason, completion_tokens) = done.unwrap();
    Out { ids, reason, completion_tokens }
}

fn req(prompt: &[u32], n: usize) -> GenRequest {
    GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: 0.0, ignore_eos: true, ..Default::default() }
}

const PROMPTS: [&[u32]; 3] = [&[3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12], &[90, 1, 2, 3, 4, 5], &[44; 21]];

#[test]
fn dspine_greedy_ids_equal_spec_off_on_the_hybrid() {
    let (off, _) = engine(None);
    for k in [1usize, 3, 5] {
        let (on, tel) = engine(Some(k));
        for p in PROMPTS {
            // 96 new tokens: every context ends up well past the window (8)
            assert_eq!(run(&off, req(p, 96)), run(&on, req(p, 96)), "k {k} prompt {p:?}");
        }
        let t = tel.snapshot();
        assert!(t.spec_steps > 0 && t.spec_proposed > 0, "k {k}: the draft drafted");
        eprintln!("k {k}: {} verify passes, {} drafted, {} accepted", t.spec_steps, t.spec_proposed, t.spec_accepted);
    }
    let _ = HeadMode::None;
}

// ------------------------------------------------------------------------------------------------------------------
// loader refusals: each names the key it refuses
// ------------------------------------------------------------------------------------------------------------------

/// The fixture draft in a scratch dir with its config edited by `edit`; `open_dflash`'s error text, or `None` when it loads.
fn refused(tag: &str, edit: impl Fn(&mut serde_json::Value)) -> Option<String> {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("dspine-refuse-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(draft_dir().join("model.safetensors"), dir.join("model.safetensors")).unwrap();
    let mut cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(draft_dir().join("config.json")).unwrap()).unwrap();
    edit(&mut cfg);
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&cfg).unwrap()).unwrap();
    let r = intarsia_models::load::open_dflash(Some(&dir), false);
    let _ = std::fs::remove_dir_all(&dir);
    r.err().map(|e| e.to_string())
}

/// The fixture draft with the tensor named `rename` renamed (so it is missing); `open_dflash`'s error text, or `None` when it loads.
fn refused_missing(tag: &str, rename: &str) -> Option<String> {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("dspine-refuse-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(draft_dir().join("config.json"), dir.join("config.json")).unwrap();
    let raw = std::fs::read(draft_dir().join("model.safetensors")).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let mut hdr: serde_json::Value = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let key = hdr.as_object().unwrap().keys().find(|k| k.ends_with(rename)).unwrap_or_else(|| panic!("no tensor ends with {rename}")).clone();
    let v = hdr.as_object_mut().unwrap().remove(&key).unwrap();
    hdr.as_object_mut().unwrap().insert(format!("{key}.renamed"), v);
    let hb = serde_json::to_vec(&hdr).unwrap();
    let mut out = (hb.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&hb);
    out.extend_from_slice(&raw[8 + n..]);
    std::fs::write(dir.join("model.safetensors"), out).unwrap();
    let r = intarsia_models::load::open_dflash(Some(&dir), false);
    let _ = std::fs::remove_dir_all(&dir);
    r.err().map(|e| e.to_string())
}

#[test]
fn the_fixture_loads_as_dspine() {
    let dl = intarsia_models::load::open_dflash(Some(&draft_dir()), false).unwrap();
    let s = dl.config.spine.as_ref().expect("a spine config");
    assert_eq!((s.code, s.rank, s.head_topk), (16, R, K));
    assert!(dl.config.v2 && dl.config.sliding_window == WINDOW && dl.config.block == BLOCK && dl.config.n_layers == L);
}

#[test]
fn loader_refusals_name_the_key() {
    let cases: Vec<(&str, Box<dyn Fn(&mut serde_json::Value)>, &str)> = vec![
        ("nokey", Box::new(|c| { c["dflash_config"].as_object_mut().unwrap().remove("selected_prefix_code_bridge"); }), "selected_prefix_code_bridge"),
        ("terminal", Box::new(|c| c["dflash_config"]["selected_prefix_terminal"] = true.into()), "selected_prefix_terminal"),
        ("k12", Box::new(|c| { c["code_bridge_second_pass"]["head_topk"] = 12.into(); c["dflash_config"]["selected_prefix_last_write_refine"]["head_topk"] = 12.into(); }), "head_topk"),
        ("kdisagree", Box::new(|c| c["code_bridge_second_pass"]["head_topk"] = 8.into()), "head_topk"),
        ("layers", Box::new(|c| c["dflash_config"]["selected_prefix_feature_layers"] = serde_json::json!([1, 2])), "selected_prefix_feature_layers"),
        ("noncausal", Box::new(|c| c["dflash_config"]["draft_block_causal"] = false.into()), "draft_block_causal"),
        ("mixed", Box::new(|c| c["layer_types"][0] = "full_attention".into()), "layer_types"),
        ("firstexact", Box::new(|c| c["code_bridge_second_pass"]["first_exact"] = 1.into()), "first_exact"),
        ("mode", Box::new(|c| c["dflash_config"]["selected_prefix_decode_mode"] = "parallel".into()), "selected_prefix_decode_mode"),
        ("conv", Box::new(|c| c["dflash_config"]["conv_kernel_size"] = 2.into()), "conv_kernel_size"),
        ("noeps", Box::new(|c| { c.as_object_mut().unwrap().remove("rms_norm_eps"); }), "rms_norm_eps"),
    ];
    for (tag, edit, key) in cases {
        let e = refused(tag, edit).unwrap_or_else(|| panic!("{tag}: loaded"));
        assert!(e.contains(key), "{tag}: error {e:?} does not name {key}");
    }
    assert!(refused("ok", |_| {}).is_none());
    // the two bridge format scalars are required; the top-level ones are optional
    for key in ["selected_prefix_bridge.independent_write_version", "selected_prefix_bridge.terminal_disabled_version"] {
        let e = refused_missing(key, key).unwrap_or_else(|| panic!("{key} missing: loaded"));
        assert!(e.contains(key), "missing {key}: error {e:?}");
    }
    for key in ["draft_block_causal_version", "last_write_refine_version"] {
        assert!(refused_missing(key, key).is_none(), "{key} is optional");
    }
}
