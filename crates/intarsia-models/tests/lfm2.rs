//! LFM2 / LFM2.5 (lane nab2): the gated short-conv block (`Op::ShortConv`, `model/short_conv.rs`) and the `recipes/lfm2.toml` recipe
//! against Hugging Face.
//!
//! `tests/fixtures/tiny_lfm2/` is a random tiny `Lfm2ForCausalLM` (6 layers: conv, conv, full, conv, full, conv; fp32; tied embeddings;
//! the FFN width comes from `block_auto_adjust_ff_dim`: `int(2 * 96 / 3) = 64`) with the logits HF produced for a fixed 26-token
//! prompt (`tools/gen_lfm2_fixture.py`). The cpu backend runs it in f32 through the recipe-driven decoder: one-shot prefill, chunked
//! prefill (the conv state must carry across chunk boundaries), prefill followed by token-by-token decode, snapshot restore,
//! the reset after a failed pass, batched decode bitwise the solo step, and the load-time refusal of a backend without the kernel.

mod common;
#[path = "common/batch_identity.rs"]
mod batch_identity;

use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::{DType, Error, KernelImpl, Op, OpCall, Result, Telemetry};
use intarsia_models::SafeTensors;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_lfm2")
}

/// name -> (shape, values) of a safetensors file with f32 / i32 tensors.
fn read_safetensors(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
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

fn recipe() -> intarsia_core::recipe::Recipe {
    intarsia_models::parse_recipe(include_str!("../../../recipes/lfm2.toml")).unwrap()
}

fn dims() -> ModelDims {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap()).unwrap();
    ModelDims::from_hf_config(&cfg, &recipe().config).unwrap()
}

fn opts(chunk: usize) -> LoadOptions {
    LoadOptions { max_seq: 192, prefill_chunk: chunk, act_dtype: Some(DType::F32), kv_dtype: Some(DType::F32), ..Default::default() }
}

fn load_on(backend: Arc<dyn Backend>, chunk: usize) -> Result<Model> {
    let src = SafeTensors::open_dir(&fixture_dir()).unwrap();
    Model::load(backend, &recipe(), &dims(), &src, &opts(chunk), Arc::new(Telemetry::new()))
}

fn load(chunk: usize) -> Model {
    load_on(intarsia_backend_cpu::CpuBackend::arc(), chunk).unwrap()
}

fn golden() -> (Vec<u32>, Vec<Vec<f32>>) {
    let g = read_safetensors(&fixture_dir().join("golden.safetensors"));
    let ids: Vec<u32> = g["ids"].1.iter().map(|&x| x as u32).collect();
    let (shape, flat) = &g["logits"];
    (ids, flat.chunks(shape[1]).map(|r| r.to_vec()).collect())
}

/// max |got - want| over a row, and the relative L2 error.
fn compare(got: &[f32], want: &[f32]) -> (f32, f64) {
    let max = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    (max, (num / den).sqrt())
}

fn check_rows(label: &str, first_pos: usize, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len() - first_pos, "{label}: row count");
    let (mut worst_abs, mut worst_rel) = (0f32, 0f64);
    for (i, g) in got.iter().enumerate() {
        let (max, rel) = compare(g, &want[first_pos + i]);
        assert!(max < 1e-3 && rel < 1e-4, "{label}: position {}: max abs {max}, rel l2 {rel}", first_pos + i);
        worst_abs = worst_abs.max(max);
        worst_rel = worst_rel.max(rel);
    }
    println!("{label}: {} rows, worst max abs {worst_abs:.2e}, worst rel l2 {worst_rel:.2e}", got.len());
}

fn all_rows(m: &Model, st: &mut intarsia_core::state::SequenceState, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    let mut cb = |_p: usize, r: &[f32]| {
        rows.push(r.to_vec());
        Ok(())
    };
    m.forward(st, ids, HeadMode::AllLogits, Some(&mut cb)).unwrap();
    rows
}

#[test]
fn dims_follow_the_recipe_config_renames() {
    let d = dims();
    // intermediate_size 96 in the config, 64 in the weights (block_auto_adjust_ff_dim)
    assert_eq!((d.hidden, d.intermediate, d.n_heads, d.n_kv_heads, d.head_dim), (32, 64, 4, 2, 8));
    assert_eq!(d.short_conv.map(|c| c.kernel), Some(3));
    assert_eq!((d.rms_eps, d.rope_theta, d.tie_embeddings), (1e-5, 10000.0, true));
    assert!(d.linear.is_none());
}

#[test]
fn layer_kinds_follow_the_config() {
    let m = load(64);
    let kinds: Vec<_> = m.layer_kinds().into_iter().map(|l| l[0]).collect();
    assert_eq!(kinds, ["short_conv", "short_conv", "attention", "short_conv", "attention", "short_conv"]);
    // the 4 conv layers hold recurrent state (the conv's last K-1 columns), the 2 full-attention layers a KV cache
    let st = m.new_state().unwrap();
    assert_eq!(st.slots.len(), 6);
    assert!(st.has_recurrent());
    assert_eq!(m.recurrent_bytes_per_state(), 4 * 32 * 2 * 4);
}

#[test]
fn one_shot_prefill_matches_hf() {
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    let rows = all_rows(&m, &mut st, &ids);
    check_rows("prefill", 0, &rows, &want);
    assert_eq!(st.len, ids.len());
}

#[test]
fn chunked_prefill_matches_hf() {
    let (ids, want) = golden();
    // 26 tokens in chunks of 3, 5, 7: the conv state crosses every boundary
    for chunk in [3, 5, 7] {
        let m = load(chunk);
        let mut st = m.new_state().unwrap();
        let rows = all_rows(&m, &mut st, &ids);
        check_rows(&format!("chunk {chunk}"), 0, &rows, &want);
    }
}

#[test]
fn prefill_then_token_by_token_decode_matches_hf() {
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    let k = 17;
    let pre = all_rows(&m, &mut st, &ids[..k]);
    check_rows("prefill part", 0, &pre, &want[..k]);
    let mut dec = Vec::new();
    for i in k..ids.len() {
        match m.forward(&mut st, &ids[i..i + 1], HeadMode::LastLogits, None).unwrap() {
            StepOut::Logits(l) => dec.push(l),
            _ => panic!("no logits"),
        }
    }
    check_rows("decode", k, &dec, &want);
}

#[test]
fn recurrent_snapshots_restore_exactly() {
    // prefix reuse: snapshot at 12 tokens, run on to 26, roll back to 12, replay -> identical
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &ids[..12], HeadMode::None, None).unwrap();
    st.snapshot(&*m.backend, 2).unwrap();
    let rows = all_rows(&m, &mut st, &ids[12..]);
    check_rows("first pass", 12, &rows, &want);
    assert_eq!(st.restorable_len(20), 12);
    assert!(st.truncate(15, &*m.backend).is_err());
    st.truncate(12, &*m.backend).unwrap();
    let again = all_rows(&m, &mut st, &ids[12..]);
    assert_eq!(rows, again, "replaying from a restored snapshot must be bit-identical");
    st.truncate(0, &*m.backend).unwrap();
    let full = all_rows(&m, &mut st, &ids);
    check_rows("after reset", 0, &full, &want);
}

#[test]
fn failed_forward_resets_the_conv_state() {
    let (ids, want) = golden();
    let m = load(64);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &ids[..10], HeadMode::None, None).unwrap();
    assert!(m.forward(&mut st, &[9999], HeadMode::None, None).is_err());
    assert_eq!(st.len, 0);
    let rows = all_rows(&m, &mut st, &ids);
    check_rows("after failed pass", 0, &rows, &want);
}

/// Batched decode of N sequences is bitwise the solo decode step of each (the conv runs once per row on its own slot).
#[test]
fn batched_decode_is_bitwise_the_solo_step() {
    let mut m = load(64);
    m.allow_recurrent_batch().unwrap();
    m.enable_batch(4).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    let cfg = batch_identity::Cfg { lens: vec![4, 7, 10, 13], batches: vec![2, 3, 4, 2, 3, 4, 3, 2], steps: 12, vocab_range: (1, 95) };
    let n = batch_identity::run(&m, &cfg);
    assert_eq!(n, (2 + 3 + 4 + 2 + 3 + 4 + 3 + 2) * 12);
}

/// The cpu backend without its `ShortConv` kernels, as CUDA and HIP register them: the load fails with a list naming the op.
struct NoShortConv(Arc<dyn Backend>);

impl Backend for NoShortConv {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn info(&self) -> DeviceInfo {
        self.0.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> Result<Buffer> {
        self.0.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> Result<()> {
        self.0.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> Result<()> {
        self.0.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> Result<()> {
        self.0.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.0.kernels().into_iter().filter(|k| k.op != Op::ShortConv).collect()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Result<()> {
        self.0.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.0.synchronize(s)
    }
}

#[test]
fn a_backend_without_the_kernel_refuses_the_load_naming_the_op() {
    let e = load_on(Arc::new(NoShortConv(intarsia_backend_cpu::CpuBackend::arc())), 64).err().expect("load must fail");
    let m = e.to_string();
    assert!(matches!(e, Error::NoKernel(_)), "{m}");
    assert!(m.contains("short_conv") && m.contains("no kernel"), "{m}");
}

/// The CUDA and HIP kernel tables carry no `ShortConv` (and say so in their op lists), so an LFM2 checkpoint fails at load there.
#[test]
fn cuda_and_hip_register_no_short_conv_kernel() {
    assert!(!intarsia_backend_cuda::kernels::all().iter().any(|k| k.op == Op::ShortConv));
    assert!(!intarsia_backend_hip::kernels::all().iter().any(|k| k.op == Op::ShortConv));
    assert!(intarsia_backend_hip::kernels::UNIMPLEMENTED.contains(&Op::ShortConv));
}
