//! EXL3 (ExLlamaV3 trellis) cpu kernels, decoder and loader against two independent references:
//!
//! * exllamav3's own reconstruction and forward on real tensors (`tests/fixtures/exl3_cases.safetensors`, written on
//!   spark1 by `tools/gen_exl3_fixture.py` with the exllamav3 CUDA extension): slices of a published EXL3 checkpoint
//!   and tensors quantised by exllamav3's own quantiser at every width / codebook;
//! * TensorFold's MIT numpy decoder (`tools/exl3_tensorfold_oracle.py`; opt-in, `EXL3_TF_ORACLE=<file>` and
//!   `--ignored`, since it needs a TensorFold checkout to regenerate).

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::exl3::{self, Codebook};
use intarsia_core::quant::{self, HostQTensor, QuantLoadOpts, WeightMeta, EXL3};
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
use std::collections::BTreeMap;

fn load_path(path: &str) -> (BTreeMap<String, HostTensor>, serde_json::Value) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let mut out = BTreeMap::new();
    let mut meta = serde_json::Value::Null;
    for (k, v) in hdr.as_object().unwrap() {
        if k == "__metadata__" {
            meta = v.clone();
            continue;
        }
        let dt = DType::from_safetensors(v["dtype"].as_str().unwrap()).unwrap();
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let (s, e) = (v["data_offsets"][0].as_u64().unwrap() as usize, v["data_offsets"][1].as_u64().unwrap() as usize);
        out.insert(k.clone(), HostTensor::new(dt, shape, b[8 + n + s..8 + n + e].to_vec()).unwrap());
    }
    (out, meta)
}

fn fixture(name: &str) -> Option<(BTreeMap<String, HostTensor>, serde_json::Value)> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&path).exists().then(|| load_path(&path))
}

fn cases(meta: &serde_json::Value) -> Vec<serde_json::Value> {
    serde_json::from_str(meta["cases"].as_str().unwrap()).unwrap()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = v[r * cols + c];
        }
    }
    t
}

/// The fixture's tensors of case `p` in a weight source, under the checkpoint names (`p.trellis`, `p.suh`, ...).
fn source(s: &BTreeMap<String, HostTensor>, p: &str) -> MemWeights {
    let mut w = MemWeights::new();
    for part in ["trellis", "suh", "svh", "su", "sv", "mcg", "mul1"] {
        if let Some(t) = s.get(&format!("{p}.{part}")) {
            w.insert(&format!("{p}.{part}"), t.clone());
        }
    }
    w
}

/// Load case `p` through the format plugin (detect + load on the cpu backend) and download it.
fn load_case(b: &CpuBackend, s: &BTreeMap<String, HostTensor>, p: &str) -> HostQTensor {
    let w = source(s, p);
    let meta = WeightMeta::new(&w, p);
    let f = quant::detect(&meta).unwrap();
    assert_eq!(f.id(), EXL3, "{p}");
    let q = f.load(&meta, b, &QuantLoadOpts::default()).unwrap();
    assert_eq!(q.op(false), Op::MatmulExl3);
    q.download(b).unwrap()
}

fn run_op(b: &CpuBackend, kernel: &str, q: &HostQTensor, x: &HostTensor, odt: DType) -> Vec<f32> {
    let mut inputs = vec![x.clone()];
    inputs.extend(q.parts.iter().cloned());
    let hc = HostCall { op: Op::MatmulExl3, inputs, outputs: vec![OutSpec::new(odt, &[x.shape[0], q.n])], attrs: OpAttrs::default() };
    run_host(b, kernel, &hc).unwrap().remove(0).to_f32()
}

fn cb_of(q: &HostQTensor) -> Codebook {
    Codebook::from_i32(i32::from_le_bytes(q.parts[3].data[..4].try_into().unwrap())).unwrap()
}

/// exllamav3's reconstruction of real EXL3 tensors: the decoded W_q bit for bit, the full weight and the kernel's
/// forward within fp16 rounding. Prints the measured errors (the M2 results doc quotes them).
#[test]
fn matches_exllamav3_on_real_tensors() {
    let Some((s, meta)) = fixture("exl3_cases.safetensors") else {
        panic!("tests/fixtures/exl3_cases.safetensors missing (tools/gen_exl3_fixture.py)");
    };
    let b = CpuBackend::new();
    let cs = cases(&meta);
    assert!(!cs.is_empty());
    for c in &cs {
        let p = c["name"].as_str().unwrap();
        let q = load_case(&b, &s, p);
        let (k, n) = (q.k, q.n);
        let k2 = exl3::k2_of(&q.parts[0].shape).unwrap();
        // 1. the rotated-basis decode: bitwise
        let wq_want = &s[&format!("{p}.wq")];
        assert_eq!(wq_want.shape, vec![k, n], "{p}");
        let wq = exl3::unpack(&q.parts[0].data, k, n, k2, cb_of(&q));
        let want: Vec<u16> = wq_want.data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let bad = wq.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "{p} ({c}): {bad} of {} decoded W_q values differ from exllamav3's reconstruct", wq.len());
        // 2. the dequantised weight (ours f64 -> f32; exllamav3's get_weight_tensor rounds to fp16 after each step)
        let w_ref = transpose(&s[&format!("{p}.w")].to_f32(), k, n);
        let w = q.dequant();
        let (wr, wm) = (rel_l2(&w, &w_ref), max_abs(&w, &w_ref));
        let wmax = w_ref.iter().fold(0f64, |m, v| m.max(v.abs() as f64));
        // 3. the op against exllamav3's own forward (fp16 GEMV kernel), both kernel sets, f32 output
        let x = s[&format!("{p}.x")].clone();
        let y_ref = s[&format!("{p}.y")].to_f32();
        let mut yr = 0f64;
        for kernel in ["ref_matmul_exl3", "cpu_f32_matmul_exl3"] {
            let y = run_op(&b, kernel, &q, &x, DType::F32);
            yr = rel_l2(&y, &y_ref);
            let ymax = y_ref.iter().fold(0f64, |m, v| m.max(v.abs() as f64));
            eprintln!(
                "{p} {} bits={} cb={} K={k} N={n} {kernel}: W_q bitwise; W rel_l2 {wr:.2e} max_abs {wm:.2e} (max |W| {wmax:.2e}); y rel_l2 {yr:.2e} max_abs {:.2e} (max |y| {ymax:.2e})",
                c["source"].as_str().unwrap_or(""),
                k2 as f64 / 2.0,
                cb_of(&q).name(),
                max_abs(&y, &y_ref)
            );
            // exllamav3 rounds xh, W_q products and y to fp16: a few fp16 ulps relative
            assert!(yr < 3e-3, "{p} {kernel}: y rel_l2 {yr} vs exllamav3 forward");
        }
        assert!(wr < 2e-3, "{p}: W rel_l2 {wr} vs exllamav3 get_weight_tensor");
        let _ = yr;
    }
}

/// TensorFold's numpy decoder: W_q bitwise, W and y within f64 -> f32 rounding (opt-in, see the module doc).
#[test]
#[ignore]
fn matches_tensorfold_numpy_oracle() {
    let path = std::env::var("EXL3_TF_ORACLE").expect("set EXL3_TF_ORACLE to the output of tools/exl3_tensorfold_oracle.py");
    let (s, meta) = load_path(&path);
    let b = CpuBackend::new();
    let cs = cases(&meta);
    assert_eq!(cs.len(), 27, "every width x codebook");
    for c in &cs {
        let p = c["name"].as_str().unwrap();
        let (k, n) = (c["K"].as_u64().unwrap() as usize, c["N"].as_u64().unwrap() as usize);
        let bits = c["bits"].as_f64().unwrap();
        let cb = Codebook::parse(c["codebook"].as_str().unwrap()).unwrap();
        let k2 = (bits * 2.0) as usize;
        let tr = &s[&format!("{p}.trellis")];
        let wq = exl3::unpack(&tr.data, k, n, k2, cb);
        let want: Vec<u16> = s[&format!("{p}.wq")].data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(wq, want, "{p} bits={bits} {cb:?}: W_q");
        let mut parts = vec![tr.clone(), s[&format!("{p}.suh")].clone(), s[&format!("{p}.svh")].clone()];
        parts.push(HostTensor { dtype: DType::I32, shape: vec![1], data: (cb as i32).to_le_bytes().to_vec() });
        let q = HostQTensor { format: EXL3, n, k, parts, act_quant: false };
        let w_ref = transpose(&s[&format!("{p}.w")].to_f32(), k, n);
        let r = rel_l2(&q.dequant(), &w_ref);
        assert!(r < 1e-7, "{p} bits={bits} {cb:?}: W rel_l2 {r}");
        let y = run_op(&b, "ref_matmul_exl3", &q, &s[&format!("{p}.x")], DType::F32);
        let r = rel_l2(&y, &s[&format!("{p}.y")].to_f32());
        assert!(r < 1e-7, "{p} bits={bits} {cb:?}: y rel_l2 {r}");
    }
}

/// The op's output dtypes and the two kernel sets agree with the f64 forward on random tensors of every width.
#[test]
fn op_matches_f64_forward_every_width() {
    let b = CpuBackend::new();
    let mut rng = intarsia_core::rng::SplitMix64::new(11);
    for (k2, cb) in exl3::widths() {
        let (m, n, k) = (3, 256, 384);
        let q = exl3::random(&mut rng, n, k, k2, cb);
        let x = HostTensor::from_f32_as(DType::Bf16, &[m, k], &rng.normal_vec(m * k, 1.0));
        let (suh, svh) = (exl3::f16_bytes(&q.parts[1].data), exl3::f16_bytes(&q.parts[2].data));
        let want: Vec<f32> = exl3::forward(&x.to_f32(), m, k, n, &q.parts[0].data, &suh, &svh, k2, cb).iter().map(|v| *v as f32).collect();
        for kernel in ["ref_matmul_exl3", "cpu_f32_matmul_exl3"] {
            assert_eq!(run_op(&b, kernel, &q, &x, DType::F32), want, "{k2} {cb:?} {kernel}");
            let yb = run_op(&b, kernel, &q, &x, DType::Bf16);
            assert!(rel_l2(&yb, &want) < 4e-3, "{k2} {cb:?} {kernel} bf16");
        }
    }
}

/// The loader: f16 `suh`/`svh` or packed `su`/`sv`, the codebook from the marker tensors, refusals for what
/// exllamav3's LinearEXL3 would not load either.
#[test]
fn loader_reads_markers_and_packed_signs() {
    let b = CpuBackend::new();
    let mut rng = intarsia_core::rng::SplitMix64::new(5);
    let q = exl3::random(&mut rng, 128, 256, 8, Codebook::Mcg);
    let marker = |v: u32| HostTensor { dtype: DType::I32, shape: vec![], data: v.to_le_bytes().to_vec() };
    // packed signs: +-1 scales
    let pack = |n: usize, rng: &mut intarsia_core::rng::SplitMix64| -> HostTensor {
        let d: Vec<u8> = (0..n / 16).flat_map(|_| (rng.next_u64() as u16).to_le_bytes()).collect();
        HostTensor { dtype: DType::I16, shape: vec![n / 16], data: d }
    };
    let mut w = MemWeights::new();
    w.insert("a.trellis", q.parts[0].clone());
    w.insert("a.suh", q.parts[1].clone());
    w.insert("a.svh", q.parts[2].clone());
    w.insert("a.mcg", marker(exl3::MCG_MUL));
    w.insert("b.trellis", q.parts[0].clone());
    w.insert("b.su", pack(256, &mut rng));
    w.insert("b.sv", pack(128, &mut rng));
    w.insert("b.mul1", marker(exl3::MUL1_MUL));
    w.insert("c.trellis", q.parts[0].clone());
    w.insert("c.suh", q.parts[1].clone());
    w.insert("c.svh", q.parts[2].clone());
    w.insert("c.mcg", marker(0x1234));
    w.insert("d.trellis", q.parts[0].clone());
    w.insert("d.suh", q.parts[1].clone());
    w.insert("d.svh", q.parts[2].clone());
    w.insert("d.bias", HostTensor::from_f32_as(DType::F16, &[128], &[0.0; 128]));
    w.insert("e.trellis", q.parts[0].clone());
    w.insert("e.suh", q.parts[1].clone());
    w.insert("e.svh", q.parts[2].clone());
    let ld = |p: &str| quant::load_linear(&w, p, &b, &QuantLoadOpts::default());
    let a = ld("a").unwrap().download(&b).unwrap();
    assert_eq!((a.n, a.k, cb_of(&a)), (128, 256, Codebook::Mcg));
    assert_eq!(a.dequant(), q.dequant());
    let bq = ld("b").unwrap().download(&b).unwrap();
    assert_eq!(cb_of(&bq), Codebook::Mul1);
    let su = exl3::f16_bytes(&bq.parts[1].data);
    assert!(su.iter().all(|v| v.abs() == 1.0) && su.iter().any(|v| *v < 0.0));
    assert!(ld("c").unwrap_err().to_string().contains("marker"));
    assert!(ld("d").unwrap_err().to_string().contains("bias"));
    assert_eq!(cb_of(&ld("e").unwrap().download(&b).unwrap()), Codebook::Inst3);
    // tensor parallelism is refused, not silently loaded whole
    let sh = intarsia_core::weights::Shard { axis: intarsia_core::weights::ShardAxis::Rows, rank: 0, world: 2 };
    assert!(quant::load_linear(&w, "a", &b, &QuantLoadOpts { shard: Some(sh), ..Default::default() }).is_err());
}
