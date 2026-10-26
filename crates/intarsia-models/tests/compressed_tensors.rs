//! compressed-tensors NVFP4A16 (`nvfp4-pack-quantized`) checkpoints: the loader reads them as ModelOpt NVFP4 and
//! dequantises like the compressed-tensors reference.
//!
//! Fixtures (`tests/fixtures/ct/*.json`, made by `scripts/ct/make_reference.py`): packed weights, e4m3 scales and the global
//! scale of a small matrix, with `expected_f32_hex` = the float32 dequantisation by compressed-tensors' own
//! `unpack_fp4_from_uint8` + `dequantize` (`w = e2m1 * (weight_scale / weight_global_scale)`). The committed fixtures are
//! synthetic (every packed byte value; a compress / decompress round trip). Slices of real checkpoint layers are not committed
//! (licences): `scripts/ct/extract_slice.py` cuts them and the opt-in test `real_slices_match_the_reference` reads them from
//! the directory named by `CT_REAL_SLICES_DIR` (skipped when it is unset).
//!
//! The reference divides by the global scale and the engine multiplies by its reciprocal (computed in f32 at open), so the
//! two differ by rounding only: the test requires every element within 2 f32 ulp, and reports how many are bit-equal.

use intarsia_core::quant::compressed_tensors::CtNvfp4;
use intarsia_core::quant::nvfp4::Nvfp4;
use intarsia_core::quant::scheme::CT_NVFP4A16;
use intarsia_core::quant::{load_linear, QuantFormat, QuantLoadOpts, NVFP4};
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{DType, HostTensor};
use intarsia_models::load::{read_quant_scheme, SchemedSource};
use serde_json::Value;
use std::path::PathBuf;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

struct Fixture {
    name: String,
    n: usize,
    k: usize,
    packed: Vec<u8>,
    scale: Vec<u8>,
    gs: f32,
    expected: Vec<f32>,
}

fn fixtures() -> Vec<Fixture> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ct");
    let v = fixtures_in(&dir);
    assert!(v.len() >= 2, "fixtures missing in {}", dir.display());
    v
}

fn fixtures_in(dir: &std::path::Path) -> Vec<Fixture> {
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
    files.sort();
    files
        .iter()
        .map(|p| {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            let h = |k: &str| hex(v[k].as_str().unwrap());
            Fixture {
                name: p.file_name().unwrap().to_string_lossy().into_owned(),
                n: v["n"].as_u64().unwrap() as usize,
                k: v["k"].as_u64().unwrap() as usize,
                packed: h("weight_packed_hex"),
                scale: h("weight_scale_hex"),
                gs: v["weight_global_scale"].as_f64().unwrap() as f32,
                expected: h("expected_f32_hex").chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            }
        })
        .collect()
}

/// The module `m` as compressed-tensors stores it.
fn ct_weights(f: &Fixture, m: &str) -> MemWeights {
    let mut w = MemWeights::new();
    w.insert(&format!("{m}.weight_packed"), HostTensor { dtype: DType::U8, shape: vec![f.n, f.k / 2], data: f.packed.clone() });
    w.insert(&format!("{m}.weight_scale"), HostTensor { dtype: DType::Fp8E4m3, shape: vec![f.n, f.k / 16], data: f.scale.clone() });
    w.insert(&format!("{m}.weight_global_scale"), HostTensor::from_f32(&[1], &[f.gs]));
    w
}

fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        (if i < 0 { i32::MIN.wrapping_sub(i) } else { i }) as i64
    };
    key(a).abs_diff(key(b))
}

fn check_against_reference(fixtures: Vec<Fixture>) {
    let backend = intarsia_backend_cpu::CpuBackend::arc();
    for f in fixtures {
        let src = CtNvfp4::new(ct_weights(&f, "m")).unwrap();
        let q = load_linear(&src, "m", backend.as_ref(), &QuantLoadOpts::default()).unwrap();
        assert_eq!((q.format, q.n, q.k), (NVFP4, f.n, f.k), "{}", f.name);
        let got = Nvfp4.dequant_ref(&q, backend.as_ref()).unwrap().to_f32();
        assert_eq!(got.len(), f.expected.len());
        let (mut worst, mut equal) = (0, 0);
        for (g, e) in got.iter().zip(&f.expected) {
            let u = ulps(*g, *e);
            worst = worst.max(u);
            equal += (u == 0) as usize;
        }
        println!("{}: {} elements, {} bit-equal, worst {} ulp", f.name, got.len(), equal, worst);
        assert!(worst <= 2, "{}: worst difference {worst} f32 ulp", f.name);
    }
}

#[test]
fn dequantises_like_the_compressed_tensors_reference() {
    check_against_reference(fixtures());
}

/// Opt-in: slices of real layers (`scripts/ct/extract_slice.py` + `make_reference.py`) in the directory `CT_REAL_SLICES_DIR`.
#[test]
fn real_slices_match_the_reference() {
    let Some(dir) = std::env::var_os("CT_REAL_SLICES_DIR") else {
        eprintln!("CT_REAL_SLICES_DIR is not set: real-slice check skipped");
        return;
    };
    let v = fixtures_in(std::path::Path::new(&dir));
    assert!(!v.is_empty(), "no *.json slices in {:?}", dir);
    check_against_reference(v);
}

#[test]
fn the_view_shows_modelopt_names_and_hides_the_compressed_tensors_ones() {
    let f = &fixtures()[0];
    let mut w = ct_weights(f, "model.layers.0.mlp.up_proj");
    w.insert("lm_head.weight", HostTensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
    let src = CtNvfp4::new(w).unwrap();
    assert_eq!(src.modules(), 1);
    let mut names = src.names();
    names.sort();
    assert_eq!(
        names,
        ["lm_head.weight", "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.up_proj.weight_scale", "model.layers.0.mlp.up_proj.weight_scale_2"]
    );
    let m = "model.layers.0.mlp.up_proj";
    assert!(src.info(&format!("{m}.weight_packed")).is_none() && src.info(&format!("{m}.weight_global_scale")).is_none());
    assert!(src.bytes(&format!("{m}.weight_packed")).is_err());
    let wi = src.info(&format!("{m}.weight")).unwrap();
    assert_eq!((wi.dtype, wi.shape), (DType::U8, vec![f.n, f.k / 2]));
    let s2 = src.bytes(&format!("{m}.weight_scale_2")).unwrap();
    assert_eq!(f32::from_le_bytes(s2.try_into().unwrap()), 1.0 / f.gs);
    assert_eq!(src.info(&format!("{m}.weight_scale_2")).unwrap().shape, vec![1]);
    assert_eq!(src.bytes("lm_head.weight").unwrap().len(), 16, "other tensors pass through");
}

#[test]
fn modelopt_and_int4_tensors_are_left_alone() {
    // a ModelOpt NVFP4 module: nothing to adapt, names and bytes unchanged
    let f = &fixtures()[0];
    let mut w = MemWeights::new();
    w.insert("a.weight", HostTensor { dtype: DType::U8, shape: vec![f.n, f.k / 2], data: f.packed.clone() });
    w.insert("a.weight_scale", HostTensor { dtype: DType::Fp8E4m3, shape: vec![f.n, f.k / 16], data: f.scale.clone() });
    w.insert("a.weight_scale_2", HostTensor::from_f32(&[], &[0.25]));
    let before = w.names();
    let src = CtNvfp4::new(w).unwrap();
    assert_eq!(src.modules(), 0);
    assert_eq!(src.names(), before);
    assert_eq!(src.info("a.weight_scale_2").unwrap().shape, Vec::<usize>::new());
    // INT4 pack-quantized: weight_packed + weight_shape, no global scale: not an NVFP4 module, not renamed
    let mut w = MemWeights::new();
    w.insert("b.weight_packed", HostTensor { dtype: DType::I32, shape: vec![4, 4], data: vec![0; 64] });
    w.insert("b.weight_shape", HostTensor { dtype: DType::I32, shape: vec![2], data: vec![0; 8] });
    let src = CtNvfp4::new(w).unwrap();
    assert!(src.info("b.weight").is_none() && src.info("b.weight_packed").is_some());
    assert!(load_linear(&src, "b", intarsia_backend_cpu::CpuBackend::arc().as_ref(), &QuantLoadOpts::default()).is_err());
}

#[test]
fn malformed_modules_are_refused_with_their_name() {
    let f = &fixtures()[0];
    let with_gs = |gs: &[f32], shape: &[usize]| {
        let mut w = ct_weights(f, "m");
        w.insert("m.weight_global_scale", HostTensor::from_f32(shape, gs));
        CtNvfp4::new(w).err().map(|e| e.to_string())
    };
    assert!(with_gs(&[8384.0], &[1]).is_none());
    for bad in [0.0, -3.0, f32::NAN, f32::INFINITY] {
        let e = with_gs(&[bad], &[1]).unwrap();
        assert!(e.contains("'m'") && e.contains("finite positive"), "{e}");
    }
    assert!(with_gs(&[1.0, 2.0], &[2]).unwrap().contains("single f32"));
    // bf16 scales
    let mut w = ct_weights(f, "m");
    w.insert("m.weight_scale", HostTensor::from_f32(&[f.n, f.k / 16], &vec![1.0; f.n * f.k / 16]));
    assert!(CtNvfp4::new(w).err().unwrap().to_string().contains("fp8_e4m3"));
    // a module that has both
    let mut w = ct_weights(f, "m");
    w.insert("m.weight", HostTensor::from_f32(&[1, 1], &[0.0]));
    assert!(CtNvfp4::new(w).err().unwrap().to_string().contains("both"));
}

/// Minimal safetensors writer: `(name, dtype, shape, bytes)`.
fn write_safetensors(path: &std::path::Path, ts: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut hdr = serde_json::Map::new();
    let mut off = 0;
    for (n, d, s, b) in ts {
        hdr.insert((*n).into(), serde_json::json!({"dtype": d, "shape": s, "data_offsets": [off, off + b.len()]}));
        off += b.len();
    }
    let h = serde_json::to_vec(&Value::Object(hdr)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (_, _, _, b) in ts {
        out.extend(b);
    }
    std::fs::write(path, out).unwrap();
}

#[test]
fn a_checkpoint_directory_loads_through_the_schemed_source() {
    let f = &fixtures()[1];
    let dir = std::env::temp_dir().join(format!("intarsia-ct-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = serde_json::json!({
        "architectures": ["X"],
        "quantization_config": {
            "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized", "quantization_status": "compressed",
            "config_groups": {"group_0": {"format": "nvfp4-pack-quantized", "input_activations": null, "output_activations": null,
                "targets": ["Linear"], "weights": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group",
                "symmetric": true, "scale_dtype": "torch.float8_e4m3fn"}}},
            "ignore": ["lm_head"], "sparsity_config": {}, "transform_config": {}
        }
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            ("x.weight_packed", "U8", vec![f.n, f.k / 2], f.packed.clone()),
            ("x.weight_scale", "F8_E4M3", vec![f.n, f.k / 16], f.scale.clone()),
            ("x.weight_global_scale", "F32", vec![1], f.gs.to_le_bytes().to_vec()),
            ("lm_head.weight", "F32", vec![1, 2], [1f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect()),
        ],
    );
    let scheme = read_quant_scheme(&dir).unwrap().unwrap();
    assert_eq!(scheme.quant_algo.as_deref(), Some(CT_NVFP4A16));
    let src = SchemedSource::open(&dir).unwrap();
    assert!(src.scheme().is_some());
    let backend = intarsia_backend_cpu::CpuBackend::arc();
    let q = load_linear(&src, "x", backend.as_ref(), &QuantLoadOpts::default()).unwrap();
    src.quant_scheme().unwrap().check_linear("x", q.format).unwrap();
    src.quant_scheme().unwrap().check_linear("lm_head", intarsia_core::quant::BF16).unwrap();
    let got = Nvfp4.dequant_ref(&q, backend.as_ref()).unwrap().to_f32();
    assert!(got.iter().zip(&f.expected).all(|(g, e)| ulps(*g, *e) <= 2));

    // the same directory declared W4A4 is refused at open, not run as weight-only
    let mut bad = cfg.clone();
    bad["quantization_config"]["config_groups"]["group_0"]["input_activations"] = serde_json::json!({"num_bits": 4, "type": "float"});
    std::fs::write(dir.join("config.json"), bad.to_string()).unwrap();
    assert!(SchemedSource::open(&dir).err().unwrap().to_string().contains("W4A4"));
    // another compressed-tensors format (per-tensor FP8) keeps its old behaviour: no declared scheme, found by tensor names
    let mut fp8 = cfg.clone();
    fp8["quantization_config"] = serde_json::json!({"quant_method": "compressed-tensors", "format": "float-quantized",
        "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float", "strategy": "tensor"}}}});
    std::fs::write(dir.join("config.json"), fp8.to_string()).unwrap();
    assert!(read_quant_scheme(&dir).unwrap().is_none());
    let src = SchemedSource::open(&dir).unwrap();
    assert!(src.scheme().is_none() && src.info("x.weight_packed").is_some() && src.info("x.weight").is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}
