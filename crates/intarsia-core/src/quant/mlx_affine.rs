//! MLX affine-quantised checkpoints (`mlx_lm.convert -q`, `mlx-community/*`), read as bf16.
//!
//! An MLX linear (or embedding) layer stores three tensors instead of `weight`:
//!
//! ```text
//! <m>.weight   u32 [.., cols * bits / 32]   little-endian bit stream per row, element i is bits [i*bits, (i+1)*bits)
//! <m>.scales   bf16|f16|f32 [.., cols / group]
//! <m>.biases   same shape and dtype as scales
//! w[r, c] = scales[r, c / group] * q[r, c] + biases[r, c / group]          (MLX `mx.dequantize`, mode "affine")
//! ```
//!
//! `config.json` carries `"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}` plus per-module overrides
//! (`"model.embed_tokens": {"group_size": 64, "bits": 6, "mode": "affine"}`). 2, 3, 4, 5, 6 and 8 bits all use the same
//! bit stream (checked against `mx.quantize` in `tools/gen_mlx_affine_fixture.py`).
//!
//! [`MlxAffine`] wraps a [`WeightSource`] and presents every such module as one plain bf16 `<m>.weight`: the engine's
//! dequantise-at-load path. It costs memory (a bf16 copy is 16 / bits times the stored size: 4x for 4-bit) and gives up
//! the bandwidth win of reading 4-bit weights; a native MLX-quant GEMV is a later change. The arithmetic is f32
//! (`scale * q + bias`, the product is exact for 8-bit `q` and 8-bit-mantissa scales) rounded once to bf16, which is
//! bitwise what `mx.dequantize` returns for bf16 scales.
//!
//! Two conversion artefacts of `mlx_lm` are undone as well. The `mlx-vlm` tensor names (`language_model.model.*` to `model.language_model.*`, the HF
//! names the recipes use) and the MLX depthwise-conv layout `[C, K, 1]` to HF's `[C, 1, K]` (same bytes).
//! And `mlx_lm`'s `sanitize` bakes the `1 +` of zero-centred RMSNorms (Qwen3.5 / Qwen3-Next: `x * (1 + w)`) into the stored
//! weight, so the layer norms read mean ~1 instead of ~0; the engine's recipes apply the `1 +` themselves
//! (`model.norm_offset`), so [`MlxAffine`] gives those norms back as `w - 1` (exact in bf16: see `unshift`) when the config
//! is such a family and layer 0's `input_layernorm` has mean above 0.5.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::weights::{WeightInfo, WeightSource};
use half::{bf16, f16};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

/// `config.json` `quantization` of an MLX checkpoint: the default and per-module `(group_size, bits)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MlxAffineConfig {
    pub group_size: usize,
    pub bits: usize,
    /// Per-module overrides, keyed by the checkpoint's module name (the tensor name without `.weight`).
    pub overrides: BTreeMap<String, (usize, usize)>,
    /// Modules the config lists as `false` (stored as plain float).
    pub plain: BTreeSet<String>,
    /// `model_type` is a zero-centred-RMSNorm family (Qwen3.5, Qwen3-Next): the converter may have baked `1 +` into the norms.
    pub zero_centred_norms: bool,
}

fn mode_ok(v: &Value, what: &str) -> Result<()> {
    match v.get("mode").and_then(Value::as_str) {
        None | Some("affine") => Ok(()),
        Some(m) => Err(Error::Unsupported(format!(
            "MLX quantization mode \"{m}\" ({what}); only \"affine\" is read (no mxfp4 / nvfp4 / mxfp8 from MLX)"
        ))),
    }
}

fn gb(v: &Value, what: &str) -> Result<(usize, usize)> {
    let g = v.get("group_size").and_then(Value::as_u64);
    let b = v.get("bits").and_then(Value::as_u64);
    match (g, b) {
        (Some(g), Some(b)) if g > 0 && matches!(b, 2 | 3 | 4 | 5 | 6 | 8) => Ok((g as usize, b as usize)),
        _ => Err(Error::model(format!(
            "MLX quantization ({what}): need group_size > 0 and bits in 2, 3, 4, 5, 6, 8, got {v}"
        ))),
    }
}

impl MlxAffineConfig {
    /// `Ok(None)` when `config.json` (or its `text_config`) declares no MLX `quantization`. A `quantization_config` that
    /// has a `quant_method` (GPTQ, AWQ, compressed-tensors, ...) is not MLX: `None`.
    pub fn from_config(cfg: &Value) -> Result<Option<MlxAffineConfig>> {
        let q = cfg.get("quantization").or_else(|| cfg.get("text_config").and_then(|t| t.get("quantization")));
        let Some(q) = q.filter(|q| q.is_object()) else { return Ok(None) };
        if q.get("quant_method").is_some() {
            return Ok(None);
        }
        mode_ok(q, "default")?;
        let (group_size, bits) = gb(q, "default")?;
        let mut overrides = BTreeMap::new();
        let mut plain = BTreeSet::new();
        for (k, v) in q.as_object().into_iter().flatten() {
            match v {
                Value::Object(_) => {
                    mode_ok(v, k)?;
                    overrides.insert(k.clone(), gb(v, k)?);
                }
                Value::Bool(false) => {
                    plain.insert(k.clone());
                }
                _ => {}
            }
        }
        let mt = cfg.get("text_config").and_then(|t| t.get("model_type")).or_else(|| cfg.get("model_type")).and_then(Value::as_str);
        let zero_centred_norms = mt.is_some_and(|m| m.starts_with("qwen3_5") || m.starts_with("qwen3_next"));
        Ok(Some(MlxAffineConfig { group_size, bits, overrides, plain, zero_centred_norms }))
    }

    /// `(group_size, bits)` the config declares for a module (checkpoint name, with or without `.weight`).
    pub fn params_for(&self, module: &str) -> (usize, usize) {
        let m = module.strip_suffix(".weight").unwrap_or(module);
        self.overrides.get(m).copied().unwrap_or((self.group_size, self.bits))
    }
}

/// The HF / recipe name of an `mlx-vlm` tensor (identity for `mlx-lm` names, which are already HF's).
pub fn map_name(n: &str) -> String {
    if let Some(r) = n.strip_prefix("language_model.model.") {
        format!("model.language_model.{r}")
    } else if let Some(r) = n.strip_prefix("language_model.lm_head.") {
        format!("lm_head.{r}")
    } else {
        n.to_string()
    }
}

/// MLX stores a depthwise `Conv1d` weight as `[C, K, 1]`; HF as `[C, 1, K]`. The bytes are identical.
pub fn fix_conv_shape(name: &str, shape: &[usize]) -> Vec<usize> {
    if name.ends_with("conv1d.weight") && shape.len() == 3 && shape[2] == 1 && shape[1] > 1 {
        vec![shape[0], 1, shape[1]]
    } else {
        shape.to_vec()
    }
}

/// `(cols, bits)` of a packed module from its packed last dimension and its scales' last dimension.
pub fn logical_cols(name: &str, packed_cols: usize, scales_cols: usize, group: usize) -> Result<(usize, usize)> {
    let cols = scales_cols * group;
    if cols == 0 || (packed_cols * 32) % cols != 0 {
        return Err(Error::model(format!(
            "MLX affine {name}: packed width {packed_cols} words does not fit {scales_cols} scale groups of {group}"
        )));
    }
    let bits = packed_cols * 32 / cols;
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) || cols * bits != packed_cols * 32 {
        return Err(Error::model(format!(
            "MLX affine {name}: {packed_cols} packed words for {cols} columns is {bits} bits per weight"
        )));
    }
    Ok((cols, bits))
}

/// The norms `mlx_lm.sanitize` shifts by `+1` (names after `map_name`); the gated norm inside GatedDeltaNet is not one.
fn is_shifted_norm(name: &str) -> bool {
    name == "model.norm.weight"
        || name == "model.language_model.norm.weight"
        || [".input_layernorm.weight", ".post_attention_layernorm.weight", ".self_attn.q_norm.weight", ".self_attn.k_norm.weight"]
            .iter()
            .any(|s| name.ends_with(s))
}

/// `w - 1` in the tensor's own dtype. Exact for bf16 values in [0.5, 8) that came from `bf16(w0 + 1)`: they are multiples of
/// 2^-8 (2^-7 above 1), and so is the difference, which needs no more mantissa bits than the sum had.
fn unshift(dtype: DType, b: &[u8]) -> Result<Vec<u8>> {
    let mut out = b.to_vec();
    match dtype {
        DType::Bf16 => {
            for c in out.chunks_exact_mut(2) {
                let v = bf16::from_le_bytes([c[0], c[1]]).to_f32() - 1.0;
                c.copy_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F32 => {
            for c in out.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) - 1.0;
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        d => return Err(Error::model(format!("MLX norm weight dtype {d}: cannot undo the +1 shift"))),
    }
    Ok(out)
}

fn float_at(dtype: DType, b: &[u8], i: usize) -> f32 {
    match dtype {
        DType::Bf16 => bf16::from_le_bytes([b[2 * i], b[2 * i + 1]]).to_f32(),
        DType::F16 => f16::from_le_bytes([b[2 * i], b[2 * i + 1]]).to_f32(),
        _ => f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]),
    }
}

/// Dequantise `rows x cols` MLX affine weights to bf16 bytes (little-endian).
///
/// `words`: the packed `u32` bytes, `rows * cols * bits / 8` of them; `scales` / `biases`: `rows * cols / group` values of
/// `sb_dtype` (bf16, f16 or f32).
pub fn dequantize_bf16(
    words: &[u8],
    scales: &[u8],
    biases: &[u8],
    sb_dtype: DType,
    rows: usize,
    cols: usize,
    group: usize,
    bits: usize,
) -> Result<Vec<u8>> {
    if !matches!(sb_dtype, DType::Bf16 | DType::F16 | DType::F32) {
        return Err(Error::model(format!("MLX affine scales / biases dtype {sb_dtype} (want bf16, f16 or f32)")));
    }
    if group == 0 || cols % group != 0 || !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) || (cols * bits) % 32 != 0 {
        return Err(Error::model(format!("MLX affine: {cols} columns, group {group}, {bits} bits is not a valid layout")));
    }
    let (ng, row_bytes, esz) = (cols / group, cols * bits / 8, sb_dtype.size_bytes());
    if words.len() != rows * row_bytes || scales.len() != rows * ng * esz || biases.len() != scales.len() {
        return Err(Error::model(format!(
            "MLX affine {rows}x{cols} (group {group}, {bits} bits): got {} / {} / {} bytes of words / scales / biases, want {} / {} / {}",
            words.len(),
            scales.len(),
            biases.len(),
            rows * row_bytes,
            rows * ng * esz,
            rows * ng * esz
        )));
    }
    let mut out = vec![0u8; rows * cols * 2];
    let mask = (1u32 << bits) - 1;
    let work = |r0: usize, dst: &mut [u8]| {
        for (ri, drow) in dst.chunks_exact_mut(cols * 2).enumerate() {
            let r = r0 + ri;
            let wrow = &words[r * row_bytes..(r + 1) * row_bytes];
            for g in 0..ng {
                let s = float_at(sb_dtype, scales, r * ng + g);
                let b = float_at(sb_dtype, biases, r * ng + g);
                for c in g * group..(g + 1) * group {
                    let bit = c * bits;
                    let (byte, sh) = (bit / 8, bit % 8);
                    let lo = wrow[byte] as u32;
                    let hi = wrow.get(byte + 1).copied().unwrap_or(0) as u32;
                    let q = (((hi << 8) | lo) >> sh) & mask;
                    let v = bf16::from_f32(s * q as f32 + b);
                    drow[2 * c..2 * c + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    };
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(rows.max(1));
    if threads <= 1 || rows * cols < (1 << 16) {
        work(0, &mut out);
    } else {
        let per = rows.div_ceil(threads);
        std::thread::scope(|sc| {
            for (i, chunk) in out.chunks_mut(per * cols * 2).enumerate() {
                let work = &work;
                sc.spawn(move || work(i * per, chunk));
            }
        });
    }
    Ok(out)
}

enum Entry {
    /// A tensor passed through under (possibly) another name and shape.
    Alias { inner: String, info: WeightInfo },
    /// A norm weight the converter shifted by `+1`: given back as `w - 1` on first use.
    Unshift { inner: String, info: WeightInfo, cell: OnceLock<std::result::Result<Box<[u8]>, String>> },
    /// A packed module: dequantised on first use.
    Quant { module: String, rows: usize, cols: usize, group: usize, bits: usize, sb: DType, info: WeightInfo, cell: OnceLock<std::result::Result<Box<[u8]>, String>> },
}

/// A [`WeightSource`] whose MLX affine modules read as bf16 `.weight` tensors (module docs).
pub struct MlxAffine<S> {
    inner: S,
    map: BTreeMap<String, Entry>,
    norms_unshifted: bool,
}

impl<S: WeightSource> MlxAffine<S> {
    /// `inner` must report the packed `u32` tensors as `DType::I32` (same 4 bytes per element; the loader of
    /// `intarsia-models` maps safetensors `U32` that way).
    pub fn new(inner: S, cfg: &MlxAffineConfig) -> Result<MlxAffine<S>> {
        let names: BTreeSet<String> = inner.names().into_iter().collect();
        let mut map: BTreeMap<String, Entry> = BTreeMap::new();
        let put = |map: &mut BTreeMap<String, Entry>, outer: String, e: Entry| -> Result<()> {
            if map.insert(outer.clone(), e).is_some() {
                return Err(Error::model(format!("MLX checkpoint: two tensors map to the name '{outer}'")));
            }
            Ok(())
        };
        for n in &names {
            let module = n.strip_suffix(".weight");
            let packed = module.filter(|m| {
                names.contains(&format!("{m}.scales"))
                    && names.contains(&format!("{m}.biases"))
                    && inner.info(n).is_some_and(|i| i.dtype == DType::I32)
            });
            if let Some(m) = packed {
                let (wi, si) = (inner.info(n).unwrap(), inner.info(&format!("{m}.scales")).unwrap());
                let bi = inner.info(&format!("{m}.biases")).unwrap();
                if si.shape != bi.shape || si.dtype != bi.dtype || wi.shape.len() < 2 || si.shape.len() != wi.shape.len() || wi.shape[..wi.shape.len() - 1] != si.shape[..si.shape.len() - 1] {
                    return Err(Error::model(format!(
                        "MLX affine {m}: weight {:?} {} against scales {:?} {} and biases {:?} {}",
                        wi.shape, wi.dtype, si.shape, si.dtype, bi.shape, bi.dtype
                    )));
                }
                let (group, want_bits) = cfg.params_for(m);
                let nd = wi.shape.len();
                let (cols, bits) = logical_cols(m, wi.shape[nd - 1], si.shape[nd - 1], group)?;
                if bits != want_bits {
                    return Err(Error::model(format!(
                        "MLX affine {m}: stored as {bits} bits (from the tensor shapes, group {group}) but config.json \"quantization\" says {want_bits} bits"
                    )));
                }
                let rows: usize = wi.shape[..nd - 1].iter().product();
                let mut shape = wi.shape[..nd - 1].to_vec();
                shape.push(cols);
                let info = WeightInfo { dtype: DType::Bf16, shape };
                put(&mut map, map_name(n), Entry::Quant { module: m.to_string(), rows, cols, group, bits, sb: si.dtype, info, cell: OnceLock::new() })?;
            } else if n.strip_suffix(".scales").or_else(|| n.strip_suffix(".biases")).is_some_and(|m| {
                names.contains(&format!("{m}.scales")) && names.contains(&format!("{m}.biases")) && inner.info(&format!("{m}.weight")).is_some_and(|i| i.dtype == DType::I32)
            }) {
                // consumed by its module
            } else if let Some(i) = inner.info(n) {
                let shape = fix_conv_shape(n, &i.shape);
                put(&mut map, map_name(n), Entry::Alias { inner: n.clone(), info: WeightInfo { dtype: i.dtype, shape } })?;
            }
        }
        let mut me = MlxAffine { inner, map, norms_unshifted: false };
        if cfg.zero_centred_norms {
            let probe = me.map.keys().find(|k| k.ends_with("layers.0.input_layernorm.weight")).cloned();
            if let Some(k) = probe {
                let i = me.info(&k).unwrap();
                let mean = match i.dtype {
                    DType::Bf16 | DType::F16 | DType::F32 => {
                        let v = crate::weights::to_f32_vec(i.dtype, me.bytes(&k)?)?;
                        v.iter().sum::<f32>() / v.len().max(1) as f32
                    }
                    _ => 0.0,
                };
                if mean > 0.5 {
                    let shifted: Vec<String> = me.map.keys().filter(|k| is_shifted_norm(k)).cloned().collect();
                    for k in shifted {
                        if let Some(Entry::Alias { inner, info }) = me.map.remove(&k) {
                            me.map.insert(k, Entry::Unshift { inner, info, cell: OnceLock::new() });
                        }
                    }
                    me.norms_unshifted = true;
                }
            }
        }
        Ok(me)
    }

    /// The converter's `+1` on zero-centred norms was found and is undone on read.
    pub fn norms_unshifted(&self) -> bool {
        self.norms_unshifted
    }

    /// Number of modules dequantised on read.
    pub fn quantized_modules(&self) -> usize {
        self.map.values().filter(|e| matches!(e, Entry::Quant { .. })).count()
    }
}

impl<S: WeightSource> WeightSource for MlxAffine<S> {
    fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
    fn info(&self, name: &str) -> Option<WeightInfo> {
        match self.map.get(name)? {
            Entry::Alias { info, .. } | Entry::Unshift { info, .. } | Entry::Quant { info, .. } => Some(info.clone()),
        }
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        match self.map.get(name).ok_or_else(|| Error::model(format!("no tensor '{name}'")))? {
            Entry::Alias { inner, .. } => self.inner.bytes(inner),
            Entry::Unshift { inner, info, cell } => {
                match cell.get_or_init(|| {
                    self.inner.bytes(inner).and_then(|b| unshift(info.dtype, b)).map(Vec::into_boxed_slice).map_err(|e| e.to_string())
                }) {
                    Ok(b) => Ok(b),
                    Err(e) => Err(Error::model(e.clone())),
                }
            }
            Entry::Quant { module, rows, cols, group, bits, sb, cell, .. } => {
                let r = cell.get_or_init(|| {
                    let get = |suffix: &str| self.inner.bytes(&format!("{module}.{suffix}")).map_err(|e| e.to_string());
                    dequantize_bf16(get("weight")?, get("scales")?, get("biases")?, *sb, *rows, *cols, *group, *bits)
                        .map(Vec::into_boxed_slice)
                        .map_err(|e| e.to_string())
                });
                match r {
                    Ok(b) => Ok(b),
                    Err(e) => Err(Error::model(e.clone())),
                }
            }
        }
    }
    fn host_memory(&self, name: &str) -> Option<(std::sync::Arc<dyn crate::backend::HostMemory>, usize, usize)> {
        match self.map.get(name)? {
            Entry::Alias { inner, .. } => self.inner.host_memory(inner),
            Entry::Unshift { .. } | Entry::Quant { .. } => None,
        }
    }
    fn quant_scheme(&self) -> Option<&crate::quant::scheme::QuantScheme> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::HostTensor;
    use crate::weights::MemWeights;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// 18 cases (2, 3, 4, 5, 6, 8 bits x group 32, 64, 128) whose expected bf16 comes from the numpy reference of the MLX
    /// formula in `tools/gen_mlx_affine_fixture.py` (which also checks it against `mx.dequantize`): the reader must match
    /// it bit for bit.
    #[test]
    fn dequant_matches_the_numpy_reference_bitwise() {
        let fx: Value = serde_json::from_str(include_str!("../../tests/fixtures/mlx_affine.json")).unwrap();
        let cases = fx["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 18);
        for c in cases {
            let u = |k: &str| c[k].as_u64().unwrap() as usize;
            let (bits, group, rows, cols) = (u("bits"), u("group"), u("rows"), u("cols"));
            let h = |k: &str| unhex(c[k].as_str().unwrap());
            let got = dequantize_bf16(&h("words"), &h("scales"), &h("biases"), DType::Bf16, rows, cols, group, bits).unwrap();
            assert_eq!(got, h("expected"), "bits {bits} group {group}");
        }
    }

    #[test]
    fn rejects_inconsistent_layouts() {
        let w = vec![0u8; 4 * 32]; // 4 rows x 64 cols x 4 bits
        let s = vec![0u8; 4 * 2 * 2];
        assert!(dequantize_bf16(&w, &s, &s, DType::Bf16, 4, 64, 32, 4).is_ok());
        assert!(dequantize_bf16(&w[..127], &s, &s, DType::Bf16, 4, 64, 32, 4).is_err());
        assert!(dequantize_bf16(&w, &s, &s, DType::Bf16, 4, 64, 48, 4).is_err());
        assert!(dequantize_bf16(&w, &s, &s, DType::Bf16, 4, 64, 32, 7).is_err());
        assert!(logical_cols("m", 8, 2, 32).is_ok_and(|x| x == (64, 4)));
        assert!(logical_cols("m", 7, 2, 32).is_err());
    }

    #[test]
    fn config_parses_defaults_overrides_and_refuses_other_modes() {
        let c = serde_json::json!({"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
            "model.embed_tokens": {"group_size": 64, "bits": 6, "mode": "affine"}, "model.layers.0.gate": false}});
        let q = MlxAffineConfig::from_config(&c).unwrap().unwrap();
        assert_eq!(q.params_for("model.embed_tokens.weight"), (64, 6));
        assert_eq!(q.params_for("model.layers.1.mlp.up_proj"), (64, 4));
        assert!(q.plain.contains("model.layers.0.gate"));
        assert!(MlxAffineConfig::from_config(&serde_json::json!({"hidden_size": 8})).unwrap().is_none());
        assert!(MlxAffineConfig::from_config(&serde_json::json!({"quantization": {"quant_method": "gptq", "bits": 4}})).unwrap().is_none());
        assert!(MlxAffineConfig::from_config(&serde_json::json!({"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}})).is_err());
        assert!(MlxAffineConfig::from_config(&serde_json::json!({"quantization": {"group_size": 64, "bits": 7}})).is_err());
    }

    #[test]
    fn source_exposes_bf16_weights_and_hf_names() {
        // one 4-bit module (2 rows x 64 cols) under the mlx-vlm names, a norm, and a [C, K, 1] conv
        let fx: Value = serde_json::from_str(include_str!("../../tests/fixtures/mlx_affine.json")).unwrap();
        let c = fx["cases"].as_array().unwrap().iter().find(|c| c["bits"] == 4 && c["group"] == 64).unwrap();
        let (rows, cols) = (c["rows"].as_u64().unwrap() as usize, c["cols"].as_u64().unwrap() as usize);
        let h = |k: &str| unhex(c[k].as_str().unwrap());
        let mut mem = MemWeights::new();
        let t = |dt: DType, shape: &[usize], data: Vec<u8>| HostTensor { dtype: dt, shape: shape.to_vec(), data };
        let m = "language_model.model.layers.0.mlp.up_proj";
        mem.insert(&format!("{m}.weight"), t(DType::I32, &[rows, cols / 8], h("words")));
        mem.insert(&format!("{m}.scales"), t(DType::Bf16, &[rows, cols / 64], h("scales")));
        mem.insert(&format!("{m}.biases"), t(DType::Bf16, &[rows, cols / 64], h("biases")));
        mem.insert("language_model.model.norm.weight", HostTensor::from_f32_as(DType::Bf16, &[2], &[1.0, 2.0]));
        mem.insert("language_model.model.layers.0.linear_attn.conv1d.weight", HostTensor::from_f32_as(DType::Bf16, &[2, 4, 1], &[0.0; 8]));
        let cfg = MlxAffineConfig::from_config(&serde_json::json!({"quantization": {"group_size": 64, "bits": 4}})).unwrap().unwrap();
        let src = MlxAffine::new(mem, &cfg).unwrap();
        assert_eq!(src.quantized_modules(), 1);
        let mut names = src.names();
        names.sort();
        assert_eq!(
            names,
            vec![
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                "model.language_model.layers.0.mlp.up_proj.weight",
                "model.language_model.norm.weight"
            ]
        );
        let i = src.info("model.language_model.layers.0.mlp.up_proj.weight").unwrap();
        assert_eq!((i.dtype, i.shape), (DType::Bf16, vec![rows, cols]));
        assert_eq!(src.bytes("model.language_model.layers.0.mlp.up_proj.weight").unwrap(), &h("expected")[..]);
        assert_eq!(src.info("model.language_model.layers.0.linear_attn.conv1d.weight").unwrap().shape, vec![2, 1, 4]);
        // a config that disagrees with the tensors names the module
        let wrong = MlxAffineConfig::from_config(&serde_json::json!({"quantization": {"group_size": 64, "bits": 8}})).unwrap().unwrap();
        let mut mem2 = MemWeights::new();
        mem2.insert(&format!("{m}.weight"), t(DType::I32, &[rows, cols / 8], h("words")));
        mem2.insert(&format!("{m}.scales"), t(DType::Bf16, &[rows, cols / 64], h("scales")));
        mem2.insert(&format!("{m}.biases"), t(DType::Bf16, &[rows, cols / 64], h("biases")));
        let e = MlxAffine::new(mem2, &wrong).err().unwrap().to_string();
        assert!(e.contains("up_proj") && e.contains("4 bits") && e.contains("8 bits"), "{e}");
    }

    #[test]
    fn baked_in_norm_shift_is_undone_only_where_the_converter_applied_it() {
        let mk = |ln: &[f32]| {
            let mut mem = MemWeights::new();
            mem.insert("language_model.model.layers.0.input_layernorm.weight", HostTensor::from_f32_as(DType::Bf16, &[2], ln));
            mem.insert("language_model.model.layers.0.linear_attn.norm.weight", HostTensor::from_f32_as(DType::Bf16, &[2], &[0.75, 0.5]));
            mem.insert("language_model.model.norm.weight", HostTensor::from_f32_as(DType::Bf16, &[2], &[3.5, 6.25]));
            mem
        };
        let cfg = |mt: &str| {
            MlxAffineConfig::from_config(&serde_json::json!({"model_type": mt, "quantization": {"group_size": 64, "bits": 4}})).unwrap().unwrap()
        };
        let f = |s: &MlxAffine<MemWeights>, n: &str| crate::weights::to_f32_vec(DType::Bf16, s.bytes(n).unwrap()).unwrap();
        // shifted checkpoint (mean 1.4) of a zero-centred family: input / final norms give back w - 1, the gated norm stays
        let s = MlxAffine::new(mk(&[1.5, 1.25]), &cfg("qwen3_5")).unwrap();
        assert!(s.norms_unshifted());
        assert_eq!(f(&s, "model.language_model.layers.0.input_layernorm.weight"), vec![0.5, 0.25]);
        assert_eq!(f(&s, "model.language_model.norm.weight"), vec![2.5, 5.25]);
        assert_eq!(f(&s, "model.language_model.layers.0.linear_attn.norm.weight"), vec![0.75, 0.5]);
        // already zero-centred (mean 0.1): untouched; and another family is never touched
        let s = MlxAffine::new(mk(&[0.125, -0.0625]), &cfg("qwen3_5")).unwrap();
        assert!(!s.norms_unshifted());
        assert_eq!(f(&s, "model.language_model.norm.weight"), vec![3.5, 6.25]);
        let s = MlxAffine::new(mk(&[1.5, 1.25]), &cfg("qwen3")).unwrap();
        assert!(!s.norms_unshifted());
    }
}
