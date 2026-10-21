//! The quantisation scheme a checkpoint declares (`hf_quant_config.json`, ModelOpt) and the check of what we load
//! against it.
//!
//! Format detection ([`super::detect`]) looks at tensor names alone. That is enough to load a layer but not to know
//! whether we run it the way the checkpoint says: a ModelOpt export quantised as `NVFP4` (W4A4: activations are
//! quantised to FP4 at run time with `input_scale`) has the same tensors as `W4A16_NVFP4` and would silently run as
//! weight-only. This module reads the declared scheme and
//!
//! * rejects layers that declare an algorithm we do not implement, or one that disagrees with the tensors found
//!   (`FP8` declared, bf16 tensors stored, ...);
//! * answers [`QuantScheme::kv_note`] for `kv_cache_quant_algo` (the FP8 KV cache with the checkpoint's k/v scales is
//!   not implemented; running a bf16 cache is a deviation worth logging, not an error).
//!
//! Layer keys of `quantized_layers` are checkpoint module names (`model.language_model.layers.0.linear_attn.out_proj`);
//! the fused `mlp.experts` entries have no `.weight` tensor and are not checked here. `exclude_modules` entries are
//! globs with `*`.
//!
//! ExLlamaV3 checkpoints declare `quant_method = "exl3"` in `config.json` (or a sidecar `quantization_config.json`):
//! [`QuantScheme::parse_exl3`] records the global algorithm `EXL3`, under which every linear layer must be stored as
//! EXL3 or plain float (a `head_bits = 16` head, unquantised modules), never FP8/NVFP4.

use super::{QuantId, BF16, EXL3, FP8_E4M3, NVFP4};
use crate::dtype::DType;
use crate::error::{Error, Result};
use serde_json::Value;
use std::collections::BTreeMap;

/// What a declared algorithm means for us.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algo {
    /// `FP8`: per-tensor e4m3 weights with a static activation scale (W8A8).
    Fp8W8A8,
    /// `W4A16_NVFP4`: 4-bit weights, 16-bit activations.
    Nvfp4W4A16,
    /// `NVFP4` (and AWQ variants): 4-bit weights and 4-bit activations. Not implemented.
    Nvfp4W4A4,
    /// `EXL3` (ExLlamaV3 trellis weights, 16-bit activations): layers are EXL3 or plain float.
    Exl3,
    /// Anything else (INT4_AWQ, MXFP8, per-channel FP8, ...). Not implemented.
    Other,
}

impl Algo {
    pub fn parse(s: &str) -> Algo {
        match s.trim().to_ascii_uppercase().as_str() {
            "FP8" => Algo::Fp8W8A8,
            "W4A16_NVFP4" => Algo::Nvfp4W4A16,
            "EXL3" => Algo::Exl3,
            a if a == "NVFP4" || a.starts_with("NVFP4_") || a == "W4A4_NVFP4" || a == "W4A4_NVFP4_FP8" => Algo::Nvfp4W4A4,
            _ => Algo::Other,
        }
    }
    /// The tensor format a layer of this algorithm must be stored in.
    fn format(self) -> Option<QuantId> {
        match self {
            Algo::Fp8W8A8 => Some(FP8_E4M3),
            Algo::Nvfp4W4A16 => Some(NVFP4),
            Algo::Exl3 => Some(EXL3),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct QuantScheme {
    /// `quantization.quant_algo` (`MIXED_PRECISION` when every layer is listed in `quantized_layers`).
    pub quant_algo: Option<String>,
    /// `quantization.kv_cache_quant_algo`.
    pub kv_cache_quant_algo: Option<String>,
    /// `quantized_layers`: module name -> declared algorithm.
    layers: BTreeMap<String, String>,
    /// `exclude_modules` globs.
    exclude: Vec<String>,
}

fn glob(pat: &str, s: &str) -> bool {
    match pat.split_once('*') {
        None => pat == s,
        Some((head, rest)) => {
            let Some(mut t) = s.strip_prefix(head) else { return false };
            let parts: Vec<&str> = rest.split('*').collect();
            for (i, p) in parts.iter().enumerate() {
                if i + 1 == parts.len() {
                    return t.ends_with(p);
                }
                match t.find(p) {
                    Some(k) => t = &t[k + p.len()..],
                    None => return false,
                }
            }
            true
        }
    }
}

impl QuantScheme {
    /// Parse the contents of `hf_quant_config.json` (`{"producer": .., "quantization": {..}}`).
    pub fn parse(doc: &Value) -> Result<QuantScheme> {
        let q = doc.get("quantization").unwrap_or(doc);
        let as_str = |k: &str| q.get(k).and_then(Value::as_str).map(String::from);
        let mut layers = BTreeMap::new();
        if let Some(l) = q.get("quantized_layers") {
            let o = l.as_object().ok_or_else(|| Error::model("hf_quant_config.json: quantized_layers is not an object"))?;
            for (name, v) in o {
                let algo = v
                    .get("quant_algo")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::model(format!("hf_quant_config.json: layer '{name}' has no quant_algo")))?;
                layers.insert(name.clone(), algo.to_string());
            }
        }
        let exclude = q
            .get("exclude_modules")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        Ok(QuantScheme { quant_algo: as_str("quant_algo"), kv_cache_quant_algo: as_str("kv_cache_quant_algo"), layers, exclude })
    }

    /// The scheme of an ExLlamaV3 checkpoint from its EXL3 block (`quant::exl3::config_fields`), after
    /// `quant::exl3::check_config` refused the codebooks and widths we do not read.
    pub fn parse_exl3(fields: &Value) -> Result<QuantScheme> {
        super::exl3::check_config(fields)?;
        Ok(QuantScheme { quant_algo: Some("EXL3".into()), ..QuantScheme::default() })
    }

    pub fn is_excluded(&self, module: &str) -> bool {
        self.exclude.iter().any(|p| glob(p, module))
    }

    /// Declared algorithm of a module: its `quantized_layers` entry, else the global algorithm (unless the
    /// checkpoint is `MIXED_PRECISION`, where unlisted layers are not quantised), unless excluded.
    pub fn algo_of(&self, module: &str) -> Option<&str> {
        if self.is_excluded(module) {
            return None;
        }
        if let Some(a) = self.layers.get(module) {
            return Some(a);
        }
        match self.quant_algo.as_deref() {
            Some(a) if !a.eq_ignore_ascii_case("MIXED_PRECISION") => Some(a),
            _ => None,
        }
    }

    /// Number of modules with a declared per-layer algorithm.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Check one linear layer that was loaded in `format` against what the checkpoint declares for it.
    ///
    /// Errors: the declared algorithm is not implemented (W4A4 NVFP4, INT4_AWQ, ...), or it disagrees with the
    /// stored tensors. A layer the scheme does not mention is accepted as stored.
    pub fn check_linear(&self, module: &str, format: QuantId) -> Result<()> {
        let Some(algo) = self.algo_of(module) else {
            return Ok(());
        };
        let a = Algo::parse(algo);
        match a {
            Algo::Nvfp4W4A4 => Err(Error::Unsupported(format!(
                "'{module}' is declared quant_algo '{algo}' (NVFP4 weights AND 4-bit activations) in hf_quant_config.json; this engine \
                 implements the weight-only W4A16_NVFP4 scheme and would run it with 16-bit activations, different from the \
                 checkpoint's declared scheme and from vLLM"
            ))),
            Algo::Other => Err(Error::Unsupported(format!(
                "'{module}' is declared quant_algo '{algo}' in hf_quant_config.json, which this engine does not implement \
                 (supported: FP8, W4A16_NVFP4)"
            ))),
            // EXL3 checkpoints keep some layers plain (an fp16 head with head_bits 16, unquantised modules)
            Algo::Exl3 if format == EXL3 || format == BF16 => Ok(()),
            _ => {
                let want = a.format().unwrap();
                if format != want {
                    return Err(Error::model(format!(
                        "'{module}' is declared quant_algo '{algo}' in hf_quant_config.json (stored as {want}) but its tensors are {}",
                        if format == BF16 { "plain float weights".to_string() } else { format.to_string() }
                    )));
                }
                Ok(())
            }
        }
    }

    /// A deviation note when the checkpoint declares a quantised KV cache and we run another KV dtype.
    pub fn kv_note(&self, kv: DType) -> Option<String> {
        let a = self.kv_cache_quant_algo.as_deref().filter(|a| !a.is_empty() && !a.eq_ignore_ascii_case("none"))?;
        let declared_fp8 = a.to_ascii_uppercase().starts_with("FP8");
        let ours_fp8 = matches!(kv, DType::Fp8E4m3 | DType::Fp8E5m2);
        if declared_fp8 == ours_fp8 {
            return None;
        }
        Some(format!(
            "checkpoint declares kv_cache_quant_algo {a} (its k/v scales are not used); the KV cache runs as {} here: a deviation from the \
             declared scheme (and from a reference that runs the quantised cache)",
            kv.name()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> Value {
        serde_json::from_str(
            r#"{"producer":{"name":"modelopt"},"quantization":{"quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":"FP8",
            "quantized_layers":{"model.layers.0.linear_attn.out_proj":{"quant_algo":"FP8"},
            "model.layers.0.mlp.shared_expert.up_proj":{"quant_algo":"W4A16_NVFP4","group_size":16},
            "model.layers.1.self_attn.q_proj":{"quant_algo":"NVFP4","group_size":16},
            "model.layers.1.self_attn.k_proj":{"quant_algo":"INT4_AWQ"},
            "lm_head":{"quant_algo":"W4A16_NVFP4","group_size":16}},
            "exclude_modules":["mtp*","model.layers.5.*"]}}"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_the_modelopt_layout() {
        let s = QuantScheme::parse(&doc()).unwrap();
        assert_eq!(s.n_layers(), 5);
        assert_eq!(s.kv_cache_quant_algo.as_deref(), Some("FP8"));
        assert_eq!(s.algo_of("lm_head"), Some("W4A16_NVFP4"));
        assert_eq!(s.algo_of("model.layers.9.self_attn.q_proj"), None, "MIXED_PRECISION: unlisted layers are not quantised");
        assert!(s.is_excluded("mtp.fc") && s.is_excluded("model.layers.5.mlp.gate") && !s.is_excluded("model.layers.6.mlp.gate"));
    }

    #[test]
    fn globs() {
        assert!(glob("mtp*", "mtp.layers.0.x") && glob("*.gate", "a.b.gate") && glob("a*c*e", "abcde") && glob("lm_head", "lm_head"));
        assert!(!glob("mtp*", "xmtp") && !glob("*.gate", "a.gate.x") && !glob("lm_head", "lm_head2"));
    }

    #[test]
    fn matching_formats_pass() {
        let s = QuantScheme::parse(&doc()).unwrap();
        s.check_linear("model.layers.0.linear_attn.out_proj", FP8_E4M3).unwrap();
        s.check_linear("model.layers.0.mlp.shared_expert.up_proj", NVFP4).unwrap();
        s.check_linear("lm_head", NVFP4).unwrap();
        s.check_linear("model.layers.9.self_attn.q_proj", BF16).unwrap(); // not mentioned: as stored
        s.check_linear("mtp.fc", BF16).unwrap(); // excluded
    }

    #[test]
    fn w4a4_and_unknown_algorithms_are_rejected() {
        let s = QuantScheme::parse(&doc()).unwrap();
        let e = s.check_linear("model.layers.1.self_attn.q_proj", NVFP4).unwrap_err().to_string();
        assert!(e.contains("4-bit activations") && e.contains("model.layers.1.self_attn.q_proj"), "{e}");
        let e = s.check_linear("model.layers.1.self_attn.k_proj", NVFP4).unwrap_err().to_string();
        assert!(e.contains("INT4_AWQ") && e.contains("does not implement"), "{e}");
    }

    #[test]
    fn declared_and_stored_formats_must_agree() {
        let s = QuantScheme::parse(&doc()).unwrap();
        let e = s.check_linear("model.layers.0.linear_attn.out_proj", BF16).unwrap_err().to_string();
        assert!(e.contains("FP8") && e.contains("plain float"), "{e}");
        let e = s.check_linear("lm_head", FP8_E4M3).unwrap_err().to_string();
        assert!(e.contains("W4A16_NVFP4"), "{e}");
    }

    #[test]
    fn a_global_algorithm_covers_every_layer_except_the_excluded() {
        let d: Value = serde_json::from_str(r#"{"quantization":{"quant_algo":"FP8","exclude_modules":["lm_head"]}}"#).unwrap();
        let s = QuantScheme::parse(&d).unwrap();
        s.check_linear("model.layers.0.mlp.up_proj", FP8_E4M3).unwrap();
        assert!(s.check_linear("model.layers.0.mlp.up_proj", BF16).is_err());
        s.check_linear("lm_head", BF16).unwrap();
    }

    #[test]
    fn kv_cache_deviation_is_a_note() {
        let s = QuantScheme::parse(&doc()).unwrap();
        assert!(s.kv_note(DType::Bf16).unwrap().contains("kv_cache_quant_algo FP8"));
        assert!(s.kv_note(DType::Fp8E4m3).is_none());
        let none: Value = serde_json::from_str(r#"{"quantization":{"quant_algo":"FP8"}}"#).unwrap();
        assert!(QuantScheme::parse(&none).unwrap().kv_note(DType::Bf16).is_none());
    }

    #[test]
    fn exl3_layers_are_exl3_or_plain() {
        let s = QuantScheme::parse_exl3(&serde_json::json!({"quant_method":"exl3","bits":4.0,"head_bits":6})).unwrap();
        s.check_linear("model.layers.0.mlp.up_proj", EXL3).unwrap();
        s.check_linear("lm_head", BF16).unwrap();
        let e = s.check_linear("model.layers.0.mlp.up_proj", NVFP4).unwrap_err().to_string();
        assert!(e.contains("EXL3"), "{e}");
        assert!(QuantScheme::parse_exl3(&serde_json::json!({"quant_method":"exl3","codebook":"2mad"})).is_err());
        assert!(s.kv_note(DType::Bf16).is_none());
    }

    #[test]
    fn the_real_qwen36_scheme_accepts_what_the_loader_finds() {
        let t = include_str!("../../../intarsia-models/tests/fixtures/hf_quant_config_qwen36_trimmed.json");
        let s = QuantScheme::parse(&serde_json::from_str(t).unwrap()).unwrap();
        s.check_linear("model.language_model.layers.0.linear_attn.out_proj", FP8_E4M3).unwrap();
        s.check_linear("model.language_model.layers.0.mlp.shared_expert.gate_proj", NVFP4).unwrap();
        assert!(s.check_linear("model.language_model.layers.0.linear_attn.out_proj", NVFP4).is_err());
        assert!(s.kv_note(DType::Bf16).is_some());
        assert!(s.is_excluded("mtp.layers.0.mlp.gate"));
    }
}
