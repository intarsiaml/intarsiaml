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
//! AMD Quark checkpoints declare their scheme in `config.json` `quantization_config` (`quant_method = "quark"`) instead:
//! [`QuantScheme::parse_quark`] maps its `global_quant_config` onto the same model (one global algorithm plus an
//! `exclude` list of module names). Only OCP MXFP4 weights (fp4, per_group 32, e8m0) are implemented; the activation
//! spec decides W4A4 (dynamic fp4 `input_tensors`) or W4A16 (none).

use super::{Mxfp4Act, QuantId, BF16, FP8_E4M3, MXFP4, NVFP4};
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
    /// Quark OCP MXFP4 weights with dynamic MXFP4 activations (`MXFP4_W4A4`, our name for the Quark spec).
    Mxfp4W4A4,
    /// Quark OCP MXFP4 weights, 16-bit activations (`MXFP4_W4A16`).
    Mxfp4W4A16,
    /// Anything else (INT4_AWQ, MXFP8, per-channel FP8, ...). Not implemented.
    Other,
}

impl Algo {
    pub fn parse(s: &str) -> Algo {
        match s.trim().to_ascii_uppercase().as_str() {
            "FP8" => Algo::Fp8W8A8,
            "W4A16_NVFP4" => Algo::Nvfp4W4A16,
            "MXFP4_W4A4" => Algo::Mxfp4W4A4,
            "MXFP4_W4A16" => Algo::Mxfp4W4A16,
            a if a == "NVFP4" || a.starts_with("NVFP4_") || a == "W4A4_NVFP4" || a == "W4A4_NVFP4_FP8" => Algo::Nvfp4W4A4,
            _ => Algo::Other,
        }
    }
    /// The tensor format a layer of this algorithm must be stored in.
    fn format(self) -> Option<QuantId> {
        match self {
            Algo::Fp8W8A8 => Some(FP8_E4M3),
            Algo::Nvfp4W4A16 => Some(NVFP4),
            Algo::Mxfp4W4A4 | Algo::Mxfp4W4A16 => Some(MXFP4),
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

    /// Parse a Quark `quantization_config` (the object from `config.json`, `quant_method = "quark"`).
    ///
    /// The global weight spec must be OCP MXFP4 (`dtype fp4`, `qscheme per_group`, `group_size 32`, `scale_format e8m0`)
    /// for the declared algorithm to be one we implement; anything else is recorded verbatim (`QUARK_<dtype>...`) and
    /// rejected per layer by [`QuantScheme::check_linear`]. Per-layer overrides (`layer_quant_config`,
    /// `layer_type_quant_config`) are not implemented and are an error when non-empty. `exclude` entries name modules,
    /// some with a `.weight` suffix; the suffix is dropped.
    pub fn parse_quark(qc: &Value) -> Result<QuantScheme> {
        let method = qc.get("quant_method").and_then(Value::as_str).unwrap_or("");
        if method != "quark" {
            return Err(Error::model(format!("quantization_config.quant_method is '{method}', not 'quark'")));
        }
        for k in ["layer_quant_config", "layer_type_quant_config"] {
            if qc.get(k).and_then(Value::as_object).is_some_and(|o| !o.is_empty()) {
                return Err(Error::Unsupported(format!("Quark quantization_config.{k} (per-layer overrides) is not implemented")));
            }
        }
        let g = qc
            .get("global_quant_config")
            .filter(|v| !v.is_null())
            .ok_or_else(|| Error::model("Quark quantization_config has no global_quant_config"))?;
        let field = |spec: &Value, k: &str| spec.get(k).map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string())).unwrap_or_default();
        let w = g.get("weight").filter(|v| !v.is_null()).ok_or_else(|| Error::model("Quark global_quant_config has no weight spec"))?;
        let is_mxfp4 = |spec: &Value| {
            field(spec, "dtype") == "fp4"
                && field(spec, "qscheme") == "per_group"
                && spec.get("group_size").and_then(Value::as_u64) == Some(32)
                && field(spec, "scale_format") == "e8m0"
        };
        let input = g.get("input_tensors").filter(|v| !v.is_null());
        let algo = if !is_mxfp4(w) {
            format!("QUARK_{}_{}_g{}_{}", field(w, "dtype"), field(w, "qscheme"), field(w, "group_size"), field(w, "scale_format"))
        } else {
            match input {
                None => "MXFP4_W4A16".to_string(),
                Some(i) if is_mxfp4(i) && i.get("is_dynamic").and_then(Value::as_bool) == Some(true) => {
                    // the scale rule must be the one `mxfp4::e8m0_scale_even` implements
                    match field(i, "scale_calculation_mode").as_str() {
                        "even" | "" | "null" => "MXFP4_W4A4".to_string(),
                        other => format!("MXFP4_W4A4_scale_{other}"),
                    }
                }
                Some(i) => format!("MXFP4_W4_A_{}_{}", field(i, "dtype"), field(i, "qscheme")),
            }
        };
        let exclude = qc
            .get("exclude")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.strip_suffix(".weight").unwrap_or(s).to_string())).collect())
            .unwrap_or_default();
        Ok(QuantScheme { quant_algo: Some(algo), kv_cache_quant_algo: None, layers: BTreeMap::new(), exclude })
    }

    /// The MXFP4 activation mode the checkpoint declares (`None` when it declares no MXFP4 algorithm).
    pub fn mxfp4_act(&self) -> Option<Mxfp4Act> {
        match Algo::parse(self.quant_algo.as_deref()?) {
            Algo::Mxfp4W4A4 => Some(Mxfp4Act::Dynamic),
            Algo::Mxfp4W4A16 => Some(Mxfp4Act::None),
            _ => None,
        }
    }

    /// A deviation note when the MXFP4 activation mode we run differs from the declared one.
    pub fn mxfp4_act_note(&self, ours: Mxfp4Act) -> Option<String> {
        let declared = self.mxfp4_act()?;
        (declared != ours).then(|| {
            format!(
                "checkpoint declares MXFP4 activations '{}' ({}); running '{}' (opt-in --mxfp4-act): a deviation from the declared \
                 scheme and from a reference that runs it",
                declared.name(),
                self.quant_algo.as_deref().unwrap_or(""),
                ours.name()
            )
        })
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
                 (supported: FP8, W4A16_NVFP4, Quark MXFP4 with dynamic fp4 or 16-bit activations)"
            ))),
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

    fn quark(input: &str) -> Value {
        serde_json::from_str(&format!(
            r#"{{"quant_method":"quark","layer_quant_config":{{}},"layer_type_quant_config":{{}},
            "global_quant_config":{{"weight":{{"dtype":"fp4","qscheme":"per_group","group_size":32,"scale_format":"e8m0","is_dynamic":false}},
            "input_tensors":{input}}},
            "exclude":["lm_head","mtp.fc.weight","model.visual.blocks.0.attn.qkv"]}}"#
        ))
        .unwrap()
    }

    #[test]
    fn quark_mxfp4_w4a4_and_w4a16() {
        let a4 = r#"{"dtype":"fp4","qscheme":"per_group","group_size":32,"scale_format":"e8m0","is_dynamic":true,"scale_calculation_mode":"even"}"#;
        let s = QuantScheme::parse_quark(&quark(a4)).unwrap();
        assert_eq!(s.quant_algo.as_deref(), Some("MXFP4_W4A4"));
        assert_eq!(s.mxfp4_act(), Some(Mxfp4Act::Dynamic));
        s.check_linear("model.language_model.layers.0.mlp.gate_proj", MXFP4).unwrap();
        assert!(s.check_linear("model.language_model.layers.0.mlp.gate_proj", BF16).is_err());
        s.check_linear("lm_head", BF16).unwrap(); // excluded
        s.check_linear("mtp.fc", BF16).unwrap(); // excluded with its .weight suffix dropped
        assert!(s.mxfp4_act_note(Mxfp4Act::Dynamic).is_none());
        assert!(s.mxfp4_act_note(Mxfp4Act::None).unwrap().contains("deviation"));
        let s16 = QuantScheme::parse_quark(&quark("null")).unwrap();
        assert_eq!(s16.mxfp4_act(), Some(Mxfp4Act::None));
        s16.check_linear("model.language_model.layers.0.mlp.gate_proj", MXFP4).unwrap();
    }

    #[test]
    fn quark_other_schemes_are_rejected_per_layer() {
        let floor = r#"{"dtype":"fp4","qscheme":"per_group","group_size":32,"scale_format":"e8m0","is_dynamic":true,"scale_calculation_mode":"floor"}"#;
        let s = QuantScheme::parse_quark(&quark(floor)).unwrap();
        let e = s.check_linear("model.layers.0.mlp.up_proj", MXFP4).unwrap_err().to_string();
        assert!(e.contains("MXFP4_W4A4_scale_floor") && e.contains("does not implement"), "{e}");
        let mut v = quark("null");
        v["global_quant_config"]["weight"]["dtype"] = "int4".into();
        let s = QuantScheme::parse_quark(&v).unwrap();
        assert!(s.check_linear("model.layers.0.mlp.up_proj", MXFP4).is_err());
        let mut v = quark("null");
        v["layer_quant_config"] = serde_json::json!({"model.layers.0.mlp.up_proj": {}});
        assert!(QuantScheme::parse_quark(&v).is_err());
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
