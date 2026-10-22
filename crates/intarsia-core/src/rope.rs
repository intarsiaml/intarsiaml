//! RoPE frequency scaling (`rope_type` llama3 and yarn), computed once on the host.
//!
//! The plain rotary embedding rotates pair `i` of a head by `pos * theta^(-2i/rd)`. A scaled embedding only changes
//! that per-pair frequency (and, for yarn, multiplies cos and sin by an "attention factor"), so a model with
//! scaling passes the kernels the finished table `inv_freq[rd/2]` ([`Op::RopeScaled`](crate::ops::Op)) and nothing
//! else changes: the angle of pair `i` is `pos * inv_freq[i]`. Unscaled models keep the original `Rope` op and never
//! touch this module's output.
//!
//! The formulas are Hugging Face transformers' `modeling_rope_utils._compute_llama3_parameters` and
//! `_compute_yarn_parameters` (checked against transformers 5.19; the config keys and fall-backs are the ones
//! `standardize_rope_params` applies). They are evaluated in f64 and rounded once to f32.

use crate::error::{Error, Result};

/// A rotary-embedding frequency scaling read from `config.json` (`rope_parameters` / `rope_scaling`).
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize)]
pub enum RopeScaling {
    /// Plain `theta^(-2i/rd)` frequencies (`rope_type` absent, `default`, or a null scaling block).
    #[default]
    None,
    /// Llama 3.1 / 3.2 / 3.3: high frequencies kept, low ones divided by `factor`, a smooth blend in between.
    Llama3 { factor: f64, low_freq_factor: f64, high_freq_factor: f64, original_max: f64 },
    /// YaRN: per-dimension blend of interpolated (`/ factor`) and extrapolated frequencies, plus `attention_factor`
    /// on cos / sin.
    Yarn {
        factor: f64,
        original_max: f64,
        beta_fast: f64,
        beta_slow: f64,
        truncate: bool,
        /// `attention_factor` from the config, when it gives one; else derived from `factor` / `mscale`.
        attention_factor: Option<f64>,
        mscale: Option<f64>,
        mscale_all_dim: Option<f64>,
    },
}

/// The finished frequency table of a scaled rotary embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeFreqs {
    /// `[rotary_dim / 2]` inverse frequencies.
    pub inv_freq: Vec<f32>,
    /// Multiplier on cos and sin (1.0 for llama3).
    pub attention_factor: f32,
}

impl RopeScaling {
    pub fn is_scaled(&self) -> bool {
        !matches!(self, RopeScaling::None)
    }

    /// The `rope_type` name, as in `config.json`.
    pub fn name(&self) -> &'static str {
        match self {
            RopeScaling::None => "default",
            RopeScaling::Llama3 { .. } => "llama3",
            RopeScaling::Yarn { .. } => "yarn",
        }
    }

    /// Parse one rope block (`rope_parameters` or `rope_scaling` of the text config). `top` is the text config
    /// itself (for a top-level `original_max_position_embeddings`) and `max_positions` its
    /// `max_position_embeddings`. A block that is absent, null, or has no / `default` type is [`RopeScaling::None`];
    /// `linear`, `dynamic` and `longrope` are rejected by [`check_rope_type`].
    pub fn from_config(block: &serde_json::Value, top: &serde_json::Value, max_positions: usize) -> Result<RopeScaling> {
        let Some(ty) = rope_type(block) else { return Ok(RopeScaling::None) };
        let f = |k: &str| block.get(k).and_then(|v| v.as_f64());
        let need = |k: &str| f(k).ok_or_else(|| Error::model(format!("config rope_type \"{ty}\" needs a numeric \"{k}\"")));
        let original_max = |block: &serde_json::Value| -> f64 {
            block
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_f64())
                .or_else(|| top.get("original_max_position_embeddings").and_then(|v| v.as_f64()))
                .unwrap_or(max_positions as f64)
        };
        match ty {
            "default" => Ok(RopeScaling::None),
            "llama3" => {
                let s = RopeScaling::Llama3 {
                    factor: need("factor")?,
                    low_freq_factor: need("low_freq_factor")?,
                    high_freq_factor: need("high_freq_factor")?,
                    original_max: original_max(block),
                };
                s.validate()?;
                Ok(s)
            }
            "yarn" => {
                let original = original_max(block);
                // HF: `factor` may be absent (DeepSeek-style configs); it is then max_positions / original
                let factor = match f("factor") {
                    Some(v) => v,
                    None => max_positions as f64 / original,
                };
                let s = RopeScaling::Yarn {
                    factor,
                    original_max: original,
                    // HF `rope_parameters_dict.get("beta_fast") or 32`: null / 0 fall back to the default
                    beta_fast: f("beta_fast").filter(|v| *v != 0.0).unwrap_or(32.0),
                    beta_slow: f("beta_slow").filter(|v| *v != 0.0).unwrap_or(1.0),
                    truncate: block.get("truncate").and_then(|v| v.as_bool()).unwrap_or(true),
                    attention_factor: f("attention_factor"),
                    mscale: f("mscale"),
                    mscale_all_dim: f("mscale_all_dim"),
                };
                s.validate()?;
                Ok(s)
            }
            other => Err(Error::Unsupported(format!("config rope_type \"{other}\" (see rope::check_rope_type)"))),
        }
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::model(format!("config rope_type \"{}\": {m}", self.name())));
        match *self {
            RopeScaling::None => Ok(()),
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max } => {
                if !(factor >= 1.0) || !(low_freq_factor > 0.0) || !(high_freq_factor > low_freq_factor) || !(original_max > 0.0) {
                    return bad("needs factor >= 1, 0 < low_freq_factor < high_freq_factor, original_max_position_embeddings > 0");
                }
                Ok(())
            }
            RopeScaling::Yarn { factor, original_max, beta_fast, beta_slow, .. } => {
                if !(factor > 0.0) || !(original_max > 0.0) || !(beta_fast > 0.0) || !(beta_slow > 0.0) {
                    return bad("needs factor, original_max_position_embeddings, beta_fast, beta_slow > 0");
                }
                Ok(())
            }
        }
    }

    /// The frequency table for a rotary embedding of `rotary_dim` values (even, > 0) per head with base `theta`.
    /// `None` for [`RopeScaling::None`] (the plain `Rope` op computes its own).
    pub fn freqs(&self, theta: f64, rotary_dim: usize) -> Option<RopeFreqs> {
        assert!(rotary_dim > 0 && rotary_dim % 2 == 0, "rotary_dim {rotary_dim} must be even and > 0");
        let half = rotary_dim / 2;
        let dim = rotary_dim as f64;
        // HF: 1 / base ** (arange(0, dim, 2) / dim)
        let base_inv: Vec<f64> = (0..half).map(|i| 1.0 / theta.powf(2.0 * i as f64 / dim)).collect();
        match *self {
            RopeScaling::None => None,
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max } => {
                let low_wavelen = original_max / low_freq_factor;
                let high_wavelen = original_max / high_freq_factor;
                let inv: Vec<f32> = base_inv
                    .iter()
                    .map(|&f| {
                        let wavelen = 2.0 * std::f64::consts::PI / f;
                        // wavelen < high: unchanged; wavelen > low: / factor; between: smooth blend
                        let scaled = if wavelen > low_wavelen { f / factor } else { f };
                        let smooth = (original_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                        let blended = (1.0 - smooth) * scaled / factor + smooth * scaled;
                        let medium = !(wavelen < high_wavelen) && !(wavelen > low_wavelen);
                        (if medium { blended } else { scaled }) as f32
                    })
                    .collect();
                Some(RopeFreqs { inv_freq: inv, attention_factor: 1.0 })
            }
            RopeScaling::Yarn { factor, original_max, beta_fast, beta_slow, truncate, attention_factor, mscale, mscale_all_dim } => {
                let get_mscale = |scale: f64, m: f64| if scale <= 1.0 { 1.0 } else { 0.1 * m * scale.ln() + 1.0 };
                let attention = attention_factor.unwrap_or_else(|| match (mscale, mscale_all_dim) {
                    (Some(m), Some(a)) if m != 0.0 && a != 0.0 => get_mscale(factor, m) / get_mscale(factor, a),
                    _ => get_mscale(factor, 1.0),
                });
                // dimension at which a rotation count `n` over `original_max` positions falls
                let correction_dim = |rotations: f64| dim * (original_max / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln());
                let (mut low, mut high) = (correction_dim(beta_fast), correction_dim(beta_slow));
                if truncate {
                    low = low.floor();
                    high = high.ceil();
                }
                let (low, mut high) = (low.max(0.0), high.min(dim - 1.0));
                if low == high {
                    high += 0.001; // HF: prevent the singularity
                }
                let inv: Vec<f32> = base_inv
                    .iter()
                    .enumerate()
                    .map(|(i, &extrapolation)| {
                        let interpolation = extrapolation / factor;
                        let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                        let extrapolation_factor = 1.0 - ramp;
                        (interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor) as f32
                    })
                    .collect();
                Some(RopeFreqs { inv_freq: inv, attention_factor: attention as f32 })
            }
        }
    }
}

/// The `rope_type` (or legacy `type`) of one rope block; `None` for an absent / null block.
pub fn rope_type(block: &serde_json::Value) -> Option<&str> {
    if block.is_null() {
        return None;
    }
    Some(block.get("rope_type").or_else(|| block.get("type")).and_then(|t| t.as_str()).unwrap_or("default"))
}

/// Accept `default` / absent, `llama3` and `yarn`; refuse the rest (they would silently change every position).
pub fn check_rope_type(what: &str, block: &serde_json::Value) -> Result<()> {
    match rope_type(block) {
        None | Some("default") | Some("llama3") | Some("yarn") => Ok(()),
        Some(t) => Err(Error::Unsupported(format!(
            "config {what}.rope_type = \"{t}\" (implemented: default, llama3, yarn; linear / dynamic / longrope scaling \
             would silently change every position)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn close(a: f32, b: f64, rel: f64) {
        assert!(((a as f64) - b).abs() <= rel * b.abs().max(1e-30), "{a} vs {b}");
    }

    // Reference values below come from transformers 5.19 `ROPE_INIT_FUNCTIONS[...]` on a LlamaConfig with the same
    // head_dim / rope_theta / scaling block; the d16 llama3 entries are also derived by hand in the comments.

    #[test]
    fn llama3_head16() {
        // theta 5e5, factor 8, low 1, high 4, original 8192: low_wavelen 8192, high_wavelen 2048
        let s = RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max: 8192.0 };
        let f = s.freqs(500000.0, 16).unwrap();
        assert_eq!(f.attention_factor, 1.0);
        // i = 0..3: wavelen 6.28, 32.4, 167, 861 < 2048: untouched, 5e5^(-i/8)
        let want = [1.0, 0.193922758102417, 0.03760603070259094, 0.00729266507551074, 0.0005248460220173001, 3.428102354519069e-05, 6.647869668086059e-06, 1.289173155782919e-06];
        for (i, w) in want.iter().enumerate() {
            close(f.inv_freq[i], *w, 1e-6);
        }
        // i = 4: inv = 5e5^-0.5 = 1.41421e-3, wavelen 4442.9 (medium): smooth = (8192/4442.9 - 1)/3 = 0.28129;
        //   (1 - 0.28129) * 1.41421e-3 / 8 + 0.28129 * 1.41421e-3 = 5.2484e-4
        close(f.inv_freq[4], 5.2484e-4, 1e-3);
        // i = 5: inv = 5e5^-0.625 = 2.7424e-4, wavelen 22912 > 8192: / 8 = 3.428e-5
        close(f.inv_freq[5], 2.7424e-4 / 8.0, 1e-3);
    }

    #[test]
    fn llama3_head128_llama31() {
        let s = RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max: 8192.0 };
        let f = s.freqs(500000.0, 128).unwrap();
        assert_eq!(f.inv_freq.len(), 64);
        for (i, w) in [(0, 1.0), (1, 0.8146172165870667), (10, 0.12868738174438477), (20, 0.016560440883040428), (21, 0.013490419834852219), (25, 0.005940730683505535), (30, 0.0013718936825171113), (31, 0.0008567514596506953), (32, 0.0005248460220173001), (40, 3.428102354519069e-05), (63, 3.068925877869333e-07)] {
            close(f.inv_freq[i], w, 1e-6);
        }
    }

    #[test]
    fn yarn_head16() {
        // theta 1e4, factor 4, original 4096, beta_fast 32, beta_slow 1, dim 16, by hand:
        //   correction_dim(n) = 16 ln(4096 / (2 pi n)) / (2 ln 1e4): n = 32 -> 2.6176 (floor 2), n = 1 -> 5.6286 (ceil 6)
        //   ramp(i) = clamp((i - 2) / 4, 0, 1); inv = base * (1 - ramp) + base / 4 * ramp, base = 1e4^(-i/8)
        //   i = 0..2: ramp 0 -> base (1, 0.316228, 0.1); i = 3: ramp 0.25 -> 0.031623 * 0.75 + 0.0079057 * 0.25 = 0.025693
        //   i = 4: 0.01 * 0.5 + 0.0025 * 0.5 = 0.00625; i = 5: 3.1623e-3 * 0.25 + 7.9057e-4 * 0.75 = 1.3835e-3
        //   i = 6, 7: ramp 1 -> base / 4 = 2.5e-4, 7.9057e-5; attention_factor = 0.1 ln 4 + 1 = 1.138629
        let s = RopeScaling::Yarn { factor: 4.0, original_max: 4096.0, beta_fast: 32.0, beta_slow: 1.0, truncate: true, attention_factor: None, mscale: None, mscale_all_dim: None };
        let f = s.freqs(10000.0, 16).unwrap();
        close(f.attention_factor, 1.0 + 0.1 * 4.0f64.ln(), 1e-6);
        let want = [1.0, 0.3162277638912201, 0.10000000149011612, 0.025693506002426147, 0.00624999962747097, 0.0013834965648129582, 0.0002500000118743628, 7.905694656074047e-05];
        for (i, w) in want.iter().enumerate() {
            close(f.inv_freq[i], *w, 1e-6);
        }
    }

    #[test]
    fn yarn_head128_qwen_style() {
        let s = RopeScaling::Yarn { factor: 4.0, original_max: 32768.0, beta_fast: 32.0, beta_slow: 1.0, truncate: true, attention_factor: None, mscale: None, mscale_all_dim: None };
        let f = s.freqs(1000000.0, 128).unwrap();
        close(f.attention_factor, 1.0 + 0.1 * 4.0f64.ln(), 1e-6);
        for (i, w) in [(0, 1.0), (1, 0.8058422207832336), (10, 0.11547820270061493), (20, 0.01333521492779255), (21, 0.010746078565716743), (25, 0.004131738096475601), (30, 0.0010643609566614032), (31, 0.000802959781140089), (32, 0.0006029411451891065), (40, 4.4456985051510856e-05), (63, 3.102344408034696e-07)] {
            close(f.inv_freq[i], w, 1e-6);
        }
    }

    #[test]
    fn from_config_forms() {
        let ll = json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192});
        let top = json!({});
        assert_eq!(
            RopeScaling::from_config(&ll, &top, 131072).unwrap(),
            RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max: 8192.0 }
        );
        // legacy "type" key, original_max falls back to the top-level key, then to max_position_embeddings
        let y = json!({"type": "yarn", "factor": 4.0});
        let t = RopeScaling::from_config(&y, &json!({"original_max_position_embeddings": 32768}), 131072).unwrap();
        assert!(matches!(t, RopeScaling::Yarn { factor, original_max, beta_fast, beta_slow, truncate: true, .. } if factor == 4.0 && original_max == 32768.0 && beta_fast == 32.0 && beta_slow == 1.0));
        let t = RopeScaling::from_config(&y, &top, 40960).unwrap();
        assert!(matches!(t, RopeScaling::Yarn { original_max, .. } if original_max == 40960.0));
        // default forms
        assert_eq!(RopeScaling::from_config(&json!(null), &top, 1).unwrap(), RopeScaling::None);
        assert_eq!(RopeScaling::from_config(&json!({"rope_type": "default", "rope_theta": 1e6}), &top, 1).unwrap(), RopeScaling::None);
        assert_eq!(RopeScaling::from_config(&json!({"rope_theta": 1e6}), &top, 1).unwrap(), RopeScaling::None);
        // missing key / unsupported type
        assert!(RopeScaling::from_config(&json!({"rope_type": "llama3", "factor": 8.0}), &top, 1).is_err());
        assert!(check_rope_type("rope_scaling", &json!({"rope_type": "dynamic", "factor": 2.0})).is_err());
        assert!(check_rope_type("rope_scaling", &json!({"rope_type": "longrope"})).is_err());
        assert!(check_rope_type("rope_scaling", &json!({"rope_type": "yarn", "factor": 2.0})).is_ok());
    }

    #[test]
    fn unscaled_has_no_table() {
        assert!(RopeScaling::None.freqs(10000.0, 8).is_none());
    }
}
