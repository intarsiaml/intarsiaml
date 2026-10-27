//! The top-1 floor from the golden's own reference spread (lane QGC, docs/results/2026-10-27-nmd-new-model-drill-metal.md).
//!
//! The class floor "top-1 >= 99.5 %" (ARCHITECTURE section 6) assumes the engine can agree with the reference on 199 rows in
//! 200. For a bf16 model judged against an fp32 golden it cannot always: on Qwen3.5-4B HF's own bf16 forward agrees with its
//! own fp32 forward on only 98.96 % of the 1441 short-set rows, so no bf16 run passes. A near-tie between the top two logits
//! is a coin flip once activations are rounded to bf16, however correct the engine is.
//!
//! `tools/golden.py --dtype float32` therefore measures that spread when it writes the golden and records it in the
//! metadata key [`META_KEY`] ([`RefSpread`]: top-1 agreement of HF bf16 against HF fp32 over all rows, and the mean KL).
//! The gate then asks the engine for what the reference itself reaches, less a margin ([`RefSpread::floor`]):
//!
//! * **Margin = [`Z`] binomial standard errors of the agreement at the judged row count.** An engine exactly as noisy as
//!   HF's bf16 run is a draw from the same distribution; it falls below `agreement - Z x SE` in about 2 % of runs at
//!   Z = 2, and a better engine (the drill's: 99.72 % against 98.96 %) far less often. The standard error depends on the
//!   number of rows judged, so the 160 decode rows of a short set get a wider margin (0.8 points) than its 1441 prefill
//!   rows (0.27 points): one flip is 0.6 points there, and a fixed margin of a few tenths would fail on a single extra flip.
//! * **Never stricter than the class floor.** The calibrated floor is `min(class floor, agreement - margin)`: a reference
//!   that agrees with itself on 99.9 % of rows does not raise the bar above today's 99.5 %.
//! * **Not used when the spread is too wide to mean anything** (agreement below [`MIN_AGREEMENT`]): a reference that
//!   disagrees with itself on one row in twenty is a broken golden, not a noise level, and the class floor then fails the
//!   run loudly. `--min-top1` always wins over all of this, and a golden without the key keeps the class floor unchanged.

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

/// Metadata key inside a golden's `__metadata__` (a JSON string of a [`RefSpread`]).
pub const META_KEY: &str = "ref_spread";
pub const FORMAT: &str = "intarsia-ref-spread-v1";
/// Standard errors of margin below the reference agreement.
pub const Z: f64 = 2.0;
/// Below this agreement (percent) the spread is not used.
pub const MIN_AGREEMENT: f64 = 95.0;

/// How far the reference run at the engine's precision (HF bf16) is from the golden (HF fp32), over all rows of the golden.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RefSpread {
    #[serde(default = "default_format")]
    pub format: String,
    /// `hf_bf16_vs_fp32`.
    #[serde(default)]
    pub method: String,
    /// Percent of rows on which the bf16 argmax equals the fp32 golden's.
    pub top1_agree_pct: f64,
    #[serde(default)]
    pub top1_flips: Option<u64>,
    /// Rows behind the agreement.
    pub n: u64,
    /// `21-way` (top-k golden) or `full-vocab`: which KL `kl_mean` is.
    #[serde(default)]
    pub kl_kind: Option<String>,
    /// Mean KL(fp32 || bf16) over all rows, nats. Reported, not gated here (the KL floor has its own noise record).
    #[serde(default)]
    pub kl_mean: Option<f64>,
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub how: Option<String>,
}

fn default_format() -> String {
    FORMAT.into()
}

impl RefSpread {
    pub fn parse(json: &str) -> Result<RefSpread> {
        let s: RefSpread = serde_json::from_str(json).map_err(|e| anyhow!("ref_spread record: {e}"))?;
        if s.format != FORMAT {
            bail!("ref_spread record: format {:?}, expected {FORMAT:?}", s.format);
        }
        if !(s.top1_agree_pct.is_finite() && (0.0..=100.0).contains(&s.top1_agree_pct)) {
            bail!("ref_spread record: top1_agree_pct {} is not a percentage", s.top1_agree_pct);
        }
        if s.n == 0 {
            bail!("ref_spread record: n is 0");
        }
        Ok(s)
    }

    /// The top-1 floor (percent) for a run that judges `rows` rows, or `None` when the spread is too wide to use.
    /// `class_floor` is the floor in force without this record; the result is never above it.
    pub fn floor(&self, rows: usize, class_floor: f64) -> Option<f64> {
        if self.top1_agree_pct < MIN_AGREEMENT {
            return None;
        }
        let p = self.top1_agree_pct / 100.0;
        let se = 100.0 * (p * (1.0 - p) / rows.max(1) as f64).sqrt();
        Some(class_floor.min(self.top1_agree_pct - Z * se))
    }

    /// One line for the log.
    pub fn describe(&self) -> String {
        format!(
            "HF bf16 vs fp32 top-1 agreement {:.2}% over {} rows{}{}",
            self.top1_agree_pct,
            self.n,
            match (&self.kl_kind, self.kl_mean) {
                (Some(k), Some(v)) => format!(", {k} KL {v:.3e}"),
                _ => String::new(),
            },
            self.date.as_deref().map(|d| format!(" ({d})")).unwrap_or_default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spread(pct: f64, n: u64) -> RefSpread {
        RefSpread::parse(&format!(r#"{{"format":"{FORMAT}","method":"hf_bf16_vs_fp32","top1_agree_pct":{pct},"n":{n}}}"#)).unwrap()
    }

    #[test]
    fn the_qwen35_4b_drill_numbers_pass_and_a_real_regression_fails() {
        // docs/results/2026-10-27-nmd-new-model-drill-metal.md: HF bf16 vs fp32 98.96 % of 1441 rows; engine 99.72 % (prefill),
        // 98.75 % (2 flips in 160 decode rows). The absolute floor 99.5 failed both; the calibrated one passes both.
        let s = spread(98.96, 1441);
        let pre = s.floor(1441, 99.5).unwrap();
        assert!((98.4..98.45).contains(&pre), "{pre}");
        assert!(99.72 >= pre);
        let dec = s.floor(160, 99.5).unwrap();
        assert!((97.3..97.45).contains(&dec), "{dec}");
        assert!(98.75 >= dec, "2 flips in 160 rows pass");
        assert!(100.0 * (160.0 - 5.0) / 160.0 < dec, "5 flips in 160 rows (96.9 %) still fail");
        assert!(97.0 < pre, "a build at 97 % over the prefill rows fails");
    }

    #[test]
    fn the_floor_is_never_above_the_class_floor() {
        let s = spread(99.95, 10_000);
        assert_eq!(s.floor(10_000, 99.5), Some(99.5), "a quiet reference does not raise the bar");
        // a lower class floor (recipe [accuracy] min_top1) stays the cap
        assert_eq!(s.floor(10_000, 98.24), Some(98.24));
    }

    #[test]
    fn fewer_rows_widen_the_margin() {
        let s = spread(99.0, 5000);
        assert!(s.floor(100, 99.5).unwrap() < s.floor(1000, 99.5).unwrap());
        assert!(s.floor(1000, 99.5).unwrap() < s.floor(100_000, 99.5).unwrap());
    }

    #[test]
    fn a_reference_that_disagrees_with_itself_too_often_is_not_a_noise_level() {
        assert_eq!(spread(94.9, 1000).floor(1000, 99.5), None);
        assert!(spread(95.0, 1000).floor(1000, 99.5).is_some());
    }

    #[test]
    fn parse_rejects_what_it_cannot_use() {
        assert!(RefSpread::parse("{").is_err());
        assert!(RefSpread::parse(r#"{"format":"other","top1_agree_pct":99.0,"n":10}"#).is_err());
        assert!(RefSpread::parse(&format!(r#"{{"format":"{FORMAT}","top1_agree_pct":101.0,"n":10}}"#)).is_err());
        assert!(RefSpread::parse(&format!(r#"{{"format":"{FORMAT}","top1_agree_pct":99.0,"n":0}}"#)).is_err());
        // the record golden.py writes
        let s = RefSpread::parse(
            r#"{"format":"intarsia-ref-spread-v1","method":"hf_bf16_vs_fp32","top1_agree_pct":98.96,"top1_flips":15,"n":1441,"kl_kind":"full-vocab","kl_mean":0.00032,"date":"2026-10-27","how":"x"}"#,
        )
        .unwrap();
        assert!(s.describe().contains("98.96%") && s.describe().contains("full-vocab KL 3.200e-4"), "{}", s.describe());
    }
}
