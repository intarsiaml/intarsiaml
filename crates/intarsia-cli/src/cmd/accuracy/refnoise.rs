//! Reference-noise-aware KL gate (user decisions 2026-10-21 11:23 item 3 and 12:39 item 2, docs/decisions/LOG.md).
//!
//! A golden is itself a noisy measurement: vLLM does not reproduce its own prompt log-probs between two runs, and an
//! HF reference does not reproduce itself between bf16 and f32 in a scheme that rounds activations to fp4. A static
//! class floor below that noise cannot be met by any engine. The gate therefore carries a measured **reference noise**:
//!
//! * `ref_noise` record (this file's [`RefNoise`]): mean 21-way KL between the golden and a second reference run
//!   (`run_to_run`) or a reference of another precision (`precision_spread`), over all rows and, when the golden has
//!   chat prompts, over the scored rows; when, how and from which doc it was measured. It lives in the golden's
//!   `__metadata__` (key `ref_noise`, for goldens built with it) or in a tracked **sidecar**
//!   `golden/ref_noise/<golden file>.json`, bound to the golden by file name, FNV-1a hash, size and position count.
//!   The sidecar exists because rewriting a golden's metadata changes its hash and orphans every baseline.
//! * Absolute gate (golden vs engine): PASS when `KL <= max(static floor, factor x ref_noise)`. [`judge_kl`] is that
//!   rule as one pure function; it also says which of the two bound the verdict.
//! * Change vs base (`--self-golden --self-vs-ref-noise`): the change's KL against the base build must be at most
//!   `factor x ref_noise` ([`apply_to_self_checks`]).
//! * `intarsia ref-noise` measures a record offline (no model, no GPU) from two goldens of the same prompts.

use super::selfgolden::SelfCheck;
use super::{compare_pos, turns, Golden, PromptRef, RefRows, TAIL_K};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "intarsia-ref-noise-v1";
pub const VERDICT_FORMAT: &str = "intarsia-kl-verdict-v1";
/// Metadata key inside a golden's `__metadata__` (a JSON string of a [`RefNoise`]).
pub const META_KEY: &str = "ref_noise";

// ---------------------------------------------------------------------------
// the record
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// The same reference engine, run twice on the same ids (vLLM against itself).
    RunToRun,
    /// The same reference engine at two precisions (HF bf16 against HF f32).
    PrecisionSpread,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Method::RunToRun => "run_to_run",
            Method::PrecisionSpread => "precision_spread",
        }
    }
    pub fn parse(s: &str) -> Result<Method> {
        match s {
            "run_to_run" | "run-to-run" => Ok(Method::RunToRun),
            "precision_spread" | "precision-spread" => Ok(Method::PrecisionSpread),
            o => bail!("unknown reference-noise method '{o}' (run_to_run, precision_spread)"),
        }
    }
}

/// Which golden file a record was measured on. Every field is optional; the ones given must match.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct Binding {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// FNV-1a 64 of the whole golden file, 16 hex digits (`baseline::fnv1a64`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fnv1a64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Positions of all prompts (the `n` of the all-rows value).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<usize>,
}

/// The measured noise of one golden. Values are mean 21-way KL in nats.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RefNoise {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub golden: Binding,
    pub method: Method,
    /// Mean 21-way KL(golden || second reference) over ALL rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kl21_mean: Option<f64>,
    /// The same over the SCORED rows (outside the system and user turns of chat prompts; `turns.rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kl21_mean_scored: Option<f64>,
    /// Positions behind `kl21_mean` / `kl21_mean_scored`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_scored: Option<usize>,
    /// How many reference runs were compared (2 for one re-run).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runs: Option<usize>,
    /// UTC date of the measurement, `YYYY-MM-DD`.
    pub date: String,
    /// How it was measured: the second reference, the estimator, the tool, caveats.
    pub how: String,
    /// The estimator, when it is not the gate's own (`kl21-top20-vs-top20` ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator: Option<String>,
    /// The document or log the numbers come from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Per-golden factor on the noise (the gate's default is 1.0; see [`resolve_factor`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factor: Option<f64>,
    /// Per-golden factor of the change-vs-base worst-prompt check (default 3.0; see [`resolve_prompt_factor`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_factor: Option<f64>,
}

fn default_format() -> String {
    FORMAT.into()
}

impl RefNoise {
    pub fn parse(json: &str) -> Result<RefNoise> {
        let n: RefNoise = serde_json::from_str(json).map_err(|e| anyhow!("ref_noise record: {e}"))?;
        n.validate()?;
        Ok(n)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != FORMAT {
            bail!("ref_noise record: format {:?}, expected {FORMAT:?}", self.format);
        }
        if self.kl21_mean.is_none() && self.kl21_mean_scored.is_none() {
            bail!("ref_noise record: needs kl21_mean (all rows) or kl21_mean_scored");
        }
        for (name, v) in [("kl21_mean", self.kl21_mean), ("kl21_mean_scored", self.kl21_mean_scored)] {
            if let Some(v) = v {
                if !(v >= 0.0 && v.is_finite()) {
                    bail!("ref_noise record: {name} must be a finite number >= 0 (got {v})");
                }
            }
        }
        for (name, f) in [("factor", self.factor), ("prompt_factor", self.prompt_factor)] {
            if let Some(f) = f {
                if !(f > 0.0 && f.is_finite()) {
                    bail!("ref_noise record: {name} must be a positive number (got {f})");
                }
            }
        }
        if self.date.trim().is_empty() || self.how.trim().is_empty() {
            bail!("ref_noise record: `date` and `how` are required (a noise value without its provenance is not accepted)");
        }
        Ok(())
    }

    /// One line for the log.
    pub fn describe(&self) -> String {
        let f = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.3e}"));
        format!(
            "{}, KL21 all rows {} (n={}), scored rows {} (n={}); {}; {}{}",
            self.method.name(),
            f(self.kl21_mean),
            self.n.map_or("?".to_string(), |n| n.to_string()),
            f(self.kl21_mean_scored),
            self.n_scored.map_or("?".to_string(), |n| n.to_string()),
            self.date,
            self.source.as_deref().unwrap_or("no source cited"),
            self.estimator.as_deref().map(|e| format!("; estimator {e}")).unwrap_or_default(),
        )
    }
}

// ---------------------------------------------------------------------------
// finding the record, and binding it to the golden
// ---------------------------------------------------------------------------

/// What identifies the golden a record is read for. `size` / `positions` are `None` where they are not known (a
/// self-golden knows its source golden's name and hash only).
pub struct GoldenIdent {
    pub file: String,
    pub fnv1a64: String,
    pub size: Option<u64>,
    pub positions: Option<usize>,
}

/// A record and where it came from.
pub struct Loaded {
    pub noise: RefNoise,
    pub origin: String,
}

/// Refuse a record measured on another golden; return warnings for what could not be checked. A hash, a size or a
/// position count that is given and differs is an error: a regenerated golden needs a new measurement.
pub fn check_binding(n: &RefNoise, id: &GoldenIdent) -> Result<Vec<String>> {
    let b = &n.golden;
    let mut warn = Vec::new();
    if let Some(h) = &b.fnv1a64 {
        if !h.eq_ignore_ascii_case(&id.fnv1a64) {
            bail!("ref_noise was measured on golden hash {h}, this golden is {} ({}): measure it again (`intarsia ref-noise`)", id.fnv1a64, id.file);
        }
    } else {
        warn.push(format!("ref_noise names no golden hash, so it is matched by file name and position count only ({})", id.file));
    }
    if let (Some(s), Some(t)) = (b.size, id.size) {
        if s != t {
            bail!("ref_noise was measured on a golden of {s} bytes, this one has {t}");
        }
    }
    if let (Some(p), Some(q)) = (b.positions, id.positions) {
        if p != q {
            bail!("ref_noise was measured over {p} positions, this golden has {q}");
        }
    }
    if let Some(f) = &b.file {
        if *f != id.file && b.fnv1a64.is_none() {
            bail!("ref_noise was measured on golden '{f}', this is '{}'", id.file);
        }
    }
    Ok(warn)
}

/// `<dir>/ref_noise/<golden file>.json` for each place a tracked sidecar can sit: next to the golden, then
/// `golden/ref_noise/` of the working directory, then of this checkout.
pub fn sidecar_candidates(golden_dir: &Path, file: &str) -> Vec<PathBuf> {
    let name = format!("{file}.json");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden");
    [golden_dir.to_path_buf(), PathBuf::from("golden"), repo].into_iter().map(|d| d.join("ref_noise").join(&name)).collect()
}

fn read_record(p: &Path) -> Result<RefNoise> {
    let s = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    RefNoise::parse(&s).with_context(|| p.display().to_string())
}

/// The record for a golden: `--ref-noise FILE`, else the golden's own `ref_noise` metadata, else a sidecar. `None`:
/// the golden has no measured noise (the gate then runs on the static floor alone and says so). A record that exists
/// but belongs to another golden is an error, not a silent skip.
pub fn load(flag: Option<&Path>, embedded: Option<&str>, golden_dir: &Path, id: &GoldenIdent) -> Result<Option<(Loaded, Vec<String>)>> {
    let found: Option<Loaded> = if let Some(p) = flag {
        Some(Loaded { noise: read_record(p)?, origin: format!("--ref-noise {}", p.display()) })
    } else if let Some(j) = embedded {
        Some(Loaded { noise: RefNoise::parse(j).context("golden metadata `ref_noise`")?, origin: "golden metadata".into() })
    } else {
        match sidecar_candidates(golden_dir, &id.file).into_iter().find(|p| p.is_file()) {
            Some(p) => Some(Loaded { noise: read_record(&p)?, origin: format!("sidecar {}", p.display()) }),
            None => None,
        }
    };
    match found {
        Some(l) => {
            let w = check_binding(&l.noise, id).with_context(|| format!("ref_noise from {}", l.origin))?;
            Ok(Some((l, w)))
        }
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// the factor
// ---------------------------------------------------------------------------

/// Factor on the noise: `--ref-noise-factor`, else the record's `factor` (per golden), else the recipe's
/// `[accuracy] ref_noise_factor` (per model), else 1.0 (the decision's "no larger than"). Returns the value and its source.
pub fn resolve_factor(flag: Option<f64>, record: Option<f64>, recipe: Option<(f64, String)>) -> Result<(f64, String)> {
    resolve_named("--ref-noise-factor", 1.0, flag, record, recipe)
}

/// The change-vs-base worst-prompt check allows a prompt's mean 21-way KL up to `prompt_factor x ref_noise` (not times
/// the main factor): `--ref-noise-prompt-factor`, else the record's `prompt_factor`, else the recipe's
/// `[accuracy] ref_noise_prompt_factor`, else 3.0 (the user's decision of 2026-10-21 13:23).
pub fn resolve_prompt_factor(flag: Option<f64>, record: Option<f64>, recipe: Option<(f64, String)>) -> Result<(f64, String)> {
    resolve_named("--ref-noise-prompt-factor", DEFAULT_PROMPT_FACTOR, flag, record, recipe)
}

pub const DEFAULT_PROMPT_FACTOR: f64 = 3.0;

fn resolve_named(flag_name: &str, default: f64, flag: Option<f64>, record: Option<f64>, recipe: Option<(f64, String)>) -> Result<(f64, String)> {
    let (v, src) = if let Some(v) = flag {
        (v, flag_name.to_string())
    } else if let Some(v) = record {
        (v, "golden ref_noise record".to_string())
    } else if let Some(v) = recipe {
        (v.0, v.1)
    } else {
        (default, "default".to_string())
    };
    if !(v > 0.0 && v.is_finite()) {
        bail!("{flag_name}: the factor must be a positive number (got {v}, from {src})");
    }
    Ok((v, src))
}

// ---------------------------------------------------------------------------
// the verdict
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    /// The class floor (or `--max-mean-kl`) is the larger threshold.
    Static,
    /// `factor x ref_noise` is the larger threshold.
    RefNoise,
}

impl Bound {
    pub fn name(self) -> &'static str {
        match self {
            Bound::Static => "static floor",
            Bound::RefNoise => "ref noise",
        }
    }
}

/// The noise term of a verdict: the noise value and the KL measured over the same rows as the noise.
#[derive(Clone, Copy, Debug)]
pub struct NoiseTerm {
    pub value: f64,
    /// The engine's KL over `rows`.
    pub kl: f64,
    /// "scored rows" or "all rows".
    pub rows: &'static str,
}

/// One KL verdict, as printed and written to `--write-verdict`.
#[derive(Serialize, Clone, Debug)]
pub struct KlVerdict {
    pub mode: String,
    /// "21-way KL" or "full-vocab KL".
    pub kl_metric: String,
    /// The KL judged against the static floor (scored rows), or the change's KL (change vs base).
    pub kl: f64,
    pub static_floor: Option<f64>,
    pub ref_noise: Option<f64>,
    pub ref_noise_rows: Option<String>,
    /// The engine's KL over the rows `ref_noise` was measured on (equal to `kl` when those are the same rows).
    pub kl_on_noise_rows: Option<f64>,
    pub factor: f64,
    pub factor_source: String,
    /// `factor x ref_noise`.
    pub noise_threshold: Option<f64>,
    /// The larger of the two thresholds: what the verdict is "KL <= effective threshold" against.
    pub effective_threshold: f64,
    pub bound_by: Bound,
    pub pass: bool,
    /// `kl_on_noise_rows / ref_noise`: the factor at which the noise term would just pass.
    pub factor_needed: Option<f64>,
    /// Change vs base only: the worst prompt's mean KL against `prompt_factor x ref_noise`, a hard check.
    /// `pass` already includes it.
    pub worst_prompt: Option<PromptVerdict>,
}

/// The worst-prompt check of the change-vs-base mode.
#[derive(Serialize, Clone, Debug)]
pub struct PromptVerdict {
    pub prompt: String,
    /// The worst prompt's mean 21-way KL against the base build (all its rows).
    pub kl: f64,
    pub prompt_factor: f64,
    pub prompt_factor_source: String,
    /// `prompt_factor x ref_noise`.
    pub limit: f64,
    pub pass: bool,
    /// `kl / ref_noise`: the prompt factor at which this check would just pass.
    pub factor_needed: f64,
}

/// The worst prompt (largest mean KL) of a change against `limit = prompt_factor x ref_noise`. `worst` is the prompt's
/// name and mean KL; a NaN never passes.
pub fn judge_worst_prompt(worst: (&str, f64), ref_noise: f64, prompt_factor: f64, source: &str) -> Result<PromptVerdict> {
    if !(prompt_factor > 0.0 && prompt_factor.is_finite()) {
        bail!("the worst-prompt factor must be a positive number (got {prompt_factor})");
    }
    let limit = prompt_factor * ref_noise;
    Ok(PromptVerdict {
        prompt: worst.0.into(),
        kl: worst.1,
        prompt_factor,
        prompt_factor_source: source.into(),
        limit,
        pass: worst.1 <= limit,
        factor_needed: if ref_noise > 0.0 { worst.1 / ref_noise } else { f64::INFINITY },
    })
}

/// PASS when `KL <= max(static floor, factor x ref_noise)`. The two terms are compared on their own row sets (the
/// floor on the scored rows, the noise on the rows it was measured on; the same rows in every case but a noise record
/// that has no scored value for a golden whose mask hides rows). A NaN or infinite KL never passes. With no noise
/// the verdict is the static floor alone. With no static floor (change vs base) it is the noise alone.
pub fn judge_kl(mode: &str, kl_metric: &str, kl: f64, static_floor: Option<f64>, noise: Option<NoiseTerm>, factor: f64, factor_source: &str) -> Result<KlVerdict> {
    if !(factor > 0.0 && factor.is_finite()) {
        bail!("ref_noise factor must be a positive number (got {factor})");
    }
    if static_floor.is_none() && noise.is_none() {
        bail!("a KL verdict needs a static floor or a reference noise");
    }
    let noise_threshold = noise.map(|n| factor * n.value);
    let static_ok = static_floor.is_some_and(|s| kl <= s); // NaN / inf compare false
    let noise_ok = noise.zip(noise_threshold).is_some_and(|(n, t)| n.kl <= t);
    let (effective_threshold, bound_by) = match (static_floor, noise_threshold) {
        (Some(s), Some(t)) => {
            if t > s {
                (t, Bound::RefNoise)
            } else {
                (s, Bound::Static)
            }
        }
        (Some(s), None) => (s, Bound::Static),
        (None, Some(t)) => (t, Bound::RefNoise),
        (None, None) => unreachable!(),
    };
    Ok(KlVerdict {
        mode: mode.into(),
        kl_metric: kl_metric.into(),
        kl,
        static_floor,
        ref_noise: noise.map(|n| n.value),
        ref_noise_rows: noise.map(|n| n.rows.to_string()),
        kl_on_noise_rows: noise.map(|n| n.kl),
        factor,
        factor_source: factor_source.into(),
        noise_threshold,
        effective_threshold,
        bound_by,
        pass: static_ok || noise_ok,
        factor_needed: noise.filter(|n| n.value > 0.0).map(|n| n.kl / n.value),
        worst_prompt: None,
    })
}

impl KlVerdict {
    /// The extra gate line printed under the floor line.
    pub fn line(&self) -> String {
        let e = |v: f64| format!("{v:.3e}");
        let mut s = format!("{} KL threshold [{}]: ", self.mode, self.kl_metric);
        s += &match self.static_floor {
            Some(f) => format!("static floor {}", e(f)),
            None => "no static floor".to_string(),
        };
        match (self.ref_noise, &self.ref_noise_rows) {
            (Some(n), Some(rows)) => {
                s += &format!(
                    " | ref noise {} ({rows}) x factor {} ({}) = {}",
                    e(n),
                    self.factor,
                    self.factor_source,
                    e(self.noise_threshold.unwrap_or(f64::NAN))
                );
            }
            _ => s += " | no ref noise measured for this golden",
        }
        s += &format!(
            " => effective {} bound by {}; KL {} {}",
            e(self.effective_threshold),
            self.bound_by.name(),
            e(self.kl),
            // with a worst-prompt check `pass` is the combination: the KL part alone is read off the noise term
            match (&self.worst_prompt, self.noise_threshold, self.kl_on_noise_rows) {
                (Some(_), Some(t), Some(k)) => {
                    if k <= t {
                        "PASS"
                    } else {
                        "FAIL"
                    }
                }
                _ => {
                    if self.pass {
                        "PASS"
                    } else {
                        "FAIL"
                    }
                }
            }
        );
        if let Some(k) = self.factor_needed {
            s += &format!(" | KL / ref noise = {k:.3} on {} (the factor at which the noise term passes)", self.ref_noise_rows.as_deref().unwrap_or("?"));
        }
        if let Some(w) = &self.worst_prompt {
            s += &format!(
                " | worst prompt [{}] KL {} against {} x ref noise ({}) = {} {} (prompt factor at which it passes: {:.3})",
                w.prompt,
                e(w.kl),
                w.prompt_factor,
                w.prompt_factor_source,
                e(w.limit),
                if w.pass { "PASS" } else { "FAIL" },
                w.factor_needed
            );
            s += &format!(" => change vs base {}", if self.pass { "PASS" } else { "FAIL" });
        }
        s
    }
}

/// Written by `--write-verdict`.
#[derive(Serialize, Debug)]
pub struct VerdictFile {
    pub format: &'static str,
    /// `golden` (engine against a golden) or `change_vs_base` (`--self-golden --self-vs-ref-noise`).
    pub kind: &'static str,
    pub golden: Binding,
    pub ref_noise_origin: Option<String>,
    pub ref_noise: Option<RefNoise>,
    pub verdicts: Vec<KlVerdict>,
}

pub fn write_verdicts(path: &Path, v: &VerdictFile) -> Result<()> {
    let s = serde_json::to_string_pretty(v)?;
    std::fs::write(path, s + "\n").with_context(|| format!("writing {}", path.display()))
}

// ---------------------------------------------------------------------------
// change vs base
// ---------------------------------------------------------------------------

/// `--self-vs-ref-noise` (user decision 2026-10-21 13:23): two hard checks and non-finite rows.
/// * the change's mean 21-way KL against the base must be at most `mean_limit` (= factor x ref noise, all rows: a
///   self-golden compares every row);
/// * the worst prompt's mean KL must be at most `prompt_limit` (= prompt factor x ref noise, 3.0 by default);
/// * non-finite rows still fail.
///
/// The other budgets (largest position KL, flips, tail mass) stay in the output as reported numbers and no longer gate.
pub fn apply_to_self_checks(checks: Vec<SelfCheck>, mean_limit: f64, prompt_limit: f64) -> Vec<SelfCheck> {
    checks
        .into_iter()
        .map(|mut c| {
            if c.name == "mean KL(21-way)" {
                c.name = "mean KL(21-way) vs ref noise".into();
                c.limit = mean_limit;
                c.ok = c.value <= mean_limit; // NaN never passes
            } else if c.name.starts_with("worst prompt KL(21-way)") {
                c.name = format!("{} vs ref noise", c.name);
                c.limit = prompt_limit;
                c.ok = c.value <= prompt_limit;
            } else if c.name != "non-finite rows" {
                c.gated = false;
            }
            c
        })
        .collect()
}

// ---------------------------------------------------------------------------
// measuring: `intarsia ref-noise`
// ---------------------------------------------------------------------------

#[derive(clap::Args, Debug)]
pub struct RefNoiseArgs {
    /// The golden whose noise is measured: the reference an engine is gated against.
    #[arg(long)]
    pub golden: PathBuf,
    /// A second reference run of the same prompts: the same engine run again (`run_to_run`), or the same model in
    /// another precision (`precision_spread`). Same prompt names and ids; a top-k golden, or a full-vocabulary one when
    /// `--golden` is full-vocabulary too.
    #[arg(long)]
    pub other: PathBuf,
    /// `run_to_run` or `precision_spread`.
    #[arg(long, default_value = "run_to_run")]
    pub method: String,
    /// Model directory holding `tokenizer.json`: needed for the scored-rows value (it finds the turn markers of chat
    /// prompts). Without it only the all-rows value is measured.
    #[arg(long)]
    pub tokenizer_dir: Option<PathBuf>,
    /// Turn format of the chat prompts (`chatml`) when the golden's `template` metadata names none.
    #[arg(long)]
    pub chat_turns: Option<String>,
    /// Keep this many entries of the second run's top-k rows (the estimator `scripts/m3-accuracy/setkl.py` uses; the
    /// candidate's mass on a reference id outside its kept entries is its smallest kept probability).
    #[arg(long, default_value_t = 20)]
    pub trunc: usize,
    /// Free text: how the second run was made. Default: the two file names.
    #[arg(long)]
    pub how: Option<String>,
    /// The doc or log that explains the measurement.
    #[arg(long)]
    pub source: Option<String>,
    /// Per-golden factor to store in the record.
    #[arg(long)]
    pub factor: Option<f64>,
    /// Per-golden worst-prompt factor of the change-vs-base mode to store in the record (default 3.0).
    #[arg(long)]
    pub prompt_factor: Option<f64>,
    /// Write the record (a sidecar: `golden/ref_noise/<golden file>.json`). Default: print only.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// KL21(reference || candidate) of one position where both are top-k rows (ids best first; -1 / -1e30 pad). The
/// reference support is its first [`TAIL_K`] ids; the candidate keeps `trunc` entries, an id it does not list gets its
/// smallest kept probability, and each side's "other" bucket is `1 - sum` (at least 1e-12). The same formula as
/// `scripts/m3-accuracy/split.py::kl21_rows`, whose numbers round 7 reported (docs/results/2026-10-21-cuda-round7.md 3.2).
pub fn kl21_topk_pair(rid: &[i32], rlp: &[f32], cid: &[i32], clp: &[f32], trunc: usize) -> f64 {
    let valid = |ids: &[i32], lps: &[f32], k: usize| -> Vec<(i32, f64)> {
        ids.iter().zip(lps).take(k).filter(|(&i, &l)| i >= 0 && l > -1e29).map(|(&i, &l)| (i, l as f64)).collect()
    };
    let r = valid(rid, rlp, TAIL_K);
    let c = valid(cid, clp, trunc);
    if r.is_empty() || c.is_empty() {
        return f64::INFINITY;
    }
    let floor = c.last().map(|x| x.1.exp()).unwrap_or(0.0);
    let pc: Vec<f64> = r.iter().map(|(i, _)| c.iter().find(|(j, _)| j == i).map_or(floor, |x| x.1.exp())).collect();
    let pr: Vec<f64> = r.iter().map(|x| x.1.exp()).collect();
    let pr_rest = (1.0 - pr.iter().sum::<f64>()).max(1e-12);
    let pc_rest = (1.0 - pc.iter().sum::<f64>()).max(1e-12);
    let mut kl = 0.0;
    for (p, q) in pr.iter().zip(&pc) {
        kl += p * (p.ln() - q.max(1e-30).ln());
    }
    kl += pr_rest * (pr_rest.ln() - pc_rest.ln());
    kl
}

/// Scored-row masks per prompt, when a tokenizer and a turn format are available; `None` otherwise.
fn scored_masks(g: &Golden, prompts: &[(String, PromptRef)], args: &RefNoiseArgs) -> Result<Option<(Vec<Vec<bool>>, String)>> {
    let Some(dir) = &args.tokenizer_dir else { return Ok(None) };
    let from_golden = g.meta_str("template").and_then(|t| turns::format_named(t).map(|f| (f, "golden template".to_string())));
    let from_flag = match &args.chat_turns {
        Some(n) => Some((turns::format_named(n).ok_or_else(|| anyhow!("--chat-turns {n:?} is not a known turn format (chatml)"))?, "--chat-turns".to_string())),
        None => None,
    };
    let Some((fmt, src)) = from_golden.or(from_flag) else { return Ok(None) };
    use intarsia_core::engine::TextCodec;
    let codec = intarsia_models::tokenizer::HfCodec::load(dir, &intarsia_models::ChatTemplateSource::Auto).map_err(|e| anyhow!("{e}"))?;
    let resolved = fmt.resolve(&|t| codec.encode(t, false).map_err(|e| anyhow!("{e}")))?;
    let masks: Vec<Vec<bool>> = prompts
        .iter()
        .enumerate()
        .map(|(i, (_, p))| {
            let raw = g.prompt_meta(i).is_some_and(|m| m["kind"].as_str() == Some("raw"));
            if raw {
                vec![true; p.ids.len()]
            } else {
                resolved.scored_rows(&p.ids)
            }
        })
        .collect();
    Ok(Some((masks, format!("{} ({src})", fmt.name))))
}

/// KL21 of position `pos` of the reference against the candidate's row at the same position, and, when the reference
/// and the candidate hold the full vocabulary, the full-vocabulary KL too (printed, not stored).
fn pos_kl21(r: &PromptRef, c: &PromptRef, pos: usize, trunc: usize) -> Result<(f64, Option<f64>)> {
    match (&r.rows, &c.rows) {
        (RefRows::TopK { ids: ri, lps: rl, k: rk }, RefRows::TopK { ids: ci, lps: cl, k: ck }) => Ok((
            kl21_topk_pair(&ri[pos * rk..(pos + 1) * rk], &rl[pos * rk..(pos + 1) * rk], &ci[pos * ck..(pos + 1) * ck], &cl[pos * ck..(pos + 1) * ck], trunc),
            None,
        )),
        (_, RefRows::Full { lp, vocab }) => {
            // the gate's own estimator: the candidate's full row plays "ours" (a log-prob row is a valid logit row)
            let ours: Vec<f32> = lp[pos * vocab..(pos + 1) * vocab].iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            let st = compare_pos(r.row(pos), &ours)?;
            Ok((st.kl21, matches!(r.rows, RefRows::Full { .. }).then_some(st.kl)))
        }
        (RefRows::Full { .. }, RefRows::TopK { .. }) => {
            bail!("--golden is full-vocabulary but --other is top-k: the candidate's full row cannot be rebuilt; give a top-k --golden or a full --other")
        }
    }
}

fn utc_date(unix: u64) -> String {
    // days since 1970-01-01 -> civil date (H. Hinnant's algorithm)
    let z = (unix / 86400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Mean of `n` rows summing to `sum` (NaN for none).
fn mean(sum: f64, n: usize) -> f64 {
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

pub fn run(args: RefNoiseArgs) -> Result<()> {
    let method = Method::parse(&args.method)?;
    let g = Golden::open(&args.golden)?;
    let o = Golden::open(&args.other)?;
    let names = g.names();
    if names != o.names() {
        bail!("the two goldens do not hold the same prompts ({:?} vs {:?})", names, o.names());
    }
    let mut prompts = Vec::new();
    let mut others = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let (a, b) = (g.prompt(i)?, o.prompt(i)?);
        if a.ids != b.ids {
            bail!("prompt {name}: the two goldens tokenized it differently (or hold different ids)");
        }
        prompts.push((name.clone(), a));
        others.push(b);
    }
    let masks = scored_masks(&g, &prompts, &args)?;
    let (mut sum, mut n, mut sum_sc, mut n_sc, mut sum_full) = (0.0f64, 0usize, 0.0f64, 0usize, 0.0f64);
    let mut per: Vec<(String, usize, f64)> = Vec::new();
    for (pi, ((name, r), c)) in prompts.iter().zip(&others).enumerate() {
        let (mut ps, mut pn) = (0.0f64, 0usize);
        for pos in 0..r.ids.len() {
            let (kl, full) = pos_kl21(r, c, pos, args.trunc).with_context(|| format!("{name} position {pos}"))?;
            if let Some(f) = full {
                sum_full += f;
            }
            if !kl.is_finite() {
                bail!("{name} position {pos}: KL21 is not finite ({kl}); a row is empty or non-finite");
            }
            (sum, n, ps, pn) = (sum + kl, n + 1, ps + kl, pn + 1);
            if masks.as_ref().is_none_or(|m| m.0[pi][pos]) {
                (sum_sc, n_sc) = (sum_sc + kl, n_sc + 1);
            }
        }
        per.push((name.clone(), pn, ps / pn.max(1) as f64));
    }
    let file_of = |p: &Path| p.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let estimator = match (&prompts[0].1.rows, &others[0].rows) {
        (RefRows::TopK { .. }, RefRows::TopK { .. }) => format!("kl21-top{TAIL_K}-vs-top{} (scripts/m3-accuracy/split.py rule)", args.trunc),
        _ => "kl21 (the gate's own estimator; the second run's full row is 'ours')".to_string(),
    };
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let rec = RefNoise {
        format: FORMAT.into(),
        golden: Binding {
            file: Some(file_of(&args.golden)),
            fnv1a64: Some(format!("{:016x}", super::baseline::fnv1a64(&g.data))),
            size: Some(g.data.len() as u64),
            positions: Some(n),
        },
        method,
        kl21_mean: Some(mean(sum, n)),
        kl21_mean_scored: masks.as_ref().map(|_| mean(sum_sc, n_sc)),
        n: Some(n),
        n_scored: masks.as_ref().map(|_| n_sc),
        runs: Some(2),
        date: utc_date(secs),
        how: args.how.clone().unwrap_or_else(|| format!("intarsia ref-noise: KL21({} || {}), same ids", file_of(&args.golden), file_of(&args.other))),
        estimator: Some(estimator),
        source: args.source.clone(),
        factor: args.factor,
        prompt_factor: args.prompt_factor,
    };
    rec.validate()?;
    println!("ref noise of {} ({}): {}", file_of(&args.golden), method.name(), rec.describe());
    match &masks {
        Some((_, how)) => println!("scored rows: turn format {how}: {n_sc} of {n} rows"),
        None => println!("scored rows: not measured (give --tokenizer-dir, and --chat-turns when the golden names no template)"),
    }
    if matches!((&prompts[0].1.rows, &others[0].rows), (RefRows::Full { .. }, RefRows::Full { .. })) {
        println!("full-vocabulary KL over all rows: {:.4e} (for comparison with the full-vocabulary numbers in docs/accuracy.md; not stored)", mean(sum_full, n));
    }
    for (name, pn, kl) in &per {
        println!("  {name:<22} n={pn:<6} KL21 {kl:.4e}");
    }
    if let Some(out) = &args.out {
        if let Some(d) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(out, serde_json::to_string_pretty(&rec)? + "\n").with_context(|| format!("writing {}", out.display()))?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(all: Option<f64>, scored: Option<f64>) -> RefNoise {
        RefNoise {
            format: FORMAT.into(),
            golden: Binding::default(),
            method: Method::RunToRun,
            kl21_mean: all,
            kl21_mean_scored: scored,
            n: Some(100),
            n_scored: scored.map(|_| 80),
            runs: Some(2),
            date: "2026-10-21".into(),
            how: "test".into(),
            estimator: None,
            source: None,
            factor: None,
            prompt_factor: None,
        }
    }

    fn term(value: f64, kl: f64) -> Option<NoiseTerm> {
        Some(NoiseTerm { value, kl, rows: "scored rows" })
    }

    fn verdict(kl: f64, floor: Option<f64>, noise: Option<NoiseTerm>, factor: f64) -> KlVerdict {
        judge_kl("prefill", "21-way KL", kl, floor, noise, factor, "default").unwrap()
    }

    #[test]
    fn static_floor_alone_is_the_old_gate() {
        let v = verdict(2e-2, Some(3e-2), None, 1.0);
        assert!(v.pass && v.bound_by == Bound::Static && v.effective_threshold == 3e-2 && v.factor_needed.is_none());
        assert!(!verdict(3.1e-2, Some(3e-2), None, 1.0).pass);
        assert!(verdict(3e-2, Some(3e-2), None, 1.0).pass, "the floor is inclusive");
    }

    #[test]
    fn noise_above_the_floor_raises_the_threshold_and_binds() {
        // M1: floor 0.03, reference noise 0.197, engine 0.180
        let v = verdict(0.180, Some(0.03), term(0.197, 0.180), 1.0);
        assert!(v.pass);
        assert_eq!(v.bound_by, Bound::RefNoise);
        assert_eq!(v.effective_threshold, 0.197);
        assert!((v.factor_needed.unwrap() - 0.180 / 0.197).abs() < 1e-12);
        // the same engine on the decode rows: 0.274 > 0.197
        let d = verdict(0.274, Some(0.03), term(0.197, 0.274), 1.0);
        assert!(!d.pass && d.bound_by == Bound::RefNoise);
        assert!((d.factor_needed.unwrap() - 0.274 / 0.197).abs() < 1e-12);
        // a factor of 1.4 would admit it
        assert!(verdict(0.274, Some(0.03), term(0.197, 0.274), 1.4).pass);
    }

    #[test]
    fn noise_below_the_floor_leaves_the_floor_binding() {
        // 35B long, scored rows: floor 3e-2, vLLM noise 8.417e-3, HIP main 1.42e-2
        let v = verdict(1.42e-2, Some(3e-2), term(8.417e-3, 1.42e-2), 1.0);
        assert!(v.pass && v.bound_by == Bound::Static && v.effective_threshold == 3e-2);
        assert!((v.factor_needed.unwrap() - 1.42e-2 / 8.417e-3).abs() < 1e-12);
        // a KL above the floor that is inside factor x noise passes on the noise term; threshold = the larger of the two
        let w = verdict(5e-2, Some(3e-2), term(4e-2, 5e-2), 1.5);
        assert!(w.pass && w.bound_by == Bound::RefNoise && (w.effective_threshold - 6e-2).abs() < 1e-12);
    }

    #[test]
    fn tie_goes_to_the_static_floor() {
        let v = verdict(1e-3, Some(2e-2), term(2e-2, 1e-3), 1.0);
        assert_eq!(v.bound_by, Bound::Static);
    }

    #[test]
    fn nan_and_infinity_never_pass() {
        for bad in [f64::NAN, f64::INFINITY] {
            assert!(!verdict(bad, Some(3e-2), term(0.197, bad), 1.0).pass);
            assert!(!verdict(bad, None, term(0.197, bad), 1.0).pass);
        }
        // a NaN noise value cannot raise the threshold
        assert!(!verdict(0.1, Some(3e-2), term(f64::NAN, 0.1), 1.0).pass);
    }

    #[test]
    fn row_sets_are_judged_each_on_its_own() {
        // the floor judges the scored rows (1.0e-2 > 5e-3), the noise was measured on all rows where the engine is at 0.1
        let v = judge_kl("prefill", "21-way KL", 1e-2, Some(5e-3), Some(NoiseTerm { value: 0.2, kl: 0.1, rows: "all rows" }), 1.0, "default").unwrap();
        assert!(v.pass && v.ref_noise_rows.as_deref() == Some("all rows") && v.kl_on_noise_rows == Some(0.1));
        // and the engine misses the noise on its rows even though the scored KL is small
        let w = judge_kl("prefill", "21-way KL", 1e-2, Some(5e-3), Some(NoiseTerm { value: 0.2, kl: 0.5, rows: "all rows" }), 1.0, "default").unwrap();
        assert!(!w.pass);
    }

    #[test]
    fn change_vs_base_has_no_static_floor() {
        // CUDA round 7, lt against default, KL21 all rows: long 1.884e-2 against vLLM's own 2.218e-2, short 3.358e-3 against 3.161e-3
        assert!(judge_kl("prefill", "21-way KL", 1.884e-2, None, term(2.218e-2, 1.884e-2), 1.0, "default").unwrap().pass);
        let s = judge_kl("prefill", "21-way KL", 3.358e-3, None, term(3.161e-3, 3.358e-3), 1.0, "default").unwrap();
        assert!(!s.pass && s.bound_by == Bound::RefNoise && s.static_floor.is_none());
        assert!((s.factor_needed.unwrap() - 3.358e-3 / 3.161e-3).abs() < 1e-12);
    }

    #[test]
    fn invalid_factor_and_empty_verdicts_are_errors() {
        for f in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(judge_kl("m", "k", 1.0, Some(1.0), None, f, "x").is_err(), "{f}");
            assert!(resolve_factor(Some(f), None, None).is_err());
        }
        assert!(judge_kl("m", "k", 1.0, None, None, 1.0, "x").is_err());
    }

    #[test]
    fn zero_noise_gives_no_factor_needed() {
        let v = verdict(1e-3, Some(1e-2), term(0.0, 1e-3), 1.0);
        assert!(v.pass && v.factor_needed.is_none() && v.noise_threshold == Some(0.0));
    }

    #[test]
    fn factor_precedence_is_flag_record_recipe_default() {
        let recipe = Some((2.0, "recipe qwen3_5_moe".to_string()));
        assert_eq!(resolve_factor(Some(1.5), Some(3.0), recipe.clone()).unwrap(), (1.5, "--ref-noise-factor".into()));
        assert_eq!(resolve_factor(None, Some(3.0), recipe.clone()).unwrap().0, 3.0);
        assert_eq!(resolve_factor(None, None, recipe).unwrap(), (2.0, "recipe qwen3_5_moe".into()));
        assert_eq!(resolve_factor(None, None, None).unwrap(), (1.0, "default".into()));
    }

    #[test]
    fn parses_a_full_record_and_a_minimal_one() {
        let full = r#"{"format":"intarsia-ref-noise-v1","golden":{"file":"g.safetensors","fnv1a64":"00ff","size":10,"positions":1441},
            "method":"run_to_run","kl21_mean":3.161e-3,"kl21_mean_scored":3.010e-3,"n":1441,"n_scored":1300,"runs":2,
            "date":"2026-10-21","how":"vLLM re-run","estimator":"e","source":"docs/x.md","factor":1.5}"#;
        let r = RefNoise::parse(full).unwrap();
        assert_eq!((r.method, r.kl21_mean, r.kl21_mean_scored, r.factor), (Method::RunToRun, Some(3.161e-3), Some(3.010e-3), Some(1.5)));
        assert_eq!(r.golden.positions, Some(1441));
        let min = r#"{"method":"precision_spread","kl21_mean":0.197,"date":"2026-10-21","how":"HF bf16 vs f32"}"#;
        let m = RefNoise::parse(min).unwrap();
        assert_eq!((m.method, m.kl21_mean_scored, m.golden.fnv1a64.is_none()), (Method::PrecisionSpread, None, true));
        // round trip
        assert_eq!(RefNoise::parse(&serde_json::to_string(&r).unwrap()).unwrap(), r);
    }

    #[test]
    fn rejects_bad_records() {
        let base = |extra: &str| format!(r#"{{"method":"run_to_run","date":"2026-10-21","how":"h"{extra}}}"#);
        assert!(RefNoise::parse(&base("")).is_err(), "no value at all");
        assert!(RefNoise::parse(&base(r#","kl21_mean":-1.0"#)).is_err());
        assert!(RefNoise::parse(&base(r#","kl21_mean":1e999"#)).is_err(), "infinite");
        assert!(RefNoise::parse(&base(r#","kl21_mean":0.1,"factor":0"#)).is_err());
        assert!(RefNoise::parse(&base(r#","kl21_mean":0.1,"format":"other""#)).is_err());
        assert!(RefNoise::parse(r#"{"method":"guess","kl21_mean":0.1,"date":"d","how":"h"}"#).is_err());
        assert!(RefNoise::parse(r#"{"method":"run_to_run","kl21_mean":0.1,"date":"","how":"h"}"#).is_err(), "no provenance");
        assert!(RefNoise::parse(r#"{"method":"run_to_run","kl21_mean":0.1,"date":"d","how":" "}"#).is_err());
        assert!(RefNoise::parse("not json").is_err());
        assert!(RefNoise::parse(&base(r#","kl21_mean":0.1,"unknown":1"#)).is_ok(), "unknown keys are ignored");
    }

    #[test]
    fn binding_refuses_another_golden() {
        let id = GoldenIdent { file: "g.safetensors".into(), fnv1a64: "00000000000000ff".into(), size: Some(10), positions: Some(100) };
        let mut r = rec(Some(1e-3), None);
        // nothing bound: accepted with a warning
        assert_eq!(check_binding(&r, &id).unwrap().len(), 1);
        r.golden = Binding { file: Some("g.safetensors".into()), fnv1a64: Some("00000000000000FF".into()), size: Some(10), positions: Some(100) };
        assert!(check_binding(&r, &id).unwrap().is_empty(), "hash compares case-insensitively");
        r.golden.fnv1a64 = Some("0000000000000001".into());
        assert!(check_binding(&r, &id).is_err());
        r.golden.fnv1a64 = Some("00000000000000ff".into());
        r.golden.size = Some(11);
        assert!(check_binding(&r, &id).is_err());
        r.golden.size = None;
        r.golden.positions = Some(99);
        assert!(check_binding(&r, &id).is_err());
        // without a hash the file name decides
        r.golden = Binding { file: Some("other.safetensors".into()), ..Binding::default() };
        assert!(check_binding(&r, &id).is_err());
        // a self-golden knows no size or positions: those are not checked
        let sid = GoldenIdent { size: None, positions: None, ..id };
        r.golden = Binding { file: Some("g.safetensors".into()), fnv1a64: Some("00000000000000ff".into()), size: Some(10), positions: Some(100) };
        assert!(check_binding(&r, &sid).unwrap().is_empty());
    }

    #[test]
    fn load_prefers_flag_then_metadata_then_sidecar() {
        let dir = std::env::temp_dir().join(format!("intarsia-refnoise-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("ref_noise")).unwrap();
        let id = GoldenIdent { file: "g.safetensors".into(), fnv1a64: "00000000000000ff".into(), size: None, positions: None };
        let json = |kl: f64| format!(r#"{{"method":"run_to_run","kl21_mean":{kl},"date":"2026-10-21","how":"t","golden":{{"fnv1a64":"00000000000000ff"}}}}"#);
        assert!(load(None, None, &dir, &id).unwrap().is_none(), "no record, no error");
        std::fs::write(dir.join("ref_noise").join("g.safetensors.json"), json(0.3)).unwrap();
        let (l, w) = load(None, None, &dir, &id).unwrap().unwrap();
        assert_eq!(l.noise.kl21_mean, Some(0.3));
        assert!(l.origin.starts_with("sidecar") && w.is_empty());
        let (l, _) = load(None, Some(&json(0.2)), &dir, &id).unwrap().unwrap();
        assert_eq!((l.noise.kl21_mean, l.origin.as_str()), (Some(0.2), "golden metadata"));
        let flag = dir.join("flag.json");
        std::fs::write(&flag, json(0.1)).unwrap();
        let (l, _) = load(Some(&flag), Some(&json(0.2)), &dir, &id).unwrap().unwrap();
        assert_eq!(l.noise.kl21_mean, Some(0.1));
        // a record bound to another hash is an error from any source
        let other = GoldenIdent { fnv1a64: "0000000000000001".into(), ..id };
        assert!(load(None, None, &dir, &other).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn change_vs_base_gates_the_mean_kl_and_reports_the_rest() {
        let chk = |name: &str, value: f64, limit: f64| SelfCheck { name: name.into(), value, limit, ok: value <= limit, gated: true };
        let checks = vec![
            chk("mean KL(21-way)", 3.358e-3, 1e-4),
            chk("max position KL(21-way)", 0.5, 5e-3),
            chk("top-1 flips", 40.0, 7.0),
            chk("mean |tail diff|", 1e-3, 5e-4),
            chk("non-finite rows", 0.0, 0.0),
            chk("worst prompt KL(21-way) [p]", 0.1, 5e-4),
        ];
        assert!(!checks.iter().all(|c| c.ok), "the default budgets fail this change");
        // mean 3.358e-3 against 3.161e-3 fails; the worst prompt (0.1) is within 3 x 3.161e-3? no: 0.1 > 9.5e-3
        let r = apply_to_self_checks(checks.clone(), 3.161e-3, 3.0 * 3.161e-3);
        let gated: Vec<_> = r.iter().filter(|c| c.gated).map(|c| (c.name.as_str(), c.ok)).collect();
        assert_eq!(
            gated,
            vec![("mean KL(21-way) vs ref noise", false), ("non-finite rows", true), ("worst prompt KL(21-way) [p] vs ref noise", false)]
        );
        // inside the noise on the mean but the worst prompt over 3 x noise: still fails
        let r = apply_to_self_checks(checks.clone(), 1.07 * 3.161e-3, 3.0 * 3.161e-3);
        assert!(!r.iter().filter(|c| c.gated).all(|c| c.ok), "the worst prompt (0.1) is a hard check");
        // a worst prompt of 8e-3 is inside 3 x 3.161e-3 = 9.48e-3: only the mean KL, non-finite rows and the worst prompt gate
        let mut ok = checks.clone();
        ok[5] = chk("worst prompt KL(21-way) [p]", 8e-3, 5e-4);
        let r = apply_to_self_checks(ok.clone(), 1.07 * 3.161e-3, 3.0 * 3.161e-3);
        assert!(r.iter().filter(|c| c.gated).all(|c| c.ok));
        assert_eq!(r.iter().filter(|c| !c.gated).count(), 3, "largest position KL, flips and tail mass are reported only");
        // the 3.0 is a parameter: a prompt factor of 2 (limit 6.3e-3) fails the 8e-3 prompt
        assert!(!apply_to_self_checks(ok, 1.07 * 3.161e-3, 2.0 * 3.161e-3).iter().filter(|c| c.gated).all(|c| c.ok));
        // a non-finite row still fails the change
        let mut bad = checks;
        bad[4] = chk("non-finite rows", 3.0, 0.0);
        assert!(!apply_to_self_checks(bad, 1.0, 1.0).iter().filter(|c| c.gated).all(|c| c.ok));
        // NaN mean KL or NaN worst prompt never passes
        let nan = vec![chk("mean KL(21-way)", f64::NAN, 1e-4), chk("worst prompt KL(21-way) [p]", f64::NAN, 1e-4)];
        let r = apply_to_self_checks(nan, 1.0, 1.0);
        assert!(!r[0].ok && !r[1].ok);
    }

    fn lp(p: &[f64]) -> Vec<f32> {
        p.iter().map(|x| x.ln() as f32).collect()
    }

    #[test]
    fn topk_pair_kl_matches_the_hand_computation() {
        let (rid, rlp) = ([1, 2, 3], lp(&[0.5, 0.3, 0.1])); // other bucket 0.1
        // identical rows: zero
        assert!(kl21_topk_pair(&rid, &rlp, &rid, &rlp, 20).abs() < 1e-6);
        // candidate [0.4, 0.4, 0.1], other 0.1
        let clp = lp(&[0.4, 0.4, 0.1]);
        let want = 0.5 * (0.5f64 / 0.4).ln() + 0.3 * (0.3f64 / 0.4).ln() + 0.1 * (0.1f64 / 0.1).ln() + 0.1 * (0.1f64 / 0.1).ln();
        assert!((kl21_topk_pair(&rid, &rlp, &rid, &clp, 20) - want).abs() < 1e-6, "{want}");
        // a reference id the candidate does not list gets the candidate's smallest kept probability
        let (cid, clp) = ([1, 2, 9], lp(&[0.5, 0.3, 0.05])); // id 3 -> 0.05, other = 1 - (0.5+0.3+0.05) = 0.15
        let want = 0.1 * (0.1f64 / 0.05).ln() + 0.1 * (0.1f64 / 0.15).ln();
        assert!((kl21_topk_pair(&rid, &rlp, &cid, &clp, 20) - want).abs() < 1e-6);
        // padding is ignored, an empty row is not finite
        let (pid, plp) = ([1, 2, 3, -1], [rlp[0], rlp[1], rlp[2], -1e30]);
        assert!(kl21_topk_pair(&pid, &plp, &rid, &rlp, 20).abs() < 1e-6);
        assert!(kl21_topk_pair(&[-1], &[-1e30], &rid, &rlp, 20).is_infinite());
    }

    /// One candidate row of `docs/results/2026-10-21-cuda-round7/raw/05-kl-table.txt`.
    struct TableRow {
        block: String,
        name: String,
        all: f64,
        n_all: usize,
        excl: f64,
        n_excl: usize,
        /// Per-prompt mean KL21 over all rows, in the log's order.
        prompts: Vec<(String, f64)>,
    }

    fn parse_kl_table(text: &str) -> Vec<TableRow> {
        let mut block = String::new();
        let mut rows = Vec::new();
        for l in text.lines() {
            if let Some(b) = l.strip_prefix("== ") {
                block = b.to_string();
                continue;
            }
            let Some((name, rest)) = l.split_once(char::is_whitespace) else { continue };
            // "ALL 3.161e-03 (n=1441)  excl. system/user-turn rows 3.010e-03 (n=1279)  | ..."
            let nums: Vec<&str> = rest.split(|c: char| c.is_whitespace() || c == '(' || c == ')').filter(|t| !t.is_empty()).collect();
            let pos = |key: &str| nums.iter().position(|t| *t == key);
            let (Some(a), Some(e)) = (pos("ALL"), pos("rows")) else { continue };
            let n = |t: &str| t.trim_start_matches("n=").parse::<usize>().unwrap();
            let prompts = rest
                .split_once('|')
                .map(|(_, t)| t.split_whitespace().filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.to_string(), v.parse::<f64>().unwrap())).collect())
                .unwrap_or_default();
            rows.push(TableRow {
                block: block.clone(),
                name: name.to_string(),
                all: nums[a + 1].parse().unwrap(),
                n_all: n(nums[a + 2]),
                excl: nums[e + 1].parse().unwrap(),
                n_excl: n(nums[e + 2]),
                prompts,
            });
        }
        rows
    }

    fn sidecar(file: &str) -> RefNoise {
        read_record(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/ref_noise").join(format!("{file}.json"))).unwrap()
    }

    fn term_rows(value: f64, kl: f64, rows: &'static str) -> Option<NoiseTerm> {
        Some(NoiseTerm { value, kl, rows })
    }

    /// CUDA round 7 re-scored with the reference-noise gate, from the committed log (`raw/05-kl-table.txt`) and the
    /// back-filled sidecars. `--nocapture` prints the table that docs/results/2026-10-21-a2-ref-noise-gate.md quotes.
    #[test]
    fn rescore_cuda_round7_from_the_committed_table() {
        let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/results/2026-10-21-cuda-round7/raw/05-kl-table.txt")).unwrap();
        let rows = parse_kl_table(&text);
        assert_eq!(rows.len(), 14);
        let (short, long) = (sidecar("qwen36-35b-a3b-nvfp4-vllm-top20.safetensors"), sidecar("qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors"));
        // the 4-bit class floor, scored rows (`GateClass::Fp4`: 3e-2)
        let floor = 3e-2;
        let mut seen = (0, 0);
        for r in &rows {
            let (is_long, noise) = if r.block.starts_with("35b-long") { (true, &long) } else { (false, &short) };
            let which = if is_long { "long" } else { "short" };
            if r.block.contains("stored vLLM golden") {
                if r.name.starts_with("vllm2") {
                    // the log's own "vLLM re-run" row is the record
                    assert_eq!((r.all, r.n_all, r.excl, r.n_excl), (noise.kl21_mean.unwrap(), noise.n.unwrap(), noise.kl21_mean_scored.unwrap(), noise.n_scored.unwrap()), "{}", r.name);
                    continue;
                }
                // engine against the golden, scored rows: class floor 3e-2 against factor x the scored noise
                let nz = noise.kl21_mean_scored.unwrap();
                let v = judge_kl("prefill", "21-way KL", r.excl, Some(floor), term_rows(nz, r.excl, "scored rows"), 1.0, "default").unwrap();
                let alone = judge_kl("prefill", "21-way KL", r.excl, None, term_rows(nz, r.excl, "scored rows"), 1.0, "default").unwrap();
                println!(
                    "ABS  {which:<5} {:<34} scored KL {:.3e}  ref noise {nz:.3e}  KL/noise {:.2}  gate {} (bound by {})  noise alone {}",
                    r.name.trim_end_matches(".safetensors"),
                    r.excl,
                    r.excl / nz,
                    if v.pass { "PASS" } else { "FAIL" },
                    v.bound_by.name(),
                    if alone.pass { "PASS" } else { "FAIL" }
                );
                assert!(v.pass && v.bound_by == Bound::Static, "{}: the noise is below the 4-bit floor, so the floor binds", r.name);
                assert!(!alone.pass, "{}: every build is more than 1x vLLM's own spread from the golden", r.name);
                seen.0 += 1;
            } else {
                // the change against the default build (a self-golden compares all rows; the scored rows are given for information)
                let nz = noise.kl21_mean.unwrap();
                let nzs = noise.kl21_mean_scored.unwrap();
                let v = judge_kl("prefill", "21-way KL", r.all, None, term_rows(nz, r.all, "all rows"), 1.0, "default").unwrap();
                let vs = judge_kl("prefill", "21-way KL", r.excl, None, term_rows(nzs, r.excl, "scored rows"), 1.0, "default").unwrap();
                // the worst prompt (all rows, as a self-golden reports it) against 3.0 x the all-rows noise (user decision 2026-10-21 13:23)
                let worst = r.prompts.iter().cloned().fold(("".to_string(), f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
                let wp = judge_worst_prompt((&worst.0, worst.1), nz, DEFAULT_PROMPT_FACTOR, "default").unwrap();
                println!(
                    "A/B+ {which:<5} {:<34} worst prompt {} {:.3e} / ({} x {nz:.3e} = {:.3e}) = {:.2}x of the limit {}   => change vs base overall {}",
                    r.name.trim_end_matches(".safetensors"),
                    worst.0,
                    worst.1,
                    DEFAULT_PROMPT_FACTOR,
                    wp.limit,
                    worst.1 / wp.limit,
                    if wp.pass { "PASS" } else { "FAIL" },
                    if v.pass && wp.pass { "PASS" } else { "FAIL" }
                );
                // the long golden's worst prompt is chat_reuse_4k, the one prompt vLLM itself does not reproduce (0.118)
                assert_eq!(wp.pass, false, "{}: {} ", r.name, worst.0);
                println!(
                    "A/B  {which:<5} {:<34} all rows {:.3e} / {nz:.3e} = {:.3}x {}   scored rows {:.3e} / {nzs:.3e} = {:.3}x {}   old self-golden budget (mean KL 1e-4): {:.0}x over",
                    r.name.trim_end_matches(".safetensors"),
                    r.all,
                    v.factor_needed.unwrap(),
                    if v.pass { "PASS" } else { "FAIL" },
                    r.excl,
                    vs.factor_needed.unwrap(),
                    if vs.pass { "PASS" } else { "FAIL" },
                    r.all / 1e-4
                );
                // long: inside vLLM's own spread; short: 5-11 % outside it. The same on both row sets.
                assert_eq!((v.pass, vs.pass), (is_long, is_long), "{}", r.name);
                seen.1 += 1;
            }
        }
        assert_eq!(seen, (7, 5));
    }

    /// Qwen3.8-27B W4A4 (docs/results/2026-10-21-m1-qwen38-drill.md section 5): the HF reference against itself at f32
    /// is 0.197 away; the engine is 0.180 (prefill) and 0.274 (decode) away from the bf16 reference, against a floor of 0.03.
    #[test]
    fn rescore_m1_w4a4_from_the_documented_numbers() {
        let noise = sidecar("qwen3.8-27b-mxfp4-hfquark-top20.safetensors");
        assert_eq!((noise.method, noise.kl21_mean, noise.kl21_mean_scored), (Method::PrecisionSpread, Some(0.197), None));
        for (mode, kl, pass) in [("prefill", 0.180, true), ("prefill (golden_kl.py)", 0.176, true), ("decode", 0.274, false)] {
            let v = judge_kl(mode, "21-way KL", kl, Some(0.03), term_rows(0.197, kl, "all rows"), 1.0, "default").unwrap();
            println!(
                "M1   {mode:<24} KL {kl:.3} vs floor 0.03 and noise 0.197: {} (bound by {}), KL/noise {:.3}",
                if v.pass { "PASS" } else { "FAIL" },
                v.bound_by.name(),
                v.factor_needed.unwrap()
            );
            assert_eq!((v.pass, v.bound_by), (pass, Bound::RefNoise), "{mode}");
        }
    }

    #[test]
    fn worst_prompt_check_and_prompt_factor() {
        // 3.0 x vLLM's long-golden run-to-run KL 2.218e-2 = 6.65e-2; chat_reuse_4k alone is at 0.118 in vLLM's own re-run
        let v = judge_worst_prompt(("chat_reuse_4k", 9.714e-2), 2.218e-2, 3.0, "default").unwrap();
        assert!(!v.pass && (v.limit - 6.654e-2).abs() < 1e-9 && (v.factor_needed - 9.714e-2 / 2.218e-2).abs() < 1e-12);
        assert!(judge_worst_prompt(("p", 6.6e-2), 2.218e-2, 3.0, "default").unwrap().pass);
        assert!(judge_worst_prompt(("chat_reuse_4k", 9.714e-2), 2.218e-2, 4.5, "--ref-noise-prompt-factor").unwrap().pass);
        assert!(!judge_worst_prompt(("p", f64::NAN), 1.0, 3.0, "default").unwrap().pass);
        assert!(judge_worst_prompt(("p", 1.0), 1.0, 0.0, "x").is_err());
        // precedence flag > record > recipe > 3.0, same shape as the main factor
        let recipe = Some((2.0, "recipe x".to_string()));
        assert_eq!(resolve_prompt_factor(Some(5.0), Some(4.0), recipe.clone()).unwrap(), (5.0, "--ref-noise-prompt-factor".into()));
        assert_eq!(resolve_prompt_factor(None, Some(4.0), recipe.clone()).unwrap().0, 4.0);
        assert_eq!(resolve_prompt_factor(None, None, recipe).unwrap(), (2.0, "recipe x".into()));
        assert_eq!(resolve_prompt_factor(None, None, None).unwrap(), (3.0, "default".into()));
        assert!(resolve_prompt_factor(Some(-1.0), None, None).is_err());
        let r = RefNoise::parse(r#"{"method":"run_to_run","kl21_mean":0.1,"date":"d","how":"h","prompt_factor":4.0}"#).unwrap();
        assert_eq!(r.prompt_factor, Some(4.0));
        assert!(RefNoise::parse(r#"{"method":"run_to_run","kl21_mean":0.1,"date":"d","how":"h","prompt_factor":0}"#).is_err());
    }

    #[test]
    fn dates() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(1_792_540_800), "2026-10-21");
        assert_eq!(utc_date(951_782_400), "2000-02-29");
    }
}
