//! Measured noise envelope for the per-prompt part of the baseline gate (`--write-noise-envelope`, `--noise-envelope`).
//!
//! The fixed per-prompt margins (`baseline::Margins`: KL x1.75, top-1 -1 pt or one position, |tail diff| +0.002) were
//! chosen on a dense model, where an equally correct build moves a prompt by almost nothing. On the MoE / hybrid
//! 35B they sit inside the noise: the same build with another attention kernel (flash vs SIMT), or another prefill
//! chunk, moves one prompt's KL by up to 0.13-0.17 and the others by 0.005-0.012 (docs/results/2026-10-13-phase1b.md,
//! sections 8-10), and fails its own baseline gate. A limit that is smaller than the spread of correct builds
//! measures nothing but that spread.
//!
//! The envelope is that spread, measured per prompt. Run the *same build* a second time with a declared, equally
//! correct perturbation (another prefill chunk, SIMT attention instead of flash / split-KV, ...) against the baseline
//! of the unperturbed run, and keep for every prompt and mode the absolute difference of the numbers the per-prompt
//! gate looks at (top-1, mean 21-way KL, mean |tail difference|; the largest single-position KL is recorded for the
//! reader only): `intarsia accuracy ... --noise-ref BASELINE.json --noise-perturb attn-simt --write-noise-envelope E.json`.
//! More perturbations go into the same file (their per-prompt maximum is used). The gate then allows, per prompt,
//!
//! ```text
//! limit = baseline value -/+ max(fixed margin, k x envelope delta of that prompt)      (k = 2 by default)
//! ```
//!
//! and prints for every prompt which of the two decided (`fixed` / `envelope`). The aggregate (all-prompt) checks do
//! not use the envelope and stay as hard as they were. A perturbation that does not touch a prompt (a prefill chunk
//! change leaves every prompt shorter than the chunk bit-identical) gives delta 0 for it: that is "not measured", not
//! "quiet", the fixed margin applies, and the writer says which prompts were not touched.
//!
//! The envelope is only valid for the baseline it was measured against: same golden, same run configuration, same
//! code (the build that recorded the baseline), otherwise the gate refuses it.

use super::baseline::{self, Baseline, GoldenId, ModeStats, PromptDelta, PromptStats, RunConfig};
use super::provenance::{CodeStamp, ExecInfo};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const FORMAT: &str = "intarsia-accuracy-noise-envelope-v1";
/// Default `k`: see "Choosing k" in docs/accuracy.md.
pub const DEFAULT_K: f64 = 2.0;
/// The attention kernels `attn-simt` switches off (flash and split-KV; the SIMT kernels remain). Same names on cuda
/// and hip.
pub const SIMT_ATTENTION_DISABLE: &str = "attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash";

/// A delta below this is rounding of the stored log-probs, not a perturbation that reached the prompt.
const NEGLIGIBLE: f64 = 1e-9;

/// The baseline run (unperturbed) and the perturbed run of one prompt.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PromptNoise {
    pub base: PromptStats,
    pub perturbed: PromptStats,
}

impl PromptNoise {
    pub fn delta(&self) -> PromptDelta {
        PromptDelta {
            top1_pts: (self.base.top1_pct - self.perturbed.top1_pct).abs(),
            kl21: (self.base.mean_kl21 - self.perturbed.mean_kl21).abs(),
            tail_abs: (self.base.tail_abs - self.perturbed.tail_abs).abs(),
        }
    }
    /// |difference| of the largest single-position 21-way KL (not a gated quantity).
    pub fn max_kl21_delta(&self) -> f64 {
        (self.base.max_kl21 - self.perturbed.max_kl21).abs()
    }
    pub fn untouched(&self) -> bool {
        let d = self.delta();
        d.top1_pts == 0.0 && d.kl21 < NEGLIGIBLE && d.tail_abs < NEGLIGIBLE
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ModeNoise {
    /// `prefill` / `decode`.
    pub mode: String,
    /// The aggregates of the two runs (information: the aggregate gate does not use the envelope).
    pub base_overall: ModeStats,
    pub perturbed_overall: ModeStats,
    pub prompts: Vec<PromptNoise>,
}

/// The baseline file the perturbed run was compared with.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Reference {
    pub file: String,
    /// FNV-1a 64 of the baseline file's bytes.
    pub fnv1a64: String,
    pub created_unix: u64,
}

/// One perturbed run.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Perturbation {
    /// What was changed, in words (`prefill chunk 256 -> 64`, ...).
    pub declared: String,
    pub created_unix: u64,
    pub tool_version: String,
    pub code: CodeStamp,
    pub exec: ExecInfo,
    pub run: RunConfig,
    pub reference: Reference,
    pub modes: Vec<ModeNoise>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Envelope {
    pub format: String,
    pub golden: GoldenId,
    /// The configuration of the unperturbed (baseline) run.
    pub base_run: RunConfig,
    /// The code that recorded the baseline and ran every perturbation: one build.
    pub base_code: CodeStamp,
    pub perturbations: Vec<Perturbation>,
}

impl Envelope {
    /// Per-prompt delta in `mode`: the maximum over the perturbations that have it. None when no perturbation does.
    pub fn delta(&self, mode: &str, prompt: &str) -> Option<PromptDelta> {
        let mut out: Option<PromptDelta> = None;
        for p in &self.perturbations {
            let Some(m) = p.modes.iter().find(|m| m.mode == mode) else { continue };
            let Some(n) = m.prompts.iter().find(|n| n.base.name == prompt) else { continue };
            let d = n.delta();
            out = Some(match out {
                None => d,
                Some(o) => PromptDelta { top1_pts: o.top1_pts.max(d.top1_pts), kl21: o.kl21.max(d.kl21), tail_abs: o.tail_abs.max(d.tail_abs) },
            });
        }
        out
    }

    pub fn has_mode(&self, mode: &str) -> bool {
        self.perturbations.iter().any(|p| p.modes.iter().any(|m| m.mode == mode))
    }

    /// Names of the prompts of `mode` that no perturbation changed (no change beyond rounding in any metric of any perturbation).
    pub fn untouched(&self, mode: &str) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for p in &self.perturbations {
            for m in p.modes.iter().filter(|m| m.mode == mode) {
                for n in &m.prompts {
                    if !names.contains(&n.base.name) {
                        names.push(n.base.name.clone());
                    }
                }
            }
        }
        names.retain(|name| self.delta(mode, name).is_some_and(|d| d.top1_pts == 0.0 && d.kl21 < NEGLIGIBLE && d.tail_abs < NEGLIGIBLE));
        names
    }

    pub fn describe(&self) -> String {
        self.perturbations.iter().map(|p| p.declared.as_str()).collect::<Vec<_>>().join("; ")
    }
}

pub fn read(path: &Path) -> Result<Envelope> {
    let t = std::fs::read_to_string(path).with_context(|| format!("reading noise envelope {}", path.display()))?;
    let e: Envelope = serde_json::from_str(&t).map_err(|e| anyhow!("{}: not a noise envelope: {e}", path.display()))?;
    if e.format != FORMAT {
        bail!("{}: envelope format '{}' is not {FORMAT}", path.display(), e.format);
    }
    Ok(e)
}

pub fn write(path: &Path, e: &Envelope) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(e)? + "\n").with_context(|| format!("writing {}", path.display()))
}

/// The same build: head, code tree, dirty flag and overlay all equal.
pub fn same_build(a: &CodeStamp, b: &CodeStamp) -> bool {
    a.head == b.head && a.code_tree == b.code_tree && a.dirty == b.dirty && a.overlay == b.overlay
}

/// Put one perturbed run into the envelope file `path` (a new file, or an existing one measured for the same
/// golden, baseline configuration and build: a run with the same declared text replaces the earlier one).
pub fn merge_into(existing: Option<Envelope>, header: Envelope, path: &Path) -> Result<Envelope> {
    let Some(mut e) = existing else { return Ok(header) };
    if e.golden.fnv1a64 != header.golden.fnv1a64 || e.golden.size != header.golden.size {
        bail!("{}: the envelope was measured against another golden ({}); use another file or remove it", path.display(), e.golden.file);
    }
    if e.base_run != header.base_run {
        bail!("{}: the envelope was measured for another configuration of the baseline run; use another file or remove it", path.display());
    }
    if !same_build(&e.base_code, &header.base_code) {
        bail!("{}: the envelope was measured on another build ({}); use another file or remove it", path.display(), e.base_code.describe());
    }
    for p in header.perturbations {
        match e.perturbations.iter_mut().find(|q| q.declared == p.declared) {
            Some(q) => *q = p,
            None => e.perturbations.push(p),
        }
    }
    Ok(e)
}

/// Is `e` valid for `base`: same golden, same baseline run configuration, same build.
pub fn check_for_baseline(e: &Envelope, base: &Baseline, allow_unstamped: bool) -> Result<()> {
    if e.golden.fnv1a64 != base.golden.fnv1a64 || e.golden.size != base.golden.size {
        bail!("the noise envelope was measured against another golden file ({}) than the baseline's ({})", e.golden.file, base.golden.file);
    }
    if e.base_run != base.run {
        bail!("the noise envelope belongs to a baseline run with another configuration than this baseline's: record the envelope again from this baseline (--noise-ref)");
    }
    if !(e.base_code.is_stamped() && base.code.is_stamped()) {
        if !allow_unstamped {
            bail!("the noise envelope or the baseline carries no code stamp, so they cannot be shown to come from one build (--allow-unstamped for development)");
        }
    } else if !same_build(&e.base_code, &base.code) {
        bail!(
            "the noise envelope was measured on {} but the baseline was recorded by {}: the envelope must come from the build that recorded the baseline",
            e.base_code.describe(),
            base.code.describe()
        );
    }
    Ok(())
}

/// What `--noise-perturb` asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct Declared {
    pub text: String,
    /// New `--prefill-chunk`.
    pub chunk: Option<usize>,
    /// Environment to set before the backend opens.
    pub env: Vec<(&'static str, String)>,
    /// The caller set flags / environment itself (`external:TEXT`).
    pub external: bool,
}

/// Parse `--noise-perturb`: `chunk` (baseline chunk / 4, at least 16), `chunk=N`, `attn-simt`, `external:TEXT`.
pub fn declare(spec: &str, backend: &str, base_chunk: usize) -> Result<Declared> {
    let spec = spec.trim();
    if spec == "chunk" || spec.starts_with("chunk=") {
        let n = match spec.strip_prefix("chunk=") {
            Some(v) => v.parse::<usize>().map_err(|_| anyhow!("--noise-perturb {spec}: not a chunk size"))?,
            None => (base_chunk / 4).max(16),
        };
        if n == 0 || n == base_chunk {
            bail!("--noise-perturb {spec}: the baseline run used chunk {base_chunk}; the perturbed chunk must differ");
        }
        return Ok(Declared { text: format!("prefill chunk {base_chunk} -> {n}"), chunk: Some(n), env: vec![], external: false });
    }
    if spec == "attn-simt" {
        let var = match backend {
            "cuda" => "INTARSIA_CUDA_DISABLE",
            "hip" => "INTARSIA_HIP_DISABLE",
            b => bail!("--noise-perturb attn-simt: the {b} backend has no flash / SIMT attention pair (cuda, hip); use chunk or external:TEXT"),
        };
        if std::env::var(var).is_ok_and(|v| !v.trim().is_empty()) {
            bail!("--noise-perturb attn-simt sets {var}, which is already set; unset it");
        }
        return Ok(Declared {
            text: format!("attention: SIMT kernels instead of flash / split-KV ({var}={SIMT_ATTENTION_DISABLE})"),
            chunk: None,
            env: vec![(if backend == "cuda" { "INTARSIA_CUDA_DISABLE" } else { "INTARSIA_HIP_DISABLE" }, SIMT_ATTENTION_DISABLE.to_string())],
            external: false,
        });
    }
    if let Some(t) = spec.strip_prefix("external:") {
        if t.trim().is_empty() {
            bail!("--noise-perturb external: needs a description of what was changed");
        }
        return Ok(Declared { text: t.trim().to_string(), chunk: None, env: vec![], external: true });
    }
    bail!("--noise-perturb '{spec}': expected chunk, chunk=N, attn-simt or external:TEXT")
}

/// May the perturbed run's kernel disable list differ from the baseline's?
fn may_change_kernels(d: &Declared) -> bool {
    d.external || !d.env.is_empty()
}

/// Check the perturbed run against the baseline of the unperturbed run: same golden, same configuration except the
/// declared perturbation, a baseline that was not itself perturbed, the same build, and a perturbation that changed
/// something. Returns warnings.
pub fn check_run(base: &Baseline, golden: &GoldenId, run: &RunConfig, exec: &ExecInfo, code: &CodeStamp, d: &Declared, allow_unstamped: bool) -> Result<Vec<String>> {
    if !base.exec.disable_list.is_empty() || !base.exec.fault.is_empty() {
        bail!("the reference baseline was itself recorded with kernels disabled or a fault injected ({} {}): the envelope needs a clean unperturbed run", base.exec.disable_list, base.exec.fault);
    }
    // the baseline module's strict comparison, with the declared perturbation substituted
    let mut b = base.clone();
    if d.chunk.is_some() {
        b.run.prefill_chunk = run.prefill_chunk;
    }
    if may_change_kernels(d) {
        b.exec.disable_list = exec.disable_list.clone();
    }
    let warnings = baseline::compatible(&b, golden, run, exec, false)?;
    if !(base.code.is_stamped() && code.is_stamped()) {
        if !allow_unstamped {
            bail!("the reference baseline or this build carries no code stamp, so they cannot be shown to be one build (build through scripts/phase1b-acceptance.sh, or --allow-unstamped for development)");
        }
    } else if !same_build(&base.code, code) {
        bail!(
            "the reference baseline was recorded by {} but this build is {}: a noise envelope is the spread of ONE build; record the baseline again with this binary",
            base.code.describe(),
            code.describe()
        );
    }
    let differs = run.prefill_chunk != base.run.prefill_chunk || !base.exec.diff(exec).iter().all(|l| l.starts_with("graphs:")) || base.exec.fault != exec.fault;
    if !differs {
        bail!("the perturbation '{}' changed nothing in this run (same chunk, same resolved kernels, no fault): the envelope would be zero by construction", d.text);
    }
    Ok(warnings)
}

/// Pair the two runs' numbers, mode by mode. A mode or prompt missing on either side is an error.
pub fn pair(base: &Baseline, results: &[(String, ModeStats, Vec<PromptStats>)]) -> Result<Vec<ModeNoise>> {
    let mut out = Vec::new();
    for (key, stats, prompts) in results {
        let bm = base
            .modes
            .iter()
            .find(|b| &b.overall.mode == key)
            .ok_or_else(|| anyhow!("the reference baseline has no '{key}' mode (it has: {})", base.modes.iter().map(|b| b.overall.mode.as_str()).collect::<Vec<_>>().join(", ")))?;
        if bm.prompts.len() != prompts.len() {
            bail!("the reference baseline has {} prompts in '{key}', this run {}", bm.prompts.len(), prompts.len());
        }
        let mut pn = Vec::new();
        for p in prompts {
            let b = bm.prompts.iter().find(|b| b.name == p.name).ok_or_else(|| anyhow!("prompt '{}' is not in the reference baseline", p.name))?;
            if b.positions != p.positions {
                bail!("prompt '{}': {} positions in the reference baseline, {} in this run", p.name, b.positions, p.positions);
            }
            let n = PromptNoise { base: b.clone(), perturbed: p.clone() };
            let d = n.delta();
            if ![d.top1_pts, d.kl21, d.tail_abs, n.max_kl21_delta()].iter().all(|x| x.is_finite()) {
                bail!("prompt '{}' ({key}): a non-finite number in the envelope (the baseline or the perturbed run has NaN / infinity there)", p.name);
            }
            pn.push(n);
        }
        out.push(ModeNoise { mode: key.clone(), base_overall: bm.overall.clone(), perturbed_overall: stats.clone(), prompts: pn });
    }
    Ok(out)
}

/// Human-readable table of one perturbation's per-prompt deltas, and the prompts it did not touch.
pub fn report(p: &Perturbation) -> Vec<String> {
    let mut l = Vec::new();
    for m in &p.modes {
        l.push(format!("noise envelope, {} ({}):", m.mode, p.declared));
        l.push(format!("  {:<18} {:>5} {:>12} {:>12} {:>12} {:>12}", "prompt", "pos", "d top-1 pt", "d KL(21)", "d |tail|", "d max KL(21)"));
        for n in &m.prompts {
            let d = n.delta();
            l.push(format!(
                "  {:<18} {:>5} {:>12.4} {:>12.4e} {:>12.4e} {:>12.4e}{}",
                n.base.name,
                n.base.positions,
                d.top1_pts,
                d.kl21,
                d.tail_abs,
                n.max_kl21_delta(),
                if n.untouched() { "   <- untouched by this perturbation" } else { "" }
            ));
        }
        let (b, o) = (&m.base_overall, &m.perturbed_overall);
        l.push(format!(
            "  aggregate (information; the aggregate gate does not use the envelope): top-1 {:.3} -> {:.3} %, mean KL(21) {:.4e} -> {:.4e}, max KL(21) {:.4e} -> {:.4e}",
            b.top1_pct, o.top1_pct, b.mean_kl21, o.mean_kl21, b.max_kl21, o.max_kl21
        ));
        let untouched: Vec<&str> = m.prompts.iter().filter(|n| n.untouched()).map(|n| n.base.name.as_str()).collect();
        if !untouched.is_empty() {
            l.push(format!(
                "  WARNING: this perturbation did not touch {} of {} prompts ({}): their delta is 0 because nothing moved, not because they are quiet; the fixed margin applies to them unless another perturbation measures them",
                untouched.len(),
                m.prompts.len(),
                untouched.join(", ")
            ));
        }
    }
    l
}

/// Is the model dense (no experts, no linear-attention / state-space layers), read from its `config.json`? None when
/// the file cannot be read. The per-prompt fixed margins were set on a dense model.
pub fn is_dense_model(dir: &Path) -> Option<bool> {
    let t = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&t).ok()?;
    Some(!has_non_dense_marker(&v))
}

fn has_non_dense_marker(v: &serde_json::Value) -> bool {
    use serde_json::Value;
    match v {
        Value::Object(m) => m.iter().any(|(k, x)| {
            let k = k.to_lowercase();
            let nonzero = match x {
                Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
                Value::Bool(b) => *b,
                Value::Null => false,
                Value::String(s) => !s.is_empty(),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            };
            ((k.contains("expert") || k.contains("moe")) && nonzero)
                || k.starts_with("linear_")
                || k.contains("mamba")
                || k.contains("ssm")
                || (k == "layer_types" && x.as_array().is_some_and(|a| a.iter().any(|t| t.as_str().is_some_and(|s| s != "full_attention" && s != "attention"))))
                || has_non_dense_marker(x)
        }),
        Value::Array(a) => a.iter().any(has_non_dense_marker),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd::accuracy::baseline::{Margins, ModeBaseline};

    fn ps(name: &str, n: usize, top1: f64, kl21: f64, tail: f64) -> PromptStats {
        PromptStats { name: name.into(), positions: n, top1_pct: top1, mean_kl: kl21, mean_kl21: kl21, top5_pct: 90.0, tail_excess: 0.0, max_kl21: 0.1, tail_abs: tail }
    }
    fn stats(mode: &str) -> ModeStats {
        ModeStats {
            mode: mode.into(), positions: 100, top1_pct: 95.0, mean_kl: 1e-2, mean_kl21: 1e-2, top5_pct: 90.0, tail_ref: 0.05, tail_ours: 0.05,
            max_kl: 0.5, max_kl21: 0.5, tail_abs: 0.004, nonfinite: 0, greedy_match_pct: 50.0, floor_pass: true,
        }
    }
    fn code(tree: &str) -> CodeStamp {
        CodeStamp::parse(&format!("head=4963b43 code-tree={tree} dirty=yes overlay=crates/intarsia-cli/src/cmd/accuracy.rs"))
    }
    fn golden() -> GoldenId {
        GoldenId {
            file: "g.safetensors".into(), fnv1a64: "00ff".into(), size: 10, format: "intarsia-golden-topk-v1".into(), prompts: 2, positions: 200,
            topk: Some(20), engine: "vllm".into(), engine_version: "0.1".into(), model: "m".into(), model_revision: "r".into(), kv_dtype: "fp8".into(),
        }
    }
    fn run(chunk: usize) -> RunConfig {
        RunConfig {
            model: "qwen".into(), backend: "cuda".into(), device: "GB10".into(), kv_dtype: "bf16".into(), act_dtype: "bf16".into(),
            fp8_act: "static".into(), prefill_chunk: chunk, decode_positions: 16, gate_class: "4bit".into(),
        }
    }
    fn base_with(prompts: Vec<PromptStats>) -> Baseline {
        Baseline {
            format: baseline::FORMAT.into(), created_unix: 1, tool_version: "x".into(), code: code("aaaaaaaaaaaa"), exec: ExecInfo::default(),
            golden: golden(), run: run(256), modes: vec![ModeBaseline { overall: stats("prefill"), prompts }],
        }
    }
    fn perturbed(prompts: Vec<PromptStats>) -> Vec<(String, ModeStats, Vec<PromptStats>)> {
        vec![("prefill".into(), stats("prefill"), prompts)]
    }
    fn envelope_of(base: &Baseline, p: Vec<PromptStats>, declared: &str) -> Envelope {
        Envelope {
            format: FORMAT.into(),
            golden: base.golden.clone(),
            base_run: base.run.clone(),
            base_code: base.code.clone(),
            perturbations: vec![Perturbation {
                declared: declared.into(), created_unix: 2, tool_version: "x".into(), code: base.code.clone(), exec: ExecInfo::default(), run: run(64),
                reference: Reference { file: "b.json".into(), fnv1a64: "11".into(), created_unix: 1 }, modes: pair(base, &perturbed(p)).unwrap(),
            }],
        }
    }

    #[test]
    fn delta_is_the_absolute_difference_of_the_prompt_numbers() {
        let base = base_with(vec![ps("a", 100, 95.0, 1e-2, 0.004), ps("b", 100, 98.0, 5e-3, 0.002)]);
        let e = envelope_of(&base, vec![ps("a", 100, 93.0, 4e-2, 0.007), ps("b", 100, 98.0, 5e-3, 0.002)], "chunk");
        let d = e.delta("prefill", "a").unwrap();
        assert!((d.top1_pts - 2.0).abs() < 1e-12 && (d.kl21 - 3e-2).abs() < 1e-12 && (d.tail_abs - 3e-3).abs() < 1e-12);
        assert_eq!(e.untouched("prefill"), ["b"], "b did not move at all");
        assert!(e.delta("decode", "a").is_none() && e.delta("prefill", "zzz").is_none());
        // a better perturbed run is the same distance
        let e2 = envelope_of(&base, vec![ps("a", 100, 97.0, 1e-3, 0.004), ps("b", 100, 98.0, 5e-3, 0.002)], "x");
        assert!((e2.delta("prefill", "a").unwrap().top1_pts - 2.0).abs() < 1e-12);
    }

    #[test]
    fn several_perturbations_give_the_per_prompt_maximum() {
        let base = base_with(vec![ps("a", 100, 95.0, 1e-2, 0.004)]);
        let mut e = envelope_of(&base, vec![ps("a", 100, 94.0, 2e-2, 0.004)], "chunk");
        let e2 = envelope_of(&base, vec![ps("a", 100, 90.0, 1.5e-2, 0.004)], "simt");
        e = merge_into(Some(e), e2.clone(), Path::new("e.json")).unwrap();
        assert_eq!(e.perturbations.len(), 2);
        let d = e.delta("prefill", "a").unwrap();
        assert!((d.top1_pts - 5.0).abs() < 1e-12, "top-1 from simt");
        assert!((d.kl21 - 1e-2).abs() < 1e-12, "KL from chunk");
        // the same declared text replaces the earlier run
        e = merge_into(Some(e), e2, Path::new("e.json")).unwrap();
        assert_eq!(e.perturbations.len(), 2);
        // an envelope of another build or golden cannot be extended
        let mut other = envelope_of(&base, vec![ps("a", 100, 94.0, 2e-2, 0.004)], "third");
        other.base_code = code("bbbbbbbbbbbb");
        assert!(merge_into(Some(e.clone()), other, Path::new("e.json")).unwrap_err().to_string().contains("another build"));
        let mut other = envelope_of(&base, vec![ps("a", 100, 94.0, 2e-2, 0.004)], "third");
        other.golden.fnv1a64 = "beef".into();
        assert!(merge_into(Some(e), other, Path::new("e.json")).unwrap_err().to_string().contains("another golden"));
    }

    #[test]
    fn the_envelope_widens_the_limit_only_where_the_prompt_is_noisy() {
        let m = Margins::default();
        let base = vec![ps("quiet", 100, 95.0, 1e-2, 0.004), ps("noisy", 100, 95.0, 1e-2, 0.004)];
        let e = envelope_of(&base_with(base.clone()), vec![ps("quiet", 100, 95.0, 1e-2, 0.004), ps("noisy", 100, 90.0, 1e-1, 0.012)], "p");
        // both prompts got 0.05 worse in KL and 3 pt worse in top-1
        let cur = vec![ps("quiet", 100, 92.0, 6e-2, 0.004), ps("noisy", 100, 92.0, 6e-2, 0.004)];
        let f = baseline::judge_prompts(&base, &cur, &m).unwrap();
        assert_eq!(f.len(), 2, "fixed margins fail both");
        let d = |n: &str| e.delta("prefill", n);
        let cal = baseline::Calibration { k: 2.0, delta: &d };
        let v = baseline::judge_prompts_calibrated(&base, &cur, &m, Some(&cal)).unwrap();
        assert!(!v[0].ok(), "the quiet prompt keeps the fixed margin and fails");
        assert!(v[0].checks.iter().all(|c| c.source == baseline::Source::Fixed));
        assert!(v[1].ok(), "the noisy prompt's limit follows its measured spread: {:?}", v[1]);
        assert!(v[1].checks.iter().any(|c| c.source == baseline::Source::Envelope));
        // limit = baseline + k x delta
        let kl = &v[1].checks[0];
        assert!((kl.check.limit - (1e-2 + 2.0 * 9e-2)).abs() < 1e-12 && (kl.delta.unwrap() - 9e-2).abs() < 1e-12);
        // a real regression still fails: far beyond k x delta
        let bad = vec![ps("quiet", 100, 95.0, 1e-2, 0.004), ps("noisy", 100, 80.0, 0.5, 0.004)];
        let v = baseline::judge_prompts_calibrated(&base, &bad, &m, Some(&cal)).unwrap();
        assert!(v[0].ok() && !v[1].ok());
        // k*: how far k would have to go
        let kn = v[1].checks[0].k_needed().unwrap();
        assert!((kn - 0.49 / 9e-2).abs() < 1e-9, "{kn}");
        // a prompt missing from the envelope is an error, not a pass
        let none = |_: &str| None;
        let cal = baseline::Calibration { k: 2.0, delta: &none };
        assert!(baseline::judge_prompts_calibrated(&base, &cur, &m, Some(&cal)).is_err());
    }

    #[test]
    fn without_an_envelope_the_limits_are_the_old_fixed_ones() {
        let m = Margins::default();
        let base = vec![ps("a", 87, 100.0, 6.5e-3, 0.003)];
        let v = baseline::judge_prompts_calibrated(&base, &base, &m, None).unwrap();
        let c = &v[0].checks;
        assert_eq!(c[0].check.limit, 6.5e-3 * m.prompt_kl_ratio + m.prompt_kl_eps);
        assert_eq!(c[1].check.limit, 100.0 - m.prompt_top1_pts.max(100.0 / 87.0));
        assert_eq!(c[2].check.limit, 0.003 + m.tail_abs);
        assert!(c.iter().all(|x| x.source == baseline::Source::Fixed && x.delta.is_none()));
    }

    #[test]
    fn the_envelope_must_belong_to_the_baseline() {
        let base = base_with(vec![ps("a", 100, 95.0, 1e-2, 0.004)]);
        let e = envelope_of(&base, vec![ps("a", 100, 94.0, 2e-2, 0.004)], "chunk");
        check_for_baseline(&e, &base, false).unwrap();
        let mut b2 = base.clone();
        b2.code = code("bbbbbbbbbbbb"); // recorded by another build
        assert!(check_for_baseline(&e, &b2, false).unwrap_err().to_string().contains("must come from the build that recorded"));
        let mut b3 = base.clone();
        b3.golden.fnv1a64 = "beef".into();
        assert!(check_for_baseline(&e, &b3, false).unwrap_err().to_string().contains("another golden"));
        let mut b4 = base.clone();
        b4.run.prefill_chunk = 512;
        assert!(check_for_baseline(&e, &b4, false).is_err());
        let mut b5 = base.clone();
        b5.code = CodeStamp::default();
        assert!(check_for_baseline(&e, &b5, false).is_err());
        check_for_baseline(&e, &b5, true).unwrap();
    }

    #[test]
    fn the_perturbed_run_is_checked_against_its_reference() {
        let base = base_with(vec![ps("a", 100, 95.0, 1e-2, 0.004)]);
        let ex = ExecInfo::default();
        let chunk = declare("chunk", "cuda", 256).unwrap();
        assert_eq!(chunk.chunk, Some(64));
        check_run(&base, &golden(), &run(64), &ex, &base.code, &chunk, false).unwrap();
        // the chunk did not change: the perturbation did nothing
        assert!(check_run(&base, &golden(), &run(256), &ex, &base.code, &chunk, false).unwrap_err().to_string().contains("changed nothing"));
        // another build than the baseline's
        assert!(check_run(&base, &golden(), &run(64), &ex, &code("bbbbbbbbbbbb"), &chunk, false).unwrap_err().to_string().contains("ONE build"));
        // a different configuration that is not the declared perturbation is refused
        let mut r = run(64);
        r.kv_dtype = "f16".into();
        assert!(check_run(&base, &golden(), &r, &ex, &base.code, &chunk, false).is_err());
        // a disable list needs a declared kernel perturbation
        let disabled = ExecInfo { disable_list: "INTARSIA_CUDA_DISABLE=a".into(), ..Default::default() };
        assert!(check_run(&base, &golden(), &run(256), &disabled, &base.code, &chunk, false).is_err());
        let simt = declare("attn-simt", "cuda", 256).unwrap();
        let mut ex2 = disabled.clone();
        ex2.kernels.insert("attn_prefill".into(), "attn_prefill_bf16_simt".into());
        check_run(&base, &golden(), &run(256), &ex2, &base.code, &simt, false).unwrap();
        // an injected fault declared externally
        let ext = declare("external:INTARSIA_CPU_FAULT=matmul:4.7e-3", "cpu", 256).unwrap();
        let fault = ExecInfo { fault: "INTARSIA_CPU_FAULT=matmul:4.7e-3".into(), ..Default::default() };
        check_run(&base, &golden(), &run(256), &fault, &base.code, &ext, false).unwrap();
        assert!(check_run(&base, &golden(), &run(256), &ex, &base.code, &ext, false).is_err(), "external, but nothing differs");
        // a reference recorded with a fault is not an unperturbed run
        let mut bf = base.clone();
        bf.exec.fault = "INTARSIA_CPU_FAULT=matmul:1".into();
        assert!(check_run(&bf, &golden(), &run(64), &ex, &base.code, &chunk, false).is_err());
    }

    #[test]
    fn perturbation_specs() {
        assert_eq!(declare("chunk", "cpu", 256).unwrap().chunk, Some(64));
        assert_eq!(declare("chunk", "cpu", 32).unwrap().chunk, Some(16));
        assert_eq!(declare("chunk=1024", "cpu", 256).unwrap().chunk, Some(1024));
        assert!(declare("chunk=256", "cpu", 256).is_err());
        assert!(declare("chunk=x", "cpu", 256).is_err());
        assert!(declare("attn-simt", "cpu", 256).is_err());
        let d = declare("attn-simt", "hip", 256).unwrap();
        assert_eq!(d.env[0].0, "INTARSIA_HIP_DISABLE");
        assert!(declare("external:", "cpu", 256).is_err());
        assert!(declare("bogus", "cpu", 256).is_err());
    }

    #[test]
    fn missing_or_mismatched_prompts_are_errors() {
        let base = base_with(vec![ps("a", 100, 95.0, 1e-2, 0.004), ps("b", 100, 95.0, 1e-2, 0.004)]);
        assert!(pair(&base, &perturbed(vec![ps("a", 100, 95.0, 1e-2, 0.004)])).is_err());
        assert!(pair(&base, &perturbed(vec![ps("a", 100, 95.0, 1e-2, 0.004), ps("c", 100, 95.0, 1e-2, 0.004)])).is_err());
        assert!(pair(&base, &perturbed(vec![ps("a", 100, 95.0, 1e-2, 0.004), ps("b", 99, 95.0, 1e-2, 0.004)])).is_err());
        assert!(pair(&base, &perturbed(vec![ps("a", 100, 95.0, f64::NAN, 0.004), ps("b", 100, 95.0, 1e-2, 0.004)])).is_err());
        assert!(pair(&base, &[("decode".into(), stats("decode"), vec![])]).is_err());
    }

    #[test]
    fn envelope_json_round_trips() {
        let base = base_with(vec![ps("a", 100, 95.0, 1e-2, 0.004)]);
        let e = envelope_of(&base, vec![ps("a", 100, 94.0, 2e-2, 0.004)], "chunk");
        let p = std::env::temp_dir().join(format!("intarsia-envelope-{}.json", std::process::id()));
        write(&p, &e).unwrap();
        assert_eq!(read(&p).unwrap(), e);
        let _ = std::fs::remove_file(p);
    }

    /// `config.json` of the real checkpoints (35B: quantization_config and vision_config left out), read from spark1
    /// 2026-10-13: the 35B must come out non-dense, the 8B and the 0.6B dense.
    #[test]
    fn dense_detection_on_the_real_configs() {
        let c35: serde_json::Value = serde_json::from_str(r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"dtype":"bfloat16","image_token_id":248056,"model_type":"qwen3_5_moe","text_config":{"attention_bias":false,"attention_dropout":0.0,"attn_output_gate":true,"bos_token_id":248044,"dtype":"bfloat16","eos_token_id":248044,"full_attention_interval":4,"head_dim":256,"hidden_act":"silu","hidden_size":2048,"initializer_range":0.02,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],"linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,"linear_num_value_heads":32,"linear_value_head_dim":128,"mamba_ssm_dtype":"float32","max_position_embeddings":262144,"model_type":"qwen3_5_moe_text","moe_intermediate_size":512,"mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,"num_attention_heads":16,"num_experts":256,"num_experts_per_tok":8,"num_hidden_layers":40,"num_key_value_heads":2,"output_router_logits":false,"pad_token_id":null,"partial_rotary_factor":0.25,"rms_norm_eps":1e-06,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000,"rope_type":"default"},"router_aux_loss_coef":0.001,"shared_expert_intermediate_size":512,"tie_word_embeddings":false,"use_cache":true,"vocab_size":248320},"tie_word_embeddings":false,"transformers_version":"5.7.0.dev0","video_token_id":248057,"vision_end_token_id":248054,"vision_start_token_id":248053}"#).unwrap();
        let c8: serde_json::Value = serde_json::from_str(r#"{"architectures":["Qwen3ForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"bos_token_id":151643,"eos_token_id":151645,"head_dim":128,"hidden_act":"silu","hidden_size":4096,"initializer_range":0.02,"intermediate_size":12288,"max_position_embeddings":40960,"max_window_layers":36,"model_type":"qwen3","num_attention_heads":32,"num_hidden_layers":36,"num_key_value_heads":8,"rms_norm_eps":1e-06,"rope_scaling":null,"rope_theta":1000000,"sliding_window":null,"tie_word_embeddings":false,"torch_dtype":"bfloat16","transformers_version":"4.51.0","use_cache":true,"use_sliding_window":false,"vocab_size":151936}"#).unwrap();
        assert!(has_non_dense_marker(&c35), "Qwen3.6-35B-A3B: experts and linear-attention layers");
        assert!(!has_non_dense_marker(&c8), "Qwen3-8B is dense");
        let c06: serde_json::Value = serde_json::from_str(r#"{"architectures":["Qwen3ForCausalLM"],"hidden_size":1024,"num_hidden_layers":28,"layer_types":["full_attention","full_attention"],"use_sliding_window":false}"#).unwrap();
        assert!(!has_non_dense_marker(&c06));
    }

    #[test]
    fn dense_detection() {
        let dense = serde_json::json!({"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 28, "hidden_size": 1024});
        assert!(!has_non_dense_marker(&dense));
        let moe = serde_json::json!({"text_config": {"num_experts": 256, "layer_types": ["linear_attention", "full_attention"]}});
        assert!(has_non_dense_marker(&moe));
        let hybrid = serde_json::json!({"layer_types": ["linear_attention", "full_attention"]});
        assert!(has_non_dense_marker(&hybrid));
        let dense_sliding = serde_json::json!({"layer_types": ["full_attention", "full_attention"], "num_experts": 0});
        assert!(!has_non_dense_marker(&dense_sliding));
        let gdn = serde_json::json!({"text_config": {"linear_num_value_heads": 32}});
        assert!(has_non_dense_marker(&gdn));
    }
}
