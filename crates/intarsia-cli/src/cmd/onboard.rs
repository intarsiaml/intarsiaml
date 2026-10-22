//! `intarsia onboard`: a new model's first hours in one command (ARCHITECTURE requirement #1, docs/adding-a-model.md).
//!
//! Two tiers, back to back, with the quick numbers published before the in-depth tier starts:
//!
//! 1. `recipe`: `recipe new`'s analysis. A covering recipe is used as is; otherwise a scaffold goes to
//!    `<out>/recipe.toml`, and the run stops (exit 3) while `TODO(recipe-new)` lines remain for a human.
//! 2. `smoke`: `intarsia generate` on one prompt.
//! 3. `quick`: `intarsia qualify --quick-only` (doctor, plan, kernels, gate, accuracy vs a golden, speed snapshot).
//! 4. `publish-quick`: `quick.md`, `quick.json`, `quick.prom` (+ `--on-quick-done`). Written before step 5 starts.
//! 5. `in-depth`: `intarsia qualify` again on the same state (long-context accuracy, noise envelope, speed sweep, A/B, tp).
//! 6. `publish-in-depth`: `in-depth.md`, `in-depth.json`, `in-depth.prom`.
//!
//! The qualify, accuracy and bench code paths are not copied: `quick` and `in-depth` run `intarsia qualify` as a
//! subprocess on `<out>/qualify/` and the reports are rendered from its `state.json` with qualify's own report code.
//! What this module owns is the list of stages ([`STAGES`], a few lines per stage), its state file
//! (`onboard-state.json`: per-stage verdict, start, end), the resume rule and the timeline written into every report.
//!
//! Resume: run again with the same `--out`. A stage that finished (PASS or FAIL) is skipped; an ERROR or SKIP stage and
//! a stage that was running when the process died run again (qualify then skips its own finished stages).
//!
//! Exit code: 0 all gates passed, 1 a gate failed, 2 the pipeline itself failed or stopped early, 3 the recipe needs a
//! human.

use super::qualify::{facts, proc, report, stamp_at, state::now, state::State, state::Verdict, BUILD_STAMP};
use super::recipe;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const STATE_FILE: &str = "onboard-state.json";
const STATE_VERSION: u32 = 1;
const SMOKE_PROMPT: &str = "The capital of France is";

#[derive(clap::Args, Debug, Clone)]
pub struct OnboardArgs {
    /// Model directory (config.json, tokenizer.json, *.safetensors).
    #[arg(long, short = 'm')]
    pub model: PathBuf,
    /// Backend: cpu, cuda, hip or metal (the binary must be built with the feature).
    #[arg(long, default_value = "cpu")]
    pub backend: String,
    /// Device ordinal for GPU backends.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Tensor-parallel degree of the in-depth tp variant.
    #[arg(long, default_value_t = 1)]
    pub tp: usize,
    /// Where the accuracy reference comes from: an OpenAI-compatible vLLM-style URL (`http://host:port/v1`), an HF model
    /// directory (fp32 golden), or a golden `.safetensors` file. Default: a golden found by model name, else made.
    #[arg(long)]
    pub golden_from: Option<String>,
    /// Report directory (default: reports/<model>/onboard-<timestamp>/). Give the same one again to resume.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Stop after the quick tier.
    #[arg(long)]
    pub quick_only: bool,
    /// Do not start the in-depth tier when a quick gate failed.
    #[arg(long)]
    pub fail_fast: bool,
    /// Run this stage again although it finished (repeatable): recipe, smoke, quick, publish-quick, in-depth,
    /// publish-in-depth. Re-running a stage also re-publishes the report that reads it.
    #[arg(long)]
    pub redo: Vec<String>,
    /// Recipe file instead of the matched or generated one.
    #[arg(long)]
    pub recipe: Option<PathBuf>,
    /// Extra directory with recipe *.toml files.
    #[arg(long)]
    pub recipe_dir: Vec<PathBuf>,
    /// Jinja chat template file for every stage.
    #[arg(long)]
    pub chat_template_file: Option<PathBuf>,
    /// Total memory a model-loading process may use, GiB (a fleetq `--mem` lease).
    #[arg(long)]
    pub mem_budget: Option<f64>,
    /// KV cache dtype override for every stage.
    #[arg(long)]
    pub kv_dtype: Option<String>,
    /// Name used in reports and the default report path (default: derived from the model directory).
    #[arg(long)]
    pub model_name: Option<String>,
    /// Push the headline numbers to this Prometheus Pushgateway (`http://host:9091`) when the quick tier and the in-depth
    /// tier are done (done by qualify). `quick.prom` / `in-depth.prom` are always written, for a textfile collector.
    #[arg(long)]
    pub push_metrics: Option<String>,
    /// Shell command run in the background once `quick.md` is written, with its path as `$1` (also
    /// `$INTARSIA_QUICK_REPORT`, `$INTARSIA_QUICK_JSON`, `$INTARSIA_REPORT_DIR`).
    #[arg(long)]
    pub on_quick_done: Option<String>,
    /// The intarsia binary the stages run (default: this executable).
    #[arg(long)]
    pub bin: Option<PathBuf>,
    /// Anything after `--` goes to `intarsia qualify` as is (`--skip ab`, `--long-sizes 1024,4096`, `--golden-dir ...`).
    #[arg(last = true)]
    pub qualify_args: Vec<String>,
}

// ---------------------------------------------------------------------------------------------------------------
// the stage list

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    Prepare,
    Quick,
    InDepth,
}

impl Tier {
    fn label(self) -> &'static str {
        match self {
            Tier::Prepare => "prepare",
            Tier::Quick => "quick",
            Tier::InDepth => "in-depth",
        }
    }
}

/// What a stage returns.
pub struct StageOut {
    verdict: Verdict,
    reason: String,
    facts: Map<String, Value>,
    artifacts: Vec<String>,
    /// Stop the pipeline after this stage (only stages marked `always` still run).
    halt: bool,
}

impl StageOut {
    fn new(verdict: Verdict, reason: impl Into<String>) -> StageOut {
        StageOut { verdict, reason: reason.into(), facts: Map::new(), artifacts: vec![], halt: false }
    }
    fn pass(reason: impl Into<String>) -> StageOut {
        StageOut::new(Verdict::Pass, reason)
    }
    fn fail(reason: impl Into<String>) -> StageOut {
        StageOut::new(Verdict::Fail, reason)
    }
    /// The pipeline itself failed: stops the run.
    fn error(reason: impl Into<String>) -> StageOut {
        StageOut { halt: true, ..StageOut::new(Verdict::Error, reason) }
    }
    fn fact(mut self, k: &str, v: impl Into<Value>) -> StageOut {
        self.facts.insert(k.to_string(), v.into());
        self
    }
    fn art(mut self, a: impl Into<String>) -> StageOut {
        self.artifacts.push(a.into());
        self
    }
    fn halting(mut self) -> StageOut {
        self.halt = true;
        self
    }
}

pub type StageFn<C> = fn(&mut C, &OnbState) -> Result<StageOut>;

/// One stage: a few lines to add one.
pub struct StageDef<C> {
    pub id: &'static str,
    pub title: &'static str,
    pub tier: Tier,
    /// Skipped (not recorded) unless these stages have a record: the publish of the in-depth tier needs `in-depth`.
    pub needs: &'static [&'static str],
    /// A finished stage runs again when one of these ran in this invocation: a report follows what it reports on.
    pub refresh: &'static [&'static str],
    /// Runs even after a stage halted the pipeline (the publish stages: a failed run still gets its report).
    pub always: bool,
    pub run: StageFn<C>,
}

const STAGES: &[StageDef<Ctx>] = &[
    StageDef {
        id: "recipe",
        title: "recipe: existing, or scaffold (stops while TODOs remain)",
        tier: Tier::Prepare,
        needs: &[],
        refresh: &[],
        always: false,
        run: st_recipe,
    },
    StageDef {
        id: "smoke",
        title: "smoke: load and generate one prompt",
        tier: Tier::Prepare,
        needs: &[],
        refresh: &[],
        always: false,
        run: st_smoke,
    },
    StageDef {
        id: "quick",
        title: "quick tier: qualify --quick-only (gate, accuracy, speed snapshot)",
        tier: Tier::Quick,
        needs: &[],
        refresh: &[],
        always: false,
        run: st_quick,
    },
    StageDef {
        id: "publish-quick",
        title: "publish quick.md / quick.json / quick.prom",
        tier: Tier::Quick,
        needs: &[],
        refresh: &["recipe", "smoke", "quick"],
        always: true,
        run: st_publish_quick,
    },
    StageDef {
        id: "in-depth",
        title: "in-depth tier: qualify (long-context accuracy, envelope, speed sweep, A/B, tp)",
        tier: Tier::InDepth,
        needs: &[],
        refresh: &[],
        always: false,
        run: st_in_depth,
    },
    StageDef {
        id: "publish-in-depth",
        title: "publish in-depth.md / in-depth.json / in-depth.prom",
        tier: Tier::InDepth,
        needs: &["in-depth"],
        refresh: &["in-depth"],
        always: true,
        run: st_publish_in_depth,
    },
];

// ---------------------------------------------------------------------------------------------------------------
// state and runner

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StageRec {
    pub id: String,
    pub title: String,
    pub tier: Tier,
    pub verdict: Verdict,
    pub reason: String,
    pub started_unix: u64,
    pub ended_unix: u64,
    pub wall_s: f64,
    /// The stage stopped the pipeline.
    pub halted: bool,
    /// When the first attempt started, if this record is a re-run of a stage that was killed or ended in ERROR / SKIP
    /// (`started_unix` is the attempt that produced this record).
    #[serde(default)]
    pub first_started_unix: Option<u64>,
    pub facts: Map<String, Value>,
    pub artifacts: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Running {
    pub id: String,
    pub started_unix: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OnbState {
    pub version: u32,
    /// What this report directory is about; a directory made for something else is not resumed.
    pub key: Value,
    pub created_unix: u64,
    /// The stage that was running at the last save: after a kill, this is where the run died.
    pub running: Option<Running>,
    pub quick_published_unix: Option<u64>,
    pub in_depth_published_unix: Option<u64>,
    pub stages: Vec<StageRec>,
}

impl OnbState {
    pub fn new(key: Value) -> OnbState {
        OnbState {
            version: STATE_VERSION,
            key,
            created_unix: now(),
            running: None,
            quick_published_unix: None,
            in_depth_published_unix: None,
            stages: vec![],
        }
    }

    pub fn load(dir: &Path) -> Result<Option<OnbState>> {
        let p = dir.join(STATE_FILE);
        if !p.exists() {
            return Ok(None);
        }
        let t = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let s: OnbState =
            serde_json::from_str(&t).with_context(|| format!("{} is not an onboard state file", p.display()))?;
        if s.version != STATE_VERSION {
            bail!("{} has state version {}, this build writes {}", p.display(), s.version, STATE_VERSION);
        }
        Ok(Some(s))
    }

    /// Atomic (temp file, then rename): a kill never leaves half a state file.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join(format!("{STATE_FILE}.tmp"));
        std::fs::write(&tmp, serde_json::to_string_pretty(self)? + "\n")?;
        std::fs::rename(&tmp, dir.join(STATE_FILE)).with_context(|| format!("writing {STATE_FILE}"))
    }

    pub fn get(&self, id: &str) -> Option<&StageRec> {
        self.stages.iter().find(|s| s.id == id)
    }

    /// A stage `--resume` style skipping applies to: its gate ran (PASS or FAIL).
    pub fn finished(&self, id: &str) -> Option<&StageRec> {
        self.get(id).filter(|r| r.verdict.is_done())
    }

    fn put(&mut self, rec: StageRec) {
        match self.stages.iter().position(|s| s.id == rec.id) {
            Some(i) => self.stages[i] = rec,
            None => self.stages.push(rec),
        }
    }

    fn fact(&self, stage: &str, key: &str) -> Option<&Value> {
        self.get(stage).and_then(|r| r.facts.get(key))
    }
}

pub struct RunOpts<'a> {
    pub quick_only: bool,
    pub redo: &'a [String],
}

#[derive(Debug, Default)]
pub struct RunSummary {
    /// A stage halted the pipeline.
    pub stopped: bool,
    /// Stages that ran in this invocation, in order.
    pub executed: Vec<&'static str>,
    /// Stages skipped because an earlier invocation finished them.
    pub resumed: Vec<&'static str>,
}

/// Run `defs` in order against `st`, saving `dir/onboard-state.json` after every stage. The generic parameter is the
/// stage context, so the order and resume rules are testable with stub stages.
pub fn run_pipeline<C>(
    defs: &[StageDef<C>],
    ctx: &mut C,
    st: &mut OnbState,
    dir: &Path,
    opts: &RunOpts,
    say: &dyn Fn(&str),
) -> Result<RunSummary> {
    let mut sum = RunSummary::default();
    for def in defs {
        if opts.quick_only && def.tier == Tier::InDepth {
            continue;
        }
        let redo = opts.redo.iter().any(|r| r == def.id) || def.refresh.iter().any(|r| sum.executed.contains(r));
        if !redo {
            if let Some(rec) = st.finished(def.id) {
                say(&format!("{}: already done ({}), not again", def.id, rec.verdict.label()));
                if rec.halted {
                    sum.stopped = true;
                }
                sum.resumed.push(def.id);
                continue;
            }
        }
        if sum.stopped && !def.always {
            continue;
        }
        if def.needs.iter().any(|n| st.get(n).is_none()) {
            continue;
        }
        say(&format!("== {}: {}", def.id, def.title));
        let started = now();
        let first_started = st.running.as_ref().filter(|r| r.id == def.id).map(|r| r.started_unix).or_else(|| {
            st.get(def.id).filter(|r| !r.verdict.is_done()).map(|r| r.first_started_unix.unwrap_or(r.started_unix))
        });
        let t = Instant::now();
        st.running = Some(Running { id: def.id.to_string(), started_unix: started });
        st.save(dir)?;
        let out = (def.run)(ctx, st).unwrap_or_else(|e| StageOut::error(format!("{e:#}")));
        let rec = StageRec {
            id: def.id.to_string(),
            title: def.title.to_string(),
            tier: def.tier,
            verdict: out.verdict,
            reason: out.reason,
            started_unix: started,
            ended_unix: now(),
            wall_s: t.elapsed().as_secs_f64(),
            halted: out.halt,
            first_started_unix: first_started,
            facts: out.facts,
            artifacts: out.artifacts,
        };
        say(&format!(
            "{}: {} in {}{}",
            def.id,
            rec.verdict.label(),
            report::fmt_dur(rec.wall_s),
            if rec.reason.is_empty() { String::new() } else { format!(" ({})", first_line(&rec.reason)) }
        ));
        sum.stopped |= rec.halted;
        st.running = None;
        st.put(rec);
        st.save(dir)?;
        sum.executed.push(def.id);
    }
    Ok(sum)
}

/// 0 all gates passed, 1 a gate failed, 2 the pipeline failed or ended before its last stage, 3 the recipe needs a human.
pub fn exit_code<C>(defs: &[StageDef<C>], quick_only: bool, st: &OnbState) -> i32 {
    if st.stages.iter().any(|s| s.facts.get("needs_human") == Some(&Value::Bool(true)) && !s.verdict.is_done()) {
        return 3;
    }
    let complete = defs
        .iter()
        .filter(|d| !(quick_only && d.tier == Tier::InDepth))
        .all(|d| st.get(d.id).is_some() || !d.needs.is_empty() && d.needs.iter().any(|n| st.get(n).is_none()));
    if !complete || st.stages.iter().any(|s| s.verdict == Verdict::Error) {
        return 2;
    }
    if st.stages.iter().any(|s| s.verdict == Verdict::Fail) {
        return 1;
    }
    0
}

// ---------------------------------------------------------------------------------------------------------------
// timeline and report text

/// `2026-10-21T17:30:36Z` for a unix time.
pub fn iso(unix: u64) -> String {
    let s = stamp_at(unix);
    format!("{}-{}-{}T{}:{}:{}Z", &s[0..4], &s[4..6], &s[6..8], &s[9..11], &s[11..13], &s[13..15])
}

/// The stage timeline as JSON: the onboard stages and the qualify stages they ran, with start, end and duration.
pub fn timeline_json(st: &OnbState, q: &State, upto: Option<Tier>) -> Value {
    let keep = |t: Tier| upto.map_or(true, |u| t <= u);
    let onb: Vec<Value> = st
        .stages
        .iter()
        .filter(|s| keep(s.tier))
        .map(|s| {
            json!({"id": s.id, "tier": s.tier.label(), "verdict": s.verdict.label(), "started_unix": s.started_unix,
                   "started": iso(s.started_unix), "ended_unix": s.ended_unix, "ended": iso(s.ended_unix), "wall_s": s.wall_s,
                   "first_started_unix": s.first_started_unix, "first_started": s.first_started_unix.map(iso)})
        })
        .collect();
    let qual: Vec<Value> = q
        .stages
        .iter()
        .filter(|s| upto != Some(Tier::Quick) || s.section == "quick")
        .map(|s| {
            let end = s.started_unix + s.wall_s.round() as u64;
            json!({"id": s.id, "section": s.section, "verdict": s.verdict.label(), "started_unix": s.started_unix,
                   "started": iso(s.started_unix), "ended_unix": end, "ended": iso(end), "wall_s": s.wall_s})
        })
        .collect();
    json!({"stages": onb, "qualify_stages": qual})
}

/// The markdown table appended to the qualify report.
pub fn timeline_md(tl: &Value) -> String {
    let mut o = String::from("\n## Stage timeline\n\nAll times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows \
         its last attempt; the qualify rows show every sub-stage with the time it actually ran.\n\n");
    o.push_str("| Stage | Tier | Verdict | Start | End | Duration | Note |\n|---|---|---|---|---|---|---|\n");
    let row = |o: &mut String, r: &Value, tier: &str| {
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} | {} |",
            r["id"].as_str().unwrap_or("?"),
            tier,
            r["verdict"].as_str().unwrap_or("?"),
            r["started"].as_str().unwrap_or("?"),
            r["ended"].as_str().unwrap_or("?"),
            report::fmt_dur(r["wall_s"].as_f64().unwrap_or(0.0)),
            r["first_started"].as_str().map(|f| format!("resumed; first attempt started {f}")).unwrap_or_default(),
        );
    };
    for r in tl["stages"].as_array().into_iter().flatten() {
        row(&mut o, r, r["tier"].as_str().unwrap_or("?"));
    }
    for r in tl["qualify_stages"].as_array().into_iter().flatten() {
        let tier = format!("qualify {}", r["section"].as_str().unwrap_or("?"));
        row(&mut o, r, &tier);
    }
    o
}

/// Prometheus lines for the onboard timeline (appended to qualify's headline metrics).
pub fn timeline_prom(model: &str, backend: &str, tl: &Value, tier: &str, published_unix: u64) -> String {
    let l = |extra: &str| format!("model=\"{}\",backend=\"{}\"{extra}", report::esc(model), report::esc(backend));
    let mut o = String::new();
    let _ = writeln!(
        o,
        "# HELP intarsia_onboard_published_timestamp_seconds Unix time the report of this tier was written.\n# TYPE intarsia_onboard_published_timestamp_seconds gauge"
    );
    let _ = writeln!(
        o,
        "intarsia_onboard_published_timestamp_seconds{{{}}} {published_unix}",
        l(&format!(",tier=\"{tier}\""))
    );
    let _ = writeln!(
        o,
        "# HELP intarsia_onboard_stage_seconds Wall time of an onboard stage.\n# TYPE intarsia_onboard_stage_seconds gauge"
    );
    for r in tl["stages"].as_array().into_iter().flatten() {
        let _ = writeln!(
            o,
            "intarsia_onboard_stage_seconds{{{}}} {:.1}",
            l(&format!(
                ",stage=\"{}\",tier=\"{}\",verdict=\"{}\"",
                r["id"].as_str().unwrap_or("?"),
                r["tier"].as_str().unwrap_or("?"),
                r["verdict"].as_str().unwrap_or("?")
            )),
            r["wall_s"].as_f64().unwrap_or(0.0)
        );
    }
    let _ = writeln!(
        o,
        "# HELP intarsia_onboard_stage_end_timestamp_seconds Unix time an onboard stage ended.\n# TYPE intarsia_onboard_stage_end_timestamp_seconds gauge"
    );
    for r in tl["stages"].as_array().into_iter().flatten() {
        let _ = writeln!(
            o,
            "intarsia_onboard_stage_end_timestamp_seconds{{{}}} {}",
            l(&format!(",stage=\"{}\"", r["id"].as_str().unwrap_or("?"))),
            r["ended_unix"].as_u64().unwrap_or(0)
        );
    }
    o
}

/// The verdict of a published tier: qualify's, unless an onboard stage (recipe, smoke, the qualify run itself) says worse.
/// A recipe that needs a human or a smoke test that crashed must not read as "PASS (0 stages)".
pub fn tier_verdict(st: &OnbState, qualify_verdict: &str, upto: Option<Tier>) -> &'static str {
    let recs = || st.stages.iter().filter(|s| upto.map_or(true, |u| s.tier <= u));
    if recs().any(|s| s.verdict == Verdict::Error) || qualify_verdict == "ERROR" {
        "ERROR"
    } else if recs().any(|s| s.verdict == Verdict::Fail) || qualify_verdict == "FAIL" {
        "FAIL"
    } else if qualify_verdict == "INCOMPLETE" {
        "INCOMPLETE"
    } else {
        "PASS"
    }
}

/// The lines under the title of a published report: the onboard verdict and what the stages before qualify said.
pub fn banner_md(st: &OnbState, verdict: &str, upto: Option<Tier>) -> String {
    let mut o = format!("> **Onboard verdict: {verdict}.**");
    for s in st.stages.iter().filter(|s| s.tier == Tier::Prepare || upto.map_or(true, |u| s.tier <= u)) {
        if s.id.starts_with("publish") {
            continue;
        }
        let _ = write!(o, " {}: {}", s.id, s.verdict.label());
        if !s.reason.is_empty() && s.verdict != Verdict::Pass {
            let _ = write!(o, " ({})", first_line(&s.reason));
        }
        o.push(';');
    }
    o.push_str("\n\n");
    o
}

/// `# TODO(recipe-new)` marker lines of a recipe file (the scaffold header mentions the marker mid-line; only the
/// markers at the start of a comment count).
pub fn recipe_todos(toml: &str) -> Vec<String> {
    toml.lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("# TODO(recipe-new)"))
        .map(|l| l.trim_start_matches("# TODO(recipe-new):").trim().to_string())
        .collect()
}

/// `--golden-from` as qualify flags: a URL or an HF directory is a reference, a file is a golden.
pub fn golden_flags(from: &str) -> Result<Vec<String>> {
    if from.starts_with("http://") || from.starts_with("https://") {
        return Ok(vec!["--reference".into(), from.to_string()]);
    }
    let p = Path::new(from);
    if p.is_dir() {
        Ok(vec!["--reference".into(), from.to_string()])
    } else if p.is_file() {
        Ok(vec!["--golden".into(), from.to_string()])
    } else {
        bail!("--golden-from {from}: not a URL, a model directory or a golden file")
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

// ---------------------------------------------------------------------------------------------------------------
// the real stages

pub struct Ctx {
    a: OnboardArgs,
    out: PathBuf,
    qdir: PathBuf,
    name: String,
    bin: PathBuf,
    t0: Instant,
    hooks: Vec<(String, std::process::Child)>,
}

impl Ctx {
    fn say(&self, m: &str) {
        eprintln!("[onboard {:>6.0}s] {m}", self.t0.elapsed().as_secs_f64());
    }

    /// The recipe a stage must be given (explicit, or generated into the report directory).
    fn recipe_arg(&self, st: &OnbState) -> Option<PathBuf> {
        self.a.recipe.clone().or_else(|| st.fact("recipe", "recipe_path").and_then(|v| v.as_str()).map(PathBuf::from))
    }

    /// Flags every model-loading command takes, `generate` and `qualify` alike.
    fn load_args(&self, st: &OnbState) -> Vec<String> {
        let mut v: Vec<String> = vec![
            "--model".into(),
            self.a.model.display().to_string(),
            "--backend".into(),
            self.a.backend.clone(),
            "--device".into(),
            self.a.device.to_string(),
        ];
        if let Some(r) = self.recipe_arg(st) {
            v.extend(["--recipe".into(), r.display().to_string()]);
        }
        for d in &self.a.recipe_dir {
            v.extend(["--recipe-dir".into(), d.display().to_string()]);
        }
        if let Some(t) = &self.a.chat_template_file {
            v.extend(["--chat-template-file".into(), t.display().to_string()]);
        }
        if let Some(m) = self.a.mem_budget {
            v.extend(["--mem-budget".into(), m.to_string()]);
        }
        if let Some(k) = &self.a.kv_dtype {
            v.extend(["--kv-dtype".into(), k.clone()]);
        }
        v
    }

    fn run_bin(&self, stem: &str, args: &[String], timeout: Duration) -> Result<proc::Run> {
        let mut cmd = Command::new(&self.bin);
        cmd.args(args);
        if self.a.backend == "hip" && std::env::var_os("GPU_MAX_HW_QUEUES").is_none() {
            cmd.env("GPU_MAX_HW_QUEUES", "1");
        }
        self.say(&format!("log: {}.err", self.out.join("logs").join(stem).display()));
        proc::run_captured(cmd, &self.out.join("logs").join(stem), timeout)
    }

    fn qualify_state(&self) -> State {
        State::load(&self.qdir).ok().flatten().unwrap_or_else(|| State::new(json!({})))
    }

    fn report_meta(&self, quick_only: bool) -> report::Meta {
        report::Meta {
            model: self.name.clone(),
            model_dir: self.a.model.display().to_string(),
            backend: self.a.backend.clone(),
            tp: self.a.tp,
            reference: self.a.golden_from.clone(),
            build_stamp: BUILD_STAMP.map(String::from),
            version: env!("CARGO_PKG_VERSION").to_string(),
            generated_unix: now(),
            quick_only,
        }
    }
}

fn st_recipe(c: &mut Ctx, _: &OnbState) -> Result<StageOut> {
    // a scaffold of an earlier run is the human's to finish: never regenerate it over their edits
    let scaffold = c.out.join("recipe.toml");
    let explicit = c.a.recipe.clone().or_else(|| scaffold.exists().then(|| scaffold.clone()));
    if let Some(path) = explicit {
        let todos =
            recipe_todos(&std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?);
        return Ok(recipe_verdict(&path, &todos, c.a.recipe.is_none(), c));
    }
    let an = recipe::analyse(&c.a.model, &c.a.recipe_dir)?;
    let (r, t) = an.tensor_templates_resolved;
    if let Some(src) = &an.covered_by {
        return Ok(StageOut::pass(format!(
            "{src} covers the model ({r}/{t} tensor templates resolve, quant {})",
            an.quant
        ))
        .fact("covered_by", src.clone())
        .fact("family", an.family.clone())
        .fact("quant", an.quant.clone()));
    }
    std::fs::write(&scaffold, &an.toml).with_context(|| format!("writing {}", scaffold.display()))?;
    let todos = recipe_todos(&an.toml);
    let mut o = recipe_verdict(&scaffold, &todos, true, c);
    o = o.fact("family", an.family.clone()).fact("quant", an.quant.clone());
    Ok(o)
}

fn recipe_verdict(path: &Path, todos: &[String], generated: bool, c: &Ctx) -> StageOut {
    let p = path.display().to_string();
    if todos.is_empty() {
        let how = if generated { "generated, no TODOs left" } else { "given with --recipe" };
        return StageOut::pass(format!("recipe {p} ({how})")).fact("recipe_path", p.clone()).art(p);
    }
    let mut msg = format!("{} item(s) in {p} need a human before the run can go on:\n", todos.len());
    for (i, t) in todos.iter().enumerate() {
        let _ = writeln!(msg, "  {}. {t}", i + 1);
    }
    let _ = write!(
        msg,
        "Fix them (see docs/adding-a-model.md, \"When you need code\"), delete the TODO(recipe-new) lines, then run the same \
         command again with --out {}",
        c.out.display()
    );
    eprintln!("{msg}");
    StageOut::error(msg).halting().fact("recipe_path", p.clone()).fact("needs_human", true).art(p)
}

fn st_smoke(c: &mut Ctx, st: &OnbState) -> Result<StageOut> {
    let mut args = vec!["generate".to_string()];
    args.extend(c.load_args(st));
    args.extend(["--prompt".into(), SMOKE_PROMPT.into(), "--max-tokens".into(), "16".into()]);
    let run = c.run_bin("smoke", &args, Duration::from_secs(30 * 60))?;
    if !run.ok() {
        return Ok(StageOut::error(format!("generate {}: {}", run.describe(), run.tail(5))));
    }
    let text = run.stdout().trim().to_string();
    let mut o = if text.chars().any(|ch| ch.is_alphanumeric()) {
        StageOut::pass(format!("generated: {:?}", text.chars().take(80).collect::<String>()))
    } else {
        StageOut::fail(format!("no text came out of generate (stdout {:?})", text)).halting()
    };
    // `decode 123 ms (45.6 tok/s)` on stderr
    if let Some(rate) = run
        .stderr()
        .lines()
        .find(|l| l.starts_with("finish="))
        .and_then(|l| {
            l.rsplit("decode ")
                .next()
                .and_then(|d| d.split('(').nth(1))
                .and_then(|d| d.split(' ').next().map(String::from))
        })
        .and_then(|d| d.parse::<f64>().ok())
    {
        o = o.fact("decode_tok_s", rate);
    }
    Ok(o.fact("completion", text.chars().take(200).collect::<String>()).fact("mentions_paris", text.contains("Paris")))
}

/// Run `intarsia qualify` on `<out>/qualify`, resuming, and turn its exit code into a verdict.
fn run_qualify(c: &mut Ctx, st: &OnbState, quick_only: bool, stem: &str) -> Result<StageOut> {
    let mut args = vec!["qualify".to_string()];
    args.extend(c.load_args(st));
    args.extend([
        "--out".into(),
        c.qdir.display().to_string(),
        "--resume".into(),
        "--model-name".into(),
        c.name.clone(),
    ]);
    if c.a.tp > 1 {
        args.extend(["--tp".into(), c.a.tp.to_string()]);
    }
    if quick_only {
        args.push("--quick-only".into());
    }
    if let Some(g) = &c.a.golden_from {
        args.extend(golden_flags(g)?);
        if g.starts_with("http://") || g.starts_with("https://") {
            // a reference server can also make the 4k golden, so the in-depth tier runs `acc-4k` instead of skipping it
            args.push("--make-golden-4k".into());
        }
    }
    if let Some(u) = &c.a.push_metrics {
        args.extend(["--push-metrics".into(), u.clone()]);
    }
    args.extend(c.a.qualify_args.iter().cloned());
    let run = c.run_bin(stem, &args, Duration::from_secs(24 * 3600))?;
    let q = c.qualify_state();
    let bad: Vec<String> = q
        .stages
        .iter()
        .filter(|s| matches!(s.verdict, Verdict::Fail | Verdict::Error))
        .map(|s| format!("{} {}: {}", s.id, s.verdict.label(), first_line(&s.reason)))
        .collect();
    let detail = bad.join("; ");
    let o = match run.code {
        Some(0) if !run.timed_out => StageOut::pass(format!("qualify exit 0, {} stage(s) recorded", q.stages.len())),
        Some(1) if !run.timed_out => StageOut::fail(format!("qualify exit 1 (a gate failed): {detail}")),
        _ => StageOut::error(format!("qualify {}: {} {}", run.describe(), detail, run.tail(3))),
    };
    Ok(o.fact("qualify_exit", run.code.map(Value::from).unwrap_or(Value::Null)).art("qualify/state.json"))
}

fn st_quick(c: &mut Ctx, st: &OnbState) -> Result<StageOut> {
    let mut o = run_qualify(c, st, true, "quick")?;
    if o.verdict == Verdict::Fail && c.a.fail_fast {
        o = o.halting();
        o.reason.push_str(" (--fail-fast: the in-depth tier is not started)");
    }
    Ok(o)
}

fn st_in_depth(c: &mut Ctx, st: &OnbState) -> Result<StageOut> {
    run_qualify(c, st, false, "in-depth")
}

/// Write `path` atomically, so a reader (or the mtime) never sees half a report.
fn write_atomic(path: &Path, text: &str) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
}

/// Render and write one tier's report triple (`<base>.md`, `.json`, `.prom`). The reports are qualify's own, from
/// qualify's state, with the onboard timeline appended.
fn publish(c: &mut Ctx, st: &OnbState, full: bool) -> Result<(PathBuf, u64)> {
    let q = c.qualify_state();
    let meta = c.report_meta(c.a.quick_only);
    let tl = timeline_json(st, &q, if full { None } else { Some(Tier::Quick) });
    let published = now();
    let base = if full { "in-depth" } else { "quick" };
    let upto = if full { None } else { Some(Tier::Quick) };
    let mut js = report::summary_json(&q, &meta, full);
    let verdict = tier_verdict(st, js["verdict"].as_str().unwrap_or("ERROR"), upto);
    let mut md = report::render(&q, &meta, full);
    // the qualify report's first line is its title: the onboard banner goes under it
    let at = md.find("\n\n").map(|i| i + 2).unwrap_or(0);
    md.insert_str(at, &banner_md(st, verdict, upto));
    md.push_str(&timeline_md(&tl));
    let _ = writeln!(
        md,
        "\n_Published {} by `intarsia onboard` (report directory `{}`)._",
        iso(published),
        c.out.display()
    );
    js["qualify_verdict"] = js["verdict"].clone();
    js["verdict"] = json!(verdict);
    js["exit_code"] = json!(match verdict {
        "PASS" => 0,
        "FAIL" => 1,
        _ => 2,
    });
    js["kind"] = json!("intarsia-onboard");
    js["tier"] = json!(if full { "in-depth" } else { "quick" });
    js["published_unix"] = json!(published);
    js["published"] = json!(iso(published));
    js["onboard"] = json!({
        "started_unix": st.created_unix, "started": iso(st.created_unix),
        "timeline": tl,
        "recipe": st.get("recipe").map(|r| json!({"verdict": r.verdict.label(), "reason": r.reason, "recipe_path": r.facts.get("recipe_path"), "covered_by": r.facts.get("covered_by")})),
        "smoke": st.get("smoke").map(|r| json!({"verdict": r.verdict.label(), "reason": r.reason, "decode_tok_s": r.facts.get("decode_tok_s"), "completion": r.facts.get("completion")})),
    });
    let mut prom = report::prometheus_text(&js);
    prom.push_str(&timeline_prom(&c.name, &c.a.backend, &tl, base, published));
    let md_path = c.out.join(format!("{base}.md"));
    // the .md goes last: its mtime is the moment the tier counts as published
    write_atomic(&c.out.join(format!("{base}.json")), &(serde_json::to_string_pretty(&js)? + "\n"))?;
    write_atomic(&c.out.join(format!("{base}.prom")), &prom)?;
    write_atomic(&md_path, &md)?;
    let abs = std::fs::canonicalize(&md_path).unwrap_or(md_path);
    println!("{}: {}", if full { "IN-DEPTH DONE" } else { "QUICK DONE" }, abs.display());
    Ok((abs, published))
}

fn st_publish_quick(c: &mut Ctx, st: &OnbState) -> Result<StageOut> {
    let (abs, published) = publish(c, st, false)?;
    if let Some(cmd) = c.a.on_quick_done.clone() {
        let json_path = c.out.join("quick.json");
        let mut sh = Command::new("sh");
        sh.arg("-c")
            .arg(&cmd)
            .arg("onboard")
            .arg(&abs)
            .env("INTARSIA_QUICK_REPORT", &abs)
            .env("INTARSIA_QUICK_JSON", std::fs::canonicalize(&json_path).unwrap_or(json_path))
            .env("INTARSIA_REPORT_DIR", std::fs::canonicalize(&c.out).unwrap_or(c.out.clone()));
        match sh.spawn() {
            Ok(ch) => c.hooks.push((cmd, ch)),
            Err(e) => c.say(&format!("WARNING: --on-quick-done could not start: {e}")),
        }
    }
    Ok(StageOut::pass(format!("quick.md written {}", iso(published)))
        .fact("published_unix", published)
        .art("quick.md")
        .art("quick.json")
        .art("quick.prom"))
}

fn st_publish_in_depth(c: &mut Ctx, st: &OnbState) -> Result<StageOut> {
    let (_, published) = publish(c, st, true)?;
    Ok(StageOut::pass(format!("in-depth.md written {}", iso(published)))
        .fact("published_unix", published)
        .art("in-depth.md")
        .art("in-depth.json")
        .art("in-depth.prom"))
}

// ---------------------------------------------------------------------------------------------------------------

/// A pid file in the report directory: two onboard runs on one directory would race on the state files.
struct Lock(PathBuf);

impl Lock {
    fn take(dir: &Path) -> Result<Lock> {
        let p = dir.join("onboard.lock");
        if let Ok(t) = std::fs::read_to_string(&p) {
            if let Ok(pid) = t.trim().parse::<i32>() {
                #[cfg(unix)]
                // SAFETY: signal 0 only checks that the process exists.
                if unsafe { libc::kill(pid, 0) } == 0 {
                    bail!("{} is in use by onboard process {pid}", dir.display());
                }
            }
        }
        std::fs::write(&p, std::process::id().to_string())?;
        Ok(Lock(p))
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub fn run(args: OnboardArgs) -> Result<()> {
    match pipeline(args) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("onboard: the pipeline itself failed: {e:#}");
            std::process::exit(2);
        }
    }
}

fn pipeline(mut a: OnboardArgs) -> Result<i32> {
    if !a.model.join("config.json").exists() {
        bail!("{}: no config.json (is this a Hugging Face model directory?)", a.model.display());
    }
    for r in &a.redo {
        if !STAGES.iter().any(|d| d.id == r) {
            bail!("--redo {r}: not a stage; stages are {}", STAGES.iter().map(|d| d.id).collect::<Vec<_>>().join(", "));
        }
    }
    if let Some(g) = &a.golden_from {
        golden_flags(g)?;
    }
    a.model = std::fs::canonicalize(&a.model).unwrap_or(a.model.clone());
    let name = facts::model_name(&a.model, a.model_name.as_deref());
    let out = a
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("reports").join(&name).join(format!("onboard-{}", stamp_at(now()))));
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let _lock = Lock::take(&out)?;
    let key = json!({"model": a.model.display().to_string(), "backend": a.backend});
    let mut st = match OnbState::load(&out)? {
        Some(s) if s.key == key => {
            eprintln!("[onboard] resuming {}", out.display());
            if let Some(r) = &s.running {
                eprintln!("[onboard] the last run died in stage {} (started {})", r.id, iso(r.started_unix));
            }
            s
        }
        Some(s) => bail!("{} was made for {} and cannot be resumed for {}", out.display(), s.key, key),
        None => OnbState::new(key),
    };
    let bin = match &a.bin {
        Some(b) => b.clone(),
        None => std::env::current_exe().context("finding the executable")?,
    };
    proc::forward_termination();
    let mut c =
        Ctx { qdir: out.join("qualify"), out: out.clone(), a: a.clone(), name, bin, t0: Instant::now(), hooks: vec![] };
    c.say(&format!("{} on {}, report directory {}", c.name, a.backend, out.display()));
    let say = |m: &str| eprintln!("[onboard] {m}");
    let opts = RunOpts { quick_only: a.quick_only, redo: &a.redo };
    let sum = run_pipeline(STAGES, &mut c, &mut st, &out, &opts, &say)?;
    for (cmd, mut ch) in std::mem::take(&mut c.hooks) {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(120) && ch.try_wait().ok().flatten().is_none() {
            std::thread::sleep(Duration::from_millis(200));
        }
        if ch.try_wait().ok().flatten().map(|s| s.success()) != Some(true) {
            c.say(&format!("WARNING: --on-quick-done did not exit 0: {cmd}"));
            let _ = ch.kill();
        }
    }
    let code = exit_code(STAGES, a.quick_only, &st);
    let verdict = match code {
        0 => "PASS",
        1 => "FAIL",
        3 => "RECIPE NEEDS A HUMAN",
        _ => "ERROR",
    };
    println!(
        "ONBOARD {verdict} (exit {code}{}) in {}",
        if sum.stopped { ", stopped early" } else { "" },
        report::fmt_dur(c.t0.elapsed().as_secs_f64())
    );
    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Stub context: records the order stages ran in, can panic in one (a kill) and give stages a fixed outcome.
    #[derive(Default)]
    struct T {
        ran: Vec<&'static str>,
        panic_at: Option<&'static str>,
        outcome: HashMap<&'static str, (Verdict, bool)>,
    }

    impl T {
        fn step(&mut self, id: &'static str) -> Result<StageOut> {
            self.ran.push(id);
            if self.panic_at == Some(id) {
                panic!("killed in {id}");
            }
            Ok(match self.outcome.get(id) {
                Some((Verdict::Fail, _)) => StageOut::fail("gate failed"),
                Some((Verdict::Error, halt)) => {
                    let o = StageOut::new(Verdict::Error, "tool crashed");
                    if *halt {
                        o.halting()
                    } else {
                        o
                    }
                }
                _ => StageOut::pass(""),
            })
        }
    }

    macro_rules! stub {
        ($f:ident, $id:expr) => {
            fn $f(c: &mut T, _: &OnbState) -> Result<StageOut> {
                c.step($id)
            }
        };
    }
    stub!(t_recipe, "recipe");
    stub!(t_smoke, "smoke");
    stub!(t_quick, "quick");
    stub!(t_pq, "publish-quick");
    stub!(t_deep, "in-depth");
    stub!(t_pd, "publish-in-depth");

    #[allow(clippy::too_many_arguments)]
    const fn def(
        id: &'static str,
        tier: Tier,
        needs: &'static [&'static str],
        refresh: &'static [&'static str],
        always: bool,
        run: StageFn<T>,
    ) -> StageDef<T> {
        StageDef { id, title: id, tier, needs, refresh, always, run }
    }

    // the same shape as STAGES
    const DEFS: &[StageDef<T>] = &[
        def("recipe", Tier::Prepare, &[], &[], false, t_recipe),
        def("smoke", Tier::Prepare, &[], &[], false, t_smoke),
        def("quick", Tier::Quick, &[], &[], false, t_quick),
        def("publish-quick", Tier::Quick, &[], &["recipe", "smoke", "quick"], true, t_pq),
        def("in-depth", Tier::InDepth, &[], &[], false, t_deep),
        def("publish-in-depth", Tier::InDepth, &["in-depth"], &["in-depth"], true, t_pd),
    ];

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("onboard-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn go(dir: &Path, t: &mut T, redo: &[String], quick_only: bool) -> (OnbState, RunSummary) {
        let mut st = OnbState::load(dir).unwrap().unwrap_or_else(|| OnbState::new(json!({"m": 1})));
        let sum = run_pipeline(DEFS, t, &mut st, dir, &RunOpts { quick_only, redo }, &|_| {}).unwrap();
        (st, sum)
    }

    #[test]
    fn stages_run_in_order_and_quick_is_published_before_in_depth() {
        let dir = tmp("order");
        let mut t = T::default();
        let (st, sum) = go(&dir, &mut t, &[], false);
        assert_eq!(t.ran, ["recipe", "smoke", "quick", "publish-quick", "in-depth", "publish-in-depth"]);
        assert_eq!(sum.executed, t.ran);
        assert!(!sum.stopped);
        assert_eq!(exit_code(DEFS, false, &st), 0);
        let (pq, deep) = (st.get("publish-quick").unwrap(), st.get("in-depth").unwrap());
        assert!(pq.ended_unix <= deep.started_unix);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quick_only_leaves_out_the_in_depth_tier() {
        let dir = tmp("quickonly");
        let mut t = T::default();
        let (st, _) = go(&dir, &mut t, &[], true);
        assert_eq!(t.ran, ["recipe", "smoke", "quick", "publish-quick"]);
        assert_eq!(exit_code(DEFS, true, &st), 0);
        // the same directory without --quick-only goes on with the in-depth tier only
        let mut t2 = T::default();
        let (_, sum) = go(&dir, &mut t2, &[], false);
        assert_eq!(t2.ran, ["in-depth", "publish-in-depth"]);
        assert_eq!(sum.resumed, ["recipe", "smoke", "quick", "publish-quick"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_killed_run_resumes_at_the_stage_it_died_in() {
        let dir = tmp("kill");
        let mut t = T { panic_at: Some("in-depth"), ..T::default() };
        let r = catch_unwind(AssertUnwindSafe(|| go(&dir, &mut t, &[], false)));
        assert!(r.is_err());
        // what a kill leaves on disk: the quick tier finished and published, the in-depth stage has no record
        let on_disk = OnbState::load(&dir).unwrap().unwrap();
        assert_eq!(
            on_disk.stages.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            ["recipe", "smoke", "quick", "publish-quick"]
        );
        assert_eq!(on_disk.running.as_ref().unwrap().id, "in-depth");
        let mut t2 = T::default();
        let (st, sum) = go(&dir, &mut t2, &[], false);
        assert_eq!(t2.ran, ["in-depth", "publish-in-depth"], "the quick tier is not run again");
        assert_eq!(sum.resumed, ["recipe", "smoke", "quick", "publish-quick"]);
        assert!(st.running.is_none());
        assert_eq!(exit_code(DEFS, false, &st), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_gate_is_final_but_an_error_runs_again() {
        let dir = tmp("verdicts");
        let mut t = T::default();
        t.outcome.insert("quick", (Verdict::Fail, false));
        t.outcome.insert("in-depth", (Verdict::Error, false));
        let (st, _) = go(&dir, &mut t, &[], false);
        // a FAIL does not stop the in-depth tier by itself; the ERROR makes the run incomplete
        assert_eq!(t.ran.len(), 6);
        assert_eq!(exit_code(DEFS, false, &st), 2);
        let mut t2 = T::default();
        let (st2, sum) = go(&dir, &mut t2, &[], false);
        assert_eq!(t2.ran, ["in-depth", "publish-in-depth"], "FAIL is kept, ERROR is run again");
        assert_eq!(sum.resumed, ["recipe", "smoke", "quick", "publish-quick"]);
        assert_eq!(exit_code(DEFS, false, &st2), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_halting_stage_stops_the_rest_but_not_the_publish() {
        let dir = tmp("halt");
        let mut t = T::default();
        t.outcome.insert("smoke", (Verdict::Error, true));
        let (st, sum) = go(&dir, &mut t, &[], false);
        // quick and in-depth never ran; publish-quick still wrote a report; publish-in-depth had nothing to publish
        assert_eq!(t.ran, ["recipe", "smoke", "publish-quick"]);
        assert!(sum.stopped);
        assert!(st.get("quick").is_none() && st.get("publish-in-depth").is_none());
        assert_eq!(exit_code(DEFS, false, &st), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_finished_halting_stage_still_stops_a_resumed_run() {
        let dir = tmp("halt-fail");
        let mut t = T::default();
        t.outcome.insert("smoke", (Verdict::Error, true));
        let _ = go(&dir, &mut t, &[], false);
        // the human fixed the cause: smoke is an ERROR, so it runs again and the run goes on
        let mut t2 = T::default();
        let (st, _) = go(&dir, &mut t2, &[], false);
        assert_eq!(t2.ran, ["smoke", "quick", "publish-quick", "in-depth", "publish-in-depth"]);
        assert_eq!(exit_code(DEFS, false, &st), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_resumed_stage_remembers_when_its_first_attempt_started() {
        let dir = tmp("first");
        let mut t = T { panic_at: Some("in-depth"), ..T::default() };
        let _ = catch_unwind(AssertUnwindSafe(|| go(&dir, &mut t, &[], false)));
        let killed_at = OnbState::load(&dir).unwrap().unwrap().running.unwrap().started_unix;
        let mut t2 = T::default();
        let (st, _) = go(&dir, &mut t2, &[], false);
        assert_eq!(st.get("in-depth").unwrap().first_started_unix, Some(killed_at));
        assert_eq!(st.get("quick").unwrap().first_started_unix, None, "a stage that ran once has no earlier attempt");
        // an ERROR stage that runs again also keeps its first start
        let dir2 = tmp("first2");
        let mut t3 = T::default();
        t3.outcome.insert("quick", (Verdict::Error, false));
        let (st3, _) = go(&dir2, &mut t3, &[], false);
        let first = st3.get("quick").unwrap().started_unix;
        let (st4, _) = go(&dir2, &mut T::default(), &[], false);
        assert_eq!(st4.get("quick").unwrap().first_started_unix, Some(first));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn redo_and_refresh_rerun_the_stage_and_its_report() {
        let dir = tmp("redo");
        let mut t = T::default();
        let _ = go(&dir, &mut t, &[], false);
        let mut t2 = T::default();
        let _ = go(&dir, &mut t2, &["quick".to_string()], false);
        assert_eq!(t2.ran, ["quick", "publish-quick"], "the quick report follows the quick stage");
        let mut t3 = T::default();
        let _ = go(&dir, &mut t3, &["in-depth".to_string()], false);
        assert_eq!(t3.ran, ["in-depth", "publish-in-depth"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recipe_needing_a_human_exits_3() {
        let mut st = OnbState::new(json!({}));
        let mut f = Map::new();
        f.insert("needs_human".into(), Value::Bool(true));
        st.put(StageRec {
            id: "recipe".into(),
            title: "recipe".into(),
            tier: Tier::Prepare,
            verdict: Verdict::Error,
            reason: "2 TODOs".into(),
            started_unix: 1,
            ended_unix: 2,
            wall_s: 1.0,
            halted: true,
            first_started_unix: None,
            facts: f,
            artifacts: vec![],
        });
        assert_eq!(exit_code(DEFS, false, &st), 3);
    }

    #[test]
    fn state_is_keyed_and_round_trips() {
        let dir = tmp("state");
        let mut t = T::default();
        let (st, _) = go(&dir, &mut t, &[], true);
        let back = OnbState::load(&dir).unwrap().unwrap();
        assert_eq!(back.key, st.key);
        assert_eq!(back.stages.len(), 4);
        assert!(!dir.join(format!("{STATE_FILE}.tmp")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_real_stage_list_is_ordered_and_consistent() {
        let ids: Vec<&str> = STAGES.iter().map(|d| d.id).collect();
        assert_eq!(ids, ["recipe", "smoke", "quick", "publish-quick", "in-depth", "publish-in-depth"]);
        for d in STAGES {
            for n in d.needs.iter().chain(d.refresh) {
                assert!(ids.contains(n), "{} names an unknown stage {n}", d.id);
            }
        }
        // quick is published before the first in-depth stage
        let pq = ids.iter().position(|i| *i == "publish-quick").unwrap();
        let deep = ids.iter().position(|i| *i == "in-depth").unwrap();
        assert!(pq < deep);
    }

    #[test]
    fn timeline_has_start_end_duration_and_iso_times() {
        assert_eq!(iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso(1_792_603_836), "2026-10-21T17:30:36Z");
        let dir = tmp("timeline");
        let mut t = T::default();
        let (mut st, _) = go(&dir, &mut t, &[], false);
        st.stages[0].started_unix = 1_792_603_836;
        st.stages[0].ended_unix = 1_792_603_896;
        st.stages[0].wall_s = 60.0;
        let q = State::new(json!({}));
        let tl = timeline_json(&st, &q, Some(Tier::Quick));
        assert_eq!(tl["stages"].as_array().unwrap().len(), 4, "the quick report holds the quick tier only");
        let md = timeline_md(&tl);
        assert!(
            md.contains("| recipe | prepare | PASS | 2026-10-21T17:30:36Z | 2026-10-21T17:31:36Z | 60s |  |"),
            "{md}"
        );
        let prom = timeline_prom("Qwen3-8B", "metal", &tl, "quick", 5);
        assert!(
            prom.contains("intarsia_onboard_stage_seconds{model=\"Qwen3-8B\",backend=\"metal\",stage=\"recipe\""),
            "{prom}"
        );
        assert!(prom.contains(
            "intarsia_onboard_published_timestamp_seconds{model=\"Qwen3-8B\",backend=\"metal\",tier=\"quick\"} 5"
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stopped_run_does_not_read_as_pass() {
        let dir = tmp("verdict");
        let mut t = T::default();
        t.outcome.insert("smoke", (Verdict::Error, true));
        let (st, _) = go(&dir, &mut t, &[], false);
        // qualify never ran: its state is empty and says PASS (0 stages); the onboard stages say otherwise
        assert_eq!(tier_verdict(&st, "PASS", Some(Tier::Quick)), "ERROR");
        let banner = banner_md(&st, "ERROR", Some(Tier::Quick));
        assert!(
            banner.starts_with("> **Onboard verdict: ERROR.**") && banner.contains("smoke: ERROR (tool crashed)"),
            "{banner}"
        );
        // a clean preparation leaves qualify's verdict alone
        let mut t2 = T::default();
        let dir2 = tmp("verdict2");
        let (ok, _) = go(&dir2, &mut t2, &[], false);
        assert_eq!(tier_verdict(&ok, "PASS", None), "PASS");
        assert_eq!(tier_verdict(&ok, "FAIL", None), "FAIL");
        assert_eq!(tier_verdict(&ok, "INCOMPLETE", None), "INCOMPLETE");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn recipe_todos_count_only_marker_lines() {
        let toml = "# Every `TODO(recipe-new)` below needs a human.\n[model]\n# TODO(recipe-new): rope type yarn, set rope\nrope = \"?\"\n  # TODO(recipe-new): activation gelu\n";
        assert_eq!(recipe_todos(toml), ["rope type yarn, set rope", "activation gelu"]);
        assert!(recipe_todos("[model]\nfamily = \"x\"\n").is_empty());
    }

    #[test]
    fn golden_from_maps_to_qualify_flags() {
        assert_eq!(golden_flags("http://spark:8001/v1").unwrap(), ["--reference", "http://spark:8001/v1"]);
        let dir = tmp("golden");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("g.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(golden_flags(f.to_str().unwrap()).unwrap()[0], "--golden");
        assert_eq!(golden_flags(dir.to_str().unwrap()).unwrap()[0], "--reference");
        assert!(golden_flags("/nonexistent/golden").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
