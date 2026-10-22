//! `intarsia qualify`: qualify a newly downloaded model in one run (ARCHITECTURE requirement #1, docs/adding-a-model.md).
//!
//! A pipeline of stages. The QUICK section finishes in minutes to an hour and publishes its numbers (report, JSON,
//! `--on-quick-done`, `--push-metrics`); the IN-DEPTH section follows in the same run unless `--quick-only`. Every stage
//! runs this same binary as a subprocess (so a stage cannot leak GPU memory into the next one), records its wall time,
//! verdict, provenance and artifacts in `state.json` after it ends, and `--resume` skips the stages that finished (PASS or
//! FAIL; a SKIP or ERROR stage is run again).
//!
//! Exit code: 0 every gate passed, 1 a gate failed, 2 the pipeline itself failed (a tool crashed or timed out, bad
//! arguments, an unreadable report directory, or the `--max-minutes` budget ran out before the end).

// `pub(crate)`: `intarsia onboard` reuses the reports, the state file and the subprocess helpers.
pub(crate) mod facts;
pub(crate) mod proc;
pub(crate) mod report;
pub(crate) mod state;

use anyhow::{anyhow, bail, Context, Result};
use proc::Run;
use serde_json::{json, Map, Value};
use state::{now, StageRecord, State, Verdict};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Set by `scripts/phase1b-acceptance.sh` style builds; absent for a plain `cargo build` (see accuracy/provenance.rs).
pub(crate) const BUILD_STAMP: Option<&str> = option_env!("INTARSIA_BUILD_STAMP");

#[derive(clap::Args, Debug, Clone)]
pub struct QualifyArgs {
    /// Model directory (config.json, tokenizer.json, *.safetensors).
    #[arg(long, short = 'm')]
    pub model: PathBuf,
    /// Backend: cpu, cuda, hip or metal (the binary must be built with the feature).
    #[arg(long, default_value = "cpu")]
    pub backend: String,
    /// Device ordinal for GPU backends.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Tensor-parallel degree for the in-depth tp variant (the quick and the other in-depth stages run on one device).
    #[arg(long, default_value_t = 1)]
    pub tp: usize,
    /// Device ordinals for the tp variant (comma separated, default 0..tp).
    #[arg(long, value_delimiter = ',')]
    pub devices: Vec<usize>,
    /// Reference for accuracy and A/B: an OpenAI-compatible vLLM-style server URL (`http://host:port/v1`: a top-k golden
    /// is made from its prompt_logprobs, and the in-depth A/B runs against it) or an HF model directory (fp32 golden).
    #[arg(long)]
    pub reference: Option<String>,
    /// Model id to send to the reference server (default: the first id of its /v1/models).
    #[arg(long)]
    pub reference_model: Option<String>,
    /// Golden file for the quick accuracy stage (default: found by model name in the golden directories, else made).
    #[arg(long)]
    pub golden: Option<PathBuf>,
    /// Golden file with the longer prompt set for the in-depth stages (default: found by name, else made).
    #[arg(long)]
    pub golden_long: Option<PathBuf>,
    /// Golden file with ~4k-token prompts for the `acc-4k` stage (default: `<model>-4k...` found by name, built from
    /// tools/prompts/golden_4k.jsonl; none is made). Its prompts are long enough for the prefix-reuse gate at a
    /// prefill chunk of 2048, which the long golden's 700-token prompts are not.
    #[arg(long)]
    pub golden_4k: Option<PathBuf>,
    /// Format-reference golden for the `acc-format` stage: the same quantized weights run by the format's own reference
    /// implementation (EXL3: tools/golden_exl3.py, exllamav3), metadata `reference_kind = format`. Gates the engine's
    /// error alone, at the activation-precision class. Default: `<model>-formatref...` found by name; none is made.
    #[arg(long)]
    pub format_golden: Option<PathBuf>,
    /// f64 CPU-oracle golden for the `acc-oracle` stage: `intarsia accuracy --backend cpu --golden <short golden> --mode
    /// prefill --write-self-golden FILE` (f32 accumulation, NVFP4 / MoE / GDN in f64, the same weights). Default:
    /// `oracle/<model>-cpu-oracle...` in the golden directories, else made with the cpu backend for a MoE / hybrid /
    /// quantized model (about 10 min for the 35B on an M5 Max; docs/results/2026-10-22-fx1-oracle-gate.md).
    #[arg(long)]
    pub oracle_golden: Option<PathBuf>,
    /// `acc-oracle`: mean 21-way KL(oracle || engine) on scored prefill rows must stay at or below this (default: the
    /// floor measured in docs/results/2026-10-22-fx1-oracle-gate.md, [`ORACLE_MAX_KL`]).
    #[arg(long)]
    pub oracle_max_kl: Option<f64>,
    /// Directory to search for goldens (repeatable; default: ./golden, /models/golden, $INTARSIA_GOLDEN_DIR, the repo's
    /// golden/, and the main checkout's golden/ when run from a git worktree).
    #[arg(long)]
    pub golden_dir: Vec<PathBuf>,
    /// Do not make a golden from the model itself with tools/golden.py above this many GB of weights.
    #[arg(long, default_value_t = 5.0)]
    pub make_golden_max_gb: f64,
    /// Report directory (default: reports/<model>/<timestamp>/).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Run the quick section only.
    #[arg(long)]
    pub quick_only: bool,
    /// Continue an earlier run in --out (or the newest reports/<model>/ directory): stages that finished (PASS or FAIL)
    /// are kept, SKIP and ERROR stages run again (e.g. after supplying the missing golden).
    #[arg(long)]
    pub resume: bool,
    /// Do not start the in-depth section when a quick gate failed.
    #[arg(long)]
    pub fail_fast: bool,
    /// Push the headline numbers to this Prometheus Pushgateway (`http://host:9091`) after the quick and the full section.
    #[arg(long)]
    pub push_metrics: Option<String>,
    /// Shell command run (in the background) when the quick section is done, with the quick-report.md path as `$1` and
    /// in `$INTARSIA_QUICK_REPORT` (also `$INTARSIA_QUICK_JSON`, `$INTARSIA_REPORT_DIR`), e.g. to publish it while the in-depth stages run.
    #[arg(long)]
    pub on_quick_done: Option<String>,
    /// Recipe file instead of the matched one.
    #[arg(long)]
    pub recipe: Option<PathBuf>,
    /// Extra directory with recipe *.toml files.
    #[arg(long)]
    pub recipe_dir: Vec<PathBuf>,
    /// Jinja chat template file for every stage (the bench speaks chat completions: a template that ignores the
    /// user message, as reranker templates do, cannot be benched without one).
    #[arg(long)]
    pub chat_template_file: Option<PathBuf>,
    /// Total memory a model-loading process may use, GiB (a fleetq `--mem` lease).
    #[arg(long)]
    pub mem_budget: Option<f64>,
    /// KV cache dtype override for every stage.
    #[arg(long)]
    pub kv_dtype: Option<String>,
    /// Name used in reports and the report path (default: derived from the model directory).
    #[arg(long)]
    pub model_name: Option<String>,
    /// Memory bandwidth for the byte floor, GB/s (default: a table by GPU arch, else doctor's measured device copy).
    #[arg(long)]
    pub bandwidth_gbs: Option<f64>,
    /// Weight GB read per decoded token for the byte floor (default: computed from the checkpoint).
    #[arg(long)]
    pub gb_per_token: Option<f64>,
    /// Tiny sizes and one repetition: a pipeline smoke test (use on cpu).
    #[arg(long)]
    pub smoke: bool,
    /// Prompt sizes of the quick bench (default 128,1024,4096).
    #[arg(long, value_delimiter = ',')]
    pub bench_sizes: Option<Vec<usize>>,
    /// Prompt sizes of the in-depth long-context bench (default 1024,4096,16384).
    #[arg(long, value_delimiter = ',')]
    pub long_sizes: Option<Vec<usize>>,
    /// Forced output length per bench call.
    #[arg(long = "gen")]
    pub gen: Option<usize>,
    /// Measured repetitions per bench cell (quick default 3, long-context default 2).
    #[arg(long)]
    pub reps: Option<usize>,
    /// First port to try for the model server of the bench stages (the next free one is used).
    #[arg(long, default_value_t = 8090)]
    pub port: u16,
    /// localeval checkout for the in-depth A/B (default /opt/code/deepeval; the stage is skipped when it or `uv` is missing).
    #[arg(long)]
    pub localeval_dir: Option<PathBuf>,
    /// Pass --no-unim to localeval (no unanimis record for the A/B runs).
    #[arg(long)]
    pub no_unim: bool,
    /// Skip a stage by id (repeatable): doctor, recipe, plan, kernels, gate, acc-format, acc-short, acc-oracle, bench, acc-long, acc-4k,
    /// envelope, bench-long, ab, tp.
    #[arg(long)]
    pub skip: Vec<String>,
    /// Stop starting new stages after this many minutes (exit 2; continue with --resume). For fleet lease windows.
    #[arg(long)]
    pub max_minutes: Option<f64>,
    /// Kill a single stage's subprocess after this many minutes.
    #[arg(long, default_value_t = 30)]
    pub stage_timeout_min: u64,
    /// The intarsia binary the stages run (default: this executable).
    #[arg(long)]
    pub bin: Option<PathBuf>,
}

/// What a stage returns.
pub struct Out {
    verdict: Verdict,
    reason: String,
    metrics: Value,
    artifacts: Vec<String>,
    facts: Map<String, Value>,
    notes: Vec<String>,
}

impl Out {
    fn new(verdict: Verdict, reason: impl Into<String>) -> Out {
        Out {
            verdict,
            reason: reason.into(),
            metrics: Value::Null,
            artifacts: vec![],
            facts: Map::new(),
            notes: vec![],
        }
    }
    fn pass(reason: impl Into<String>) -> Out {
        Out::new(Verdict::Pass, reason)
    }
    fn fail(reason: impl Into<String>) -> Out {
        Out::new(Verdict::Fail, reason)
    }
    fn skip(reason: impl Into<String>) -> Out {
        Out::new(Verdict::Skip, reason)
    }
    fn error(reason: impl Into<String>) -> Out {
        Out::new(Verdict::Error, reason)
    }
    fn metrics(mut self, m: Value) -> Out {
        self.metrics = m;
        self
    }
    fn art(mut self, a: impl Into<String>) -> Out {
        self.artifacts.push(a.into());
        self
    }
    fn fact(mut self, k: &str, v: impl Into<Value>) -> Out {
        self.facts.insert(k.to_string(), v.into());
        self
    }
    fn note(mut self, n: impl Into<String>) -> Out {
        self.notes.push(n.into());
        self
    }
}

type StageFn = fn(&mut Ctx) -> Result<Out>;

struct StageDef {
    id: &'static str,
    title: &'static str,
    section: &'static str,
    /// Stages that must not have failed (FAIL or ERROR) for this one to run.
    needs: &'static [&'static str],
    run: StageFn,
}

const STAGES: &[StageDef] = &[
    StageDef {
        id: "doctor",
        title: "doctor: host and device fingerprint, known pitfalls",
        section: "quick",
        needs: &[],
        run: st_doctor,
    },
    StageDef { id: "recipe", title: "recipe resolve", section: "quick", needs: &[], run: st_recipe },
    StageDef {
        id: "plan",
        title: "plan show: load the model, resolve a kernel per op",
        section: "quick",
        needs: &["recipe", "doctor"],
        run: st_plan,
    },
    StageDef {
        id: "kernels",
        title: "kernels test on the model's shapes",
        section: "quick",
        needs: &["doctor"],
        run: st_kernels,
    },
    StageDef {
        id: "gate",
        title: "quick gate: sanity prompts, perplexity, non-finite",
        section: "quick",
        needs: &["plan"],
        run: st_gate,
    },
    StageDef {
        id: "acc-format",
        title: "short accuracy against the format's own reference (same quantized weights)",
        section: "quick",
        needs: &["plan"],
        run: st_acc_format,
    },
    StageDef {
        id: "acc-short",
        title: "short accuracy against a golden",
        section: "quick",
        needs: &["plan"],
        run: st_acc_short,
    },
    StageDef {
        id: "acc-oracle",
        title: "short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference)",
        section: "quick",
        needs: &["plan"],
        run: st_acc_oracle,
    },
    StageDef {
        id: "bench",
        title: "bench speed 128 / 1024 / 4096 with the byte floor",
        section: "quick",
        needs: &["plan"],
        run: st_bench,
    },
    StageDef {
        id: "acc-long",
        title: "accuracy --mode all on the long golden (prefill, decode, prefix reuse)",
        section: "full",
        needs: &["plan"],
        run: st_acc_long,
    },
    StageDef {
        id: "acc-4k",
        title: "accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048)",
        section: "full",
        needs: &["plan"],
        run: st_acc_4k,
    },
    StageDef {
        id: "envelope",
        title: "noise envelope and baseline for regression gating",
        section: "full",
        needs: &["plan"],
        run: st_envelope,
    },
    StageDef {
        id: "bench-long",
        title: "long-context bench up to 16k",
        section: "full",
        needs: &["plan"],
        run: st_bench_long,
    },
    StageDef {
        id: "ab",
        title: "A/B against the reference server via localeval",
        section: "full",
        needs: &["plan"],
        run: st_ab,
    },
    StageDef { id: "tp", title: "tensor-parallel variant", section: "full", needs: &["plan"], run: st_tp },
];

/// Everything the stages share.
pub struct Ctx {
    a: QualifyArgs,
    out: PathBuf,
    name: String,
    bin: PathBuf,
    state: State,
    facts: Map<String, Value>,
    /// Command lines run by the current stage (provenance).
    cmds: Vec<String>,
    t0: Instant,
    /// Wall time of the sections measured by this process (resumed runs add the recorded stage times).
    hook_children: Vec<(String, std::process::Child)>,
}

enum Flow {
    Continue,
    Stop,
}

pub fn run(args: QualifyArgs) -> Result<()> {
    match pipeline(args) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("qualify: the pipeline itself failed: {e:#}");
            std::process::exit(2);
        }
    }
}

fn default_golden_dirs(a: &QualifyArgs) -> Vec<PathBuf> {
    let mut d = a.golden_dir.clone();
    if d.is_empty() {
        d.push(PathBuf::from("golden"));
        d.push(PathBuf::from("/models/golden"));
        if let Ok(p) = std::env::var("INTARSIA_GOLDEN_DIR") {
            d.push(PathBuf::from(p));
        }
        if let Some(r) = facts::repo_root() {
            d.push(r.join("golden"));
            // golden/ is untracked: a git worktree does not have it, its main checkout does
            if let Some(m) = facts::main_checkout(&r) {
                d.push(m.join("golden"));
            }
        }
    }
    d
}

fn pipeline(mut a: QualifyArgs) -> Result<i32> {
    proc::forward_termination();
    if a.tp == 0 {
        bail!("--tp must be at least 1");
    }
    if !["cpu", "cuda", "hip", "metal"].contains(&a.backend.as_str()) {
        bail!("unknown backend '{}' (cpu, cuda, hip, metal)", a.backend);
    }
    for s in &a.skip {
        if !STAGES.iter().any(|d| d.id == s) {
            bail!("--skip {s}: unknown stage (stages: {})", STAGES.iter().map(|d| d.id).collect::<Vec<_>>().join(", "));
        }
    }
    if !a.model.join("config.json").exists() {
        bail!("{}: no config.json (is this a Hugging Face model directory?)", a.model.display());
    }
    a.model = std::fs::canonicalize(&a.model).unwrap_or(a.model.clone());
    let name = facts::model_name(&a.model, a.model_name.as_deref());
    let out = match (&a.out, a.resume) {
        (Some(o), _) => o.clone(),
        (None, true) => newest_report_dir(&name)
            .ok_or_else(|| anyhow!("--resume: no reports/{name}/ directory to continue; give --out"))?,
        (None, false) => {
            let stamp = chrono_stamp();
            PathBuf::from("reports").join(&name).join(stamp)
        }
    };
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let key = json!({"model": a.model.display().to_string(), "backend": a.backend, "device": a.device, "reference": a.reference});
    let state = match State::load(&out)? {
        Some(s) if a.resume => {
            if s.key != key {
                bail!("{} was made for {} and cannot be resumed for {}", out.display(), s.key, key);
            }
            s
        }
        Some(_) => bail!("{} already holds a qualify run; use --resume to continue it or another --out", out.display()),
        None => {
            if a.resume {
                eprintln!("[qualify] --resume: {} has no state yet, starting fresh", out.display());
            }
            State::new(key)
        }
    };
    let bin = match &a.bin {
        Some(b) => b.clone(),
        None => std::env::current_exe().context("finding this executable")?,
    };
    let mut c =
        Ctx { a, out, name, bin, state, facts: Map::new(), cmds: vec![], t0: Instant::now(), hook_children: vec![] };
    c.say(&format!("{} on {}, report directory {}", c.name, c.a.backend, c.out.display()));
    if BUILD_STAMP.is_none() {
        c.say("this build carries no INTARSIA_BUILD_STAMP: baselines it writes are marked unstamped");
    }

    let mut stopped = false;
    let mut quick_published = false;
    for def in STAGES {
        if def.section == "full" && !quick_published {
            quick_published = true;
            c.finish_quick()?;
            if c.a.quick_only {
                break;
            }
            if c.a.fail_fast
                && c.state
                    .stages
                    .iter()
                    .any(|s| s.section == "quick" && (s.verdict.is_failing() || s.verdict == Verdict::Error))
            {
                c.say("--fail-fast: a quick gate failed, the in-depth section is not started");
                break;
            }
        }
        match c.run_stage(def)? {
            Flow::Continue => {}
            Flow::Stop => {
                stopped = true;
                break;
            }
        }
    }
    if !quick_published && !stopped {
        c.finish_quick()?;
    }
    let complete = !stopped;
    c.finish_full(complete)?;
    c.reap_hooks();
    c.state.save(&c.out)?;
    let code = c.exit_code(complete);
    c.say(&format!("exit code {code}"));
    Ok(code)
}

fn newest_report_dir(name: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(PathBuf::from("reports").join(name))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("state.json").exists())
        .collect();
    dirs.sort();
    dirs.pop()
}

/// `YYYYMMDD-HHMMSS` in UTC without a time crate.
fn chrono_stamp() -> String {
    stamp_at(now())
}

/// [`chrono_stamp`] of a unix time.
pub(crate) fn stamp_at(unix: u64) -> String {
    let t = unix as i64;
    let (days, secs) = (t.div_euclid(86400), t.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", secs / 3600, secs % 3600 / 60, secs % 60)
}

impl Ctx {
    fn say(&self, m: &str) {
        eprintln!("[qualify {:>6.0}s] {m}", self.t0.elapsed().as_secs_f64());
    }

    fn rel(&self, p: &Path) -> String {
        p.strip_prefix(&self.out).unwrap_or(p).display().to_string()
    }

    fn path(&self, sub: &str) -> PathBuf {
        self.out.join(sub)
    }

    fn fact_str(&self, k: &str) -> Option<String> {
        self.facts.get(k).and_then(|v| v.as_str()).map(String::from)
    }
    fn fact_f64(&self, k: &str) -> Option<f64> {
        self.facts.get(k).and_then(|v| v.as_f64())
    }

    // ---- running a stage ----

    fn run_stage(&mut self, def: &StageDef) -> Result<Flow> {
        if self.a.resume {
            if let Some(r) = self.state.finished(def.id).cloned() {
                self.say(&format!("{}: already done ({}), not run again", def.id, r.verdict.label()));
                for (k, v) in &r.facts {
                    self.facts.insert(k.clone(), v.clone());
                }
                return Ok(Flow::Continue);
            }
        }
        if self.a.skip.iter().any(|s| s == def.id) {
            let rec = self.record(def, Out::skip("skipped by --skip"), 0, 0.0);
            self.commit(rec)?;
            return Ok(Flow::Continue);
        }
        for need in def.needs {
            if let Some(r) = self.state.get(need) {
                if matches!(r.verdict, Verdict::Fail | Verdict::Error) {
                    let why = format!("blocked by '{need}' ({}): {}", r.verdict.label(), first_line(&r.reason));
                    self.say(&format!("{}: {why}", def.id));
                    let rec = self.record(def, Out::skip(why), 0, 0.0);
                    self.commit(rec)?;
                    return Ok(Flow::Continue);
                }
            }
        }
        if let Some(m) = self.a.max_minutes {
            if self.t0.elapsed().as_secs_f64() > m * 60.0 {
                self.say(&format!("--max-minutes {m} used up before '{}': stopping; continue with --resume", def.id));
                return Ok(Flow::Stop);
            }
        }
        self.say(&format!("== {} [{}]", def.id, def.title));
        self.cmds.clear();
        let started = now();
        let t = Instant::now();
        let out = match (def.run)(self) {
            Ok(o) => o,
            Err(e) => {
                let m = format!("{e:#}");
                match m.strip_prefix("SKIP: ") {
                    Some(why) => Out::skip(why.to_string()),
                    None => Out::error(m),
                }
            }
        };
        let wall = t.elapsed().as_secs_f64();
        let rec = self.record(def, out, started, wall);
        self.say(&format!(
            "{}: {} in {:.1}s{}",
            def.id,
            rec.verdict.label(),
            wall,
            if rec.reason.is_empty() { String::new() } else { format!(" ({})", first_line(&rec.reason)) }
        ));
        for (k, v) in &rec.facts {
            self.facts.insert(k.clone(), v.clone());
        }
        self.commit(rec)?;
        Ok(Flow::Continue)
    }

    fn record(&mut self, def: &StageDef, o: Out, started: u64, wall: f64) -> StageRecord {
        StageRecord {
            id: def.id.to_string(),
            title: def.title.to_string(),
            section: def.section.to_string(),
            verdict: o.verdict,
            reason: o.reason,
            started_unix: started,
            wall_s: wall,
            provenance: self.provenance(),
            artifacts: o.artifacts,
            metrics: o.metrics,
            facts: o.facts,
            notes: o.notes,
        }
    }

    fn commit(&mut self, rec: StageRecord) -> Result<()> {
        self.state.put(rec);
        self.state.save(&self.out)
    }

    fn provenance(&self) -> Value {
        let env: Map<String, Value> = std::env::vars()
            .filter(|(k, _)| {
                k.starts_with("INTARSIA_")
                    || k == "GPU_MAX_HW_QUEUES"
                    || k == "HIP_VISIBLE_DEVICES"
                    || k == "CUDA_VISIBLE_DEVICES"
                    || k == "FLEETQ_JOB"
            })
            .map(|(k, v)| (k, Value::String(v)))
            .collect();
        json!({
            "intarsia_version": env!("CARGO_PKG_VERSION"),
            "build_stamp": BUILD_STAMP.unwrap_or("unstamped"),
            "binary": self.bin.display().to_string(),
            "host": hostname(),
            "backend": self.a.backend,
            "device": self.fact_str("device_name"),
            "arch": self.fact_str("device_arch"),
            "model": self.a.model.display().to_string(),
            "model_fingerprint": facts::model_fingerprint(&self.a.model),
            "commands": self.cmds,
            "env": env,
        })
    }

    // ---- running things ----

    fn base_cmd(&self) -> Command {
        let mut c = Command::new(&self.bin);
        if self.a.backend == "hip" && std::env::var_os("GPU_MAX_HW_QUEUES").is_none() {
            c.env("GPU_MAX_HW_QUEUES", "1");
        }
        c
    }

    /// Run `intarsia <args>`; stdout and stderr land in `logs/<stem>.out` / `.err`.
    fn exec(&mut self, stem: &str, args: Vec<String>) -> Result<Run> {
        let mut cmd = self.base_cmd();
        cmd.args(&args);
        self.cmds.push(format!("{} {}", self.bin.display(), args.join(" ")));
        proc::run_captured(cmd, &self.path(&format!("logs/{stem}")), Duration::from_secs(self.a.stage_timeout_min * 60))
    }

    /// The model + backend flags shared by every model-loading command.
    fn load_args(&self, tp: usize) -> Vec<String> {
        let mut v = vec![
            "--model".into(),
            self.a.model.display().to_string(),
            "--backend".into(),
            self.a.backend.clone(),
            "--device".into(),
            self.a.device.to_string(),
        ];
        if tp > 1 {
            v.push("--tp".into());
            v.push(tp.to_string());
            if !self.a.devices.is_empty() {
                v.push("--devices".into());
                v.push(self.a.devices.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
            }
        }
        if let Some(r) = &self.a.recipe {
            v.push("--recipe".into());
            v.push(r.display().to_string());
        }
        for d in &self.a.recipe_dir {
            v.push("--recipe-dir".into());
            v.push(d.display().to_string());
        }
        if let Some(t) = &self.a.chat_template_file {
            v.push("--chat-template-file".into());
            v.push(t.display().to_string());
        }
        if let Some(m) = self.a.mem_budget {
            v.push("--mem-budget".into());
            v.push(m.to_string());
        }
        if let Some(k) = &self.a.kv_dtype {
            v.push("--kv-dtype".into());
            v.push(k.clone());
        }
        v
    }

    /// Provenance flags for baseline / envelope runs: an unstamped or dirty build is accepted (and said so in the
    /// report), because the baseline and the run that reads it are one binary within this run.
    fn stamp_flags(&self) -> Vec<String> {
        let mut v = vec![];
        match BUILD_STAMP {
            None => v.push("--allow-unstamped".to_string()),
            Some(s) if s.contains("dirty=yes") => v.push("--allow-dirty".to_string()),
            _ => {}
        }
        v
    }

    fn smoke(&self) -> bool {
        self.a.smoke
    }
    fn quick_sizes(&self) -> Vec<usize> {
        self.a.bench_sizes.clone().unwrap_or_else(|| if self.smoke() { vec![64, 256] } else { vec![128, 1024, 4096] })
    }
    fn long_sizes(&self) -> Vec<usize> {
        self.a
            .long_sizes
            .clone()
            .unwrap_or_else(|| if self.smoke() { vec![256, 1024] } else { vec![1024, 4096, 16384] })
    }
    fn gen_tokens(&self, long: bool) -> usize {
        self.a.gen.unwrap_or(if self.smoke() {
            16
        } else if long {
            64
        } else {
            128
        })
    }
    fn reps(&self, long: bool) -> usize {
        self.a.reps.unwrap_or(if self.smoke() {
            1
        } else if long {
            2
        } else {
            3
        })
    }
    fn decode_positions(&self, long: bool) -> usize {
        if self.smoke() {
            4
        } else if long {
            64
        } else {
            16
        }
    }
    fn acc_max_seq(&self, long: bool) -> usize {
        if long && !self.smoke() {
            8192
        } else if self.smoke() {
            1024
        } else {
            4096
        }
    }

    // ---- sections ----

    fn stages_of(&self, section: &str) -> Vec<&StageRecord> {
        STAGES.iter().filter(|d| d.section == section).filter_map(|d| self.state.get(d.id)).collect()
    }

    fn finish_quick(&mut self) -> Result<()> {
        let first_time = self.state.quick_report.is_none();
        let md = report::render(&self.state, &self.report_meta(), false);
        let json_v = report::summary_json(&self.state, &self.report_meta(), false);
        let md_path = self.path("quick-report.md");
        std::fs::write(&md_path, &md)?;
        std::fs::write(self.path("quick.json"), serde_json::to_string_pretty(&json_v)? + "\n")?;
        self.state.quick_report = Some("quick-report.md".to_string());
        self.state.save(&self.out)?;
        let abs = std::fs::canonicalize(&md_path).unwrap_or(md_path.clone());
        println!("QUICK DONE: {}", abs.display());
        if first_time {
            self.publish(&json_v, &abs, "quick")?;
        }
        Ok(())
    }

    fn finish_full(&mut self, complete: bool) -> Result<()> {
        if self.a.quick_only && self.state.quick_report.is_some() {
            return Ok(());
        }
        if self.stages_of("full").is_empty() && !complete {
            return Ok(());
        }
        let md = report::render(&self.state, &self.report_meta(), true);
        let json_v = report::summary_json(&self.state, &self.report_meta(), true);
        let md_path = self.path("full-report.md");
        std::fs::write(&md_path, &md)?;
        std::fs::write(self.path("full.json"), serde_json::to_string_pretty(&json_v)? + "\n")?;
        let abs = std::fs::canonicalize(&md_path).unwrap_or(md_path);
        println!("FULL DONE: {}", abs.display());
        if let Some(url) = self.a.push_metrics.clone() {
            self.push(&url, &json_v, "full");
        }
        Ok(())
    }

    fn report_meta(&self) -> report::Meta {
        report::Meta {
            model: self.name.clone(),
            model_dir: self.a.model.display().to_string(),
            backend: self.a.backend.clone(),
            tp: self.a.tp,
            reference: self.a.reference.clone(),
            build_stamp: BUILD_STAMP.map(String::from),
            version: env!("CARGO_PKG_VERSION").to_string(),
            generated_unix: now(),
            quick_only: self.a.quick_only,
        }
    }

    /// The publishing hooks of the quick section: the metrics push and `--on-quick-done`.
    fn publish(&mut self, summary: &Value, report_path: &Path, section: &str) -> Result<()> {
        if let Some(url) = self.a.push_metrics.clone() {
            self.push(&url, summary, section);
        }
        if let Some(cmd) = self.a.on_quick_done.clone() {
            let json_path = self.path("quick.json");
            let mut c = Command::new("sh");
            c.arg("-c")
                .arg(&cmd)
                .arg("qualify")
                .arg(report_path)
                .env("INTARSIA_QUICK_REPORT", report_path)
                .env("INTARSIA_QUICK_JSON", std::fs::canonicalize(&json_path).unwrap_or(json_path))
                .env("INTARSIA_REPORT_DIR", std::fs::canonicalize(&self.out).unwrap_or(self.out.clone()));
            match c.spawn() {
                Ok(ch) => {
                    self.say(&format!("--on-quick-done started: {cmd}"));
                    self.hook_children.push((cmd, ch));
                }
                Err(e) => {
                    self.say(&format!("WARNING: --on-quick-done could not start: {e}"));
                    self.state.hooks.push(
                        json!({"hook": "on-quick-done", "command": cmd, "started": false, "error": e.to_string()}),
                    );
                }
            }
        }
        Ok(())
    }

    fn push(&mut self, url: &str, summary: &Value, section: &str) {
        let (model, backend) = (self.name.clone(), self.a.backend.clone());
        let body = report::prometheus_text(summary);
        let target = format!(
            "{}/metrics/job/intarsia_qualify/model/{}/backend/{}",
            url.trim_end_matches('/'),
            facts::sanitize(&model),
            facts::sanitize(&backend)
        );
        let agent = intarsia_bench::http::agent(Some(Duration::from_secs(20)));
        let res = agent.put(&target).header("Content-Type", "text/plain; version=0.0.4").send(body.as_str());
        let (ok, detail) = match res {
            Ok(r) if r.status().as_u16() < 300 => (true, format!("HTTP {}", r.status().as_u16())),
            Ok(r) => (false, format!("HTTP {}", r.status().as_u16())),
            Err(e) => (false, e.to_string()),
        };
        self.say(&format!("--push-metrics ({section}): {} {detail}", if ok { "pushed" } else { "WARNING: failed" }));
        self.state
            .hooks
            .push(json!({"hook": "push-metrics", "section": section, "url": target, "ok": ok, "detail": detail}));
    }

    /// Wait (bounded) for the `--on-quick-done` command and record how it ended.
    fn reap_hooks(&mut self) {
        let kids = std::mem::take(&mut self.hook_children);
        for (cmd, mut ch) in kids {
            let t = Instant::now();
            let status = loop {
                match ch.try_wait() {
                    Ok(Some(s)) => break Some(s.code()),
                    Ok(None) if t.elapsed() > Duration::from_secs(120) => {
                        let _ = ch.kill();
                        let _ = ch.wait();
                        break None;
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(200)),
                    Err(_) => break None,
                }
            };
            let ok = status == Some(Some(0));
            if !ok {
                self.say(&format!(
                    "WARNING: --on-quick-done did not exit 0 ({status:?}); this does not change the exit code"
                ));
            }
            self.state.hooks.push(
                json!({"hook": "on-quick-done", "command": cmd, "started": true, "exit": status.flatten(), "ok": ok}),
            );
        }
    }

    fn exit_code(&self, complete: bool) -> i32 {
        let relevant: Vec<&StageRecord> = self.state.stages.iter().collect();
        if !complete || relevant.iter().any(|s| s.verdict == Verdict::Error) {
            return 2;
        }
        if relevant.iter().any(|s| s.verdict.is_failing()) {
            return 1;
        }
        0
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

fn hostname() -> String {
    // inside deploy/tatooine/dev.sh the container's hostname is a hash; dev.sh passes the box's name
    if let Ok(h) = std::env::var("INTARSIA_HOSTNAME") {
        if !h.is_empty() {
            return h;
        }
    }
    Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".into())
}

fn read_json(p: &Path) -> Result<Value> {
    let t = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    serde_json::from_str(&t).with_context(|| format!("{} is not JSON", p.display()))
}

fn f(v: &Value) -> Option<f64> {
    v.as_f64()
}

// =====================================================================================================================
// QUICK stages
// =====================================================================================================================

fn st_doctor(c: &mut Ctx) -> Result<Out> {
    let p = c.path("doctor.json");
    let args = vec![
        "doctor".into(),
        "--backend".into(),
        c.a.backend.clone(),
        "--device".into(),
        c.a.device.to_string(),
        "--out".into(),
        p.display().to_string(),
    ];
    let r = c.exec("doctor", args)?;
    if !r.ok() {
        return Ok(Out::error(format!("doctor {}: {}", r.describe(), r.tail(5))));
    }
    let j = read_json(&p)?;
    let checks = j["checks"].as_array().cloned().unwrap_or_default();
    let line = |c: &Value| format!("{}: {}", c["id"].as_str().unwrap_or("?"), c["detail"].as_str().unwrap_or(""));
    let fails: Vec<String> = checks.iter().filter(|c| c["status"] == "fail").map(line).collect();
    let warns: Vec<String> = checks.iter().filter(|c| c["status"] == "warn").map(line).collect();
    let dev = &j["backend"]["device"];
    let mut o = if let Some(e) = j["backend"]["error"].as_str() {
        Out::fail(format!("backend '{}' is unavailable: {e}", c.a.backend))
    } else if !fails.is_empty() {
        Out::fail(fails.join("; "))
    } else {
        Out::pass(format!(
            "{} ({}), {} kernels registered",
            dev["name"].as_str().unwrap_or("?"),
            dev["arch"].as_str().unwrap_or("?"),
            j["backend"]["kernel_count"]
        ))
    };
    o = o
        .art(c.rel(&p))
        .fact("device_name", dev["name"].clone())
        .fact("device_arch", dev["arch"].clone())
        .fact("device_free_gb", dev["free_mem"].as_f64().map(|b| b / 1e9).unwrap_or(0.0))
        .fact("copy_gbs", j["backend"]["bandwidth"]["device_copy_gbs"].clone())
        .metrics(json!({"host": j["host"], "device": dev, "bandwidth": j["backend"]["bandwidth"], "failures": fails, "warnings": warns}));
    for w in warns {
        o = o.note(format!("warning {w}"));
    }
    Ok(o)
}

fn st_recipe(c: &mut Ctx) -> Result<Out> {
    let cfg = read_json(&c.a.model.join("config.json"))?;
    let wf = facts::weight_facts(&c.a.model, &cfg).map_err(|e| anyhow!("reading the checkpoint tensors: {e:#}"))?;
    let tc = intarsia_core::recipe::text_config(&cfg);
    let archs = cfg["architectures"].clone();
    let mut o;
    if let Some(p) = &c.a.recipe {
        let r = intarsia_models::recipe::load_recipe_file(p).map_err(|e| anyhow!("{e}"))?;
        o = Out::pass(format!("explicit recipe {} (family {})", p.display(), r.model.family))
            .fact("recipe", p.display().to_string())
            .fact("family", r.model.family.clone());
    } else {
        let an = super::recipe::analyse(&c.a.model, &c.a.recipe_dir)?;
        c.cmds.push(format!("recipe::analyse({})", c.a.model.display()));
        let metrics = json!({"architectures": archs, "templates_resolved": [an.tensor_templates_resolved.0, an.tensor_templates_resolved.1], "quant": an.quant, "notes": an.notes, "todos": an.todos});
        o = match &an.covered_by {
            Some(src) => Out::pass(format!("{src} covers this model ({}/{} tensor templates resolve, quant {})", an.tensor_templates_resolved.0, an.tensor_templates_resolved.1, an.quant))
                .fact("recipe", src.clone())
                .fact("family", an.family.clone()),
            None => Out::fail(format!(
                "no existing recipe covers this model ({} item(s) need manual work: {}); run `intarsia recipe new --model {}`",
                an.todos.len(),
                an.todos.iter().take(4).cloned().collect::<Vec<_>>().join("; "),
                c.a.model.display()
            )),
        };
        o = o.metrics(metrics).fact("quant", an.quant.clone());
    }
    o = o
        .fact("weights_gb", wf.total_bytes as f64 / 1e9)
        .fact("active_gb", wf.active_bytes as f64 / 1e9)
        .fact("moe", wf.moe)
        .fact("hybrid", wf.hybrid)
        .fact("quantized", wf.quantized)
        .fact("n_layers", tc["num_hidden_layers"].clone());
    Ok(o)
}

fn st_plan(c: &mut Ctx) -> Result<Out> {
    let p = c.path("plan.json");
    let mut args = vec!["plan".to_string(), "show".into()];
    args.extend(c.load_args(1));
    args.push("--admit".into());
    args.push("--json".into());
    let r = c.exec("plan", args)?;
    if r.timed_out {
        return Ok(Out::error(format!("plan show timed out after {:.0}s", r.wall_s)));
    }
    if !r.ok() {
        let text = r.stderr();
        let missing = missing_kernel_lines(&text);
        if r.code.is_none() {
            return Ok(Out::error(format!("plan show was killed by a signal: {}", r.tail(4))));
        }
        let mut o = if text.contains("have no kernel on") {
            // a load error that names the ops without a kernel: add what this backend does register
            let list = c.exec(
                "kernels-list",
                vec![
                    "kernels".into(),
                    "list".into(),
                    "--backend".into(),
                    c.a.backend.clone(),
                    "--device".into(),
                    c.a.device.to_string(),
                ],
            )?;
            let mut o = Out::fail(format!("missing kernels on {}: {}", c.a.backend, missing.join(" | ")));
            if list.ok() {
                o = o
                    .art(c.rel(&list.out))
                    .note("the kernels this backend does register are listed in the kernels-list artifact");
            }
            o
        } else {
            Out::fail(format!("the model does not load on {}: {}", c.a.backend, r.tail(6)))
        };
        o = o.art(c.rel(&r.err)).metrics(json!({"missing_kernels": missing}));
        return Ok(o);
    }
    std::fs::copy(&r.out, &p)?;
    let j = read_json(&p)?;
    let ops = j["ops"].as_array().cloned().unwrap_or_default();
    let unadmitted: Vec<String> = ops
        .iter()
        .filter(|o| o["admitted"] != true)
        .map(|o| format!("{}:{}", o["op"].as_str().unwrap_or("?"), o["kernel"].as_str().unwrap_or("?")))
        .collect();
    let layers = j["layers"].as_array().cloned().unwrap_or_default();
    let mut kinds: Vec<String> = vec![];
    for l in &layers {
        let k = l["blocks"]
            .as_array()
            .map(|b| b.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join("+"))
            .unwrap_or_default();
        if kinds.last() != Some(&k) {
            kinds.push(k);
        }
    }
    let m = json!({
        "model": j["model"], "device": j["device"]["name"], "exactness": j["exactness"], "act_dtype": j["act_dtype"], "kv_dtype": j["kv_dtype"],
        "prefill_chunk": j["prefill_chunk"], "prefill_chunk_source": j["prefill_chunk_source"], "scratch_bytes": j["scratch_bytes"], "reference_only": j["reference_only"],
        "layers": layers.len(), "layer_blocks": kinds, "ops": ops.iter().map(|o| json!({"op": o["op"], "kernel": o["kernel"], "exactness": o["exactness"], "admitted": o["admitted"], "calls_per_decode_step": o["calls_per_decode_step"]})).collect::<Vec<_>>(),
        "load_s": r.wall_s,
    });
    let o = if j["reference_only"] == true && c.a.backend != "cpu" {
        Out::fail(format!(
            "only reference kernels would run on {}: this backend registers none for the model's ops",
            c.a.backend
        ))
    } else if !unadmitted.is_empty() && c.a.backend != "cpu" {
        Out::fail(format!(
            "{} op(s) selected a kernel that did not pass admission: {}",
            unadmitted.len(),
            unadmitted.join(", ")
        ))
    } else {
        Out::pass(format!(
            "{} ops resolved, exactness {}, loaded in {:.1}s",
            ops.len(),
            j["exactness"].as_str().unwrap_or("?"),
            r.wall_s
        ))
    };
    Ok(o.art(c.rel(&p)).metrics(m))
}

/// The "- op: N shape(s) without a kernel, ..." lines of the load error `plan.rs` raises (`have no kernel on`).
fn missing_kernel_lines(stderr: &str) -> Vec<String> {
    stderr.lines().filter_map(|l| l.trim().strip_prefix("- ")).map(|l| l.to_string()).collect()
}

fn st_kernels(c: &mut Ctx) -> Result<Out> {
    if c.a.backend == "cpu" {
        return Ok(Out::skip(
            "cpu-ref is the oracle every other backend is admitted against; admission has nothing to compare it with",
        ));
    }
    let mut args = vec![
        "kernels".to_string(),
        "test".into(),
        "--backend".into(),
        c.a.backend.clone(),
        "--device".into(),
        c.a.device.to_string(),
        "--model".into(),
        c.a.model.display().to_string(),
    ];
    if c.smoke() {
        args.push("--quick".into());
    }
    let r = c.exec("kernels", args)?;
    let text = r.stdout();
    let summary =
        text.lines().rev().find(|l| l.contains(" kernels, ") && l.contains(" cases in ")).unwrap_or("").to_string();
    if r.timed_out || r.code.is_none() {
        return Ok(Out::error(format!("kernels test {}", r.describe())));
    }
    let mut o = if r.ok() {
        // kernels with no case in the sweep do not fail `kernels test`; keep them visible in the pass line
        let incomplete = text.lines().find(|l| l.starts_with("incomplete: ")).map(|l| format!("; {l}")).unwrap_or_default();
        Out::pass(if summary.is_empty() {
            format!("all kernels admitted{incomplete}")
        } else {
            format!("{summary}{incomplete}")
        })
    } else {
        let err = r.stderr();
        if err.contains("kernel admission failed") || err.contains("nothing was tested") {
            let bad: Vec<String> =
                text.lines().filter(|l| l.contains("FAIL")).take(6).map(|l| l.trim().to_string()).collect();
            Out::fail(format!(
                "{} {}",
                first_line(
                    &err.lines()
                        .rev()
                        .find(|l| l.starts_with("Error:"))
                        .unwrap_or("kernel admission failed")
                        .to_string()
                ),
                bad.join(" | ")
            ))
        } else {
            Out::error(format!("kernels test {}: {}", r.describe(), r.tail(5)))
        }
    };
    o = o.art(c.rel(&r.out)).metrics(json!({"summary": summary}));
    Ok(o)
}

fn line_value<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.lines().find_map(|l| l.trim_start().strip_prefix(prefix))
}

fn st_gate(c: &mut Ctx) -> Result<Out> {
    let mut args = vec!["accuracy".to_string()];
    args.extend(c.load_args(1));
    args.extend(["--quick".into(), "--max-seq".into(), "2048".into()]);
    let r = c.exec("gate", args)?;
    let text = r.stdout();
    if r.timed_out || r.code.is_none() {
        return Ok(Out::error(format!("quick gate {}", r.describe())));
    }
    let ppl =
        line_value(&text, "perplexity:").and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse::<f64>().ok());
    let sane_ok = text.lines().filter(|l| l.starts_with("sanity ok")).count();
    let sane_bad = text.lines().filter(|l| l.starts_with("sanity FAIL")).count();
    let m = json!({"perplexity": ppl, "perplexity_line": line_value(&text, "perplexity:").map(|s| s.trim()), "sanity_ok": sane_ok, "sanity_failed": sane_bad});
    let o = if r.ok() && text.contains("quick gate: PASS") {
        Out::pass(format!(
            "perplexity {}, {sane_ok}/{} sanity prompts ok, no non-finite rows",
            ppl.map(|p| format!("{p:.2}")).unwrap_or("?".into()),
            sane_ok + sane_bad
        ))
    } else if text.contains("quick gate: FAIL") {
        let bad: Vec<String> = text
            .lines()
            .filter(|l| l.contains("FAIL") && !l.starts_with("quick gate:"))
            .take(5)
            .map(|l| l.trim().chars().take(200).collect())
            .collect();
        Out::fail(bad.join(" | "))
    } else {
        Out::error(format!("quick gate {}: {}", r.describe(), r.tail(5)))
    };
    Ok(o.art(c.rel(&r.out)).metrics(m))
}

// ---- goldens ----

/// Outcome of looking for / making a golden: the file and how it came to be, or the reason there is none.
enum Golden {
    Have(PathBuf, String),
    None(String),
}

fn have_cmd(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

impl Ctx {
    fn golden(&mut self, long: bool) -> Result<Golden> {
        let key = if long { "golden_long" } else { "golden_short" };
        if let Some(p) = self.fact_str(key) {
            if Path::new(&p).exists() {
                return Ok(Golden::Have(PathBuf::from(p), self.fact_str(&format!("{key}_how")).unwrap_or_default()));
            }
        }
        let given = if long { self.a.golden_long.clone() } else { self.a.golden.clone() };
        if let Some(g) = given {
            if !g.exists() {
                return Ok(Golden::None(format!(
                    "--golden{} {} does not exist",
                    if long { "-long" } else { "" },
                    g.display()
                )));
            }
            return Ok(self.have_golden(key, g, "given with --golden".into()));
        }
        let made = self.path(&format!("goldens/{}-{}.safetensors", self.name, if long { "long" } else { "short" }));
        if made.exists() {
            return Ok(self.have_golden(key, made, "made earlier in this report directory".into()));
        }
        let dirs = default_golden_dirs(&self.a);
        if let Some(g) = facts::find_golden(&self.name, &dirs, long) {
            return Ok(self.have_golden(key, g, "found by model name".into()));
        }
        Ok(match self.make_golden(long)? {
            Golden::None(why) => Golden::None(format!(
                "{why}; no {} golden for '{}' in {} (give --golden{} FILE or --golden-dir DIR, then --resume)",
                if long { "long" } else { "short" },
                self.name,
                dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "),
                if long { "-long" } else { "" },
            )),
            g => g,
        })
    }

    /// The ~4k-token golden, if there is one: `--golden-4k`, or `<model>-4k...` in the golden directories. Never made
    /// here (the HF fp32 run over its 12.6k positions takes much longer than a stage).
    fn golden_4k(&mut self) -> Result<Golden> {
        let key = "golden_4k";
        if let Some(p) = self.fact_str(key) {
            if Path::new(&p).exists() {
                return Ok(Golden::Have(PathBuf::from(p), self.fact_str(&format!("{key}_how")).unwrap_or_default()));
            }
        }
        if let Some(g) = self.a.golden_4k.clone() {
            if !g.exists() {
                return Ok(Golden::None(format!("--golden-4k {} does not exist", g.display())));
            }
            return Ok(self.have_golden(key, g, "given with --golden-4k".into()));
        }
        let dirs = default_golden_dirs(&self.a);
        if let Some(g) = facts::find_golden_4k(&self.name, &dirs) {
            return Ok(self.have_golden(key, g, "found by model name".into()));
        }
        Ok(Golden::None(format!(
            "no 4k golden for '{}' in {} (give --golden-4k FILE or --golden-dir DIR; build one with tools/golden.py --prompts tools/prompts/golden_4k.jsonl)",
            self.name,
            dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "),
        )))
    }

    fn have_golden(&mut self, key: &str, p: PathBuf, how: String) -> Golden {
        self.facts.insert(key.to_string(), json!(p.display().to_string()));
        self.facts.insert(format!("{key}_how"), json!(how));
        Golden::Have(p, how)
    }

    fn make_golden(&mut self, long: bool) -> Result<Golden> {
        let Some(repo) = facts::repo_root() else {
            return Ok(Golden::None(
                "no golden found and the repository tools/ are not reachable to make one (set INTARSIA_REPO)".into(),
            ));
        };
        let key = if long { "golden_long" } else { "golden_short" };
        let out = self.path(&format!("goldens/{}-{}.safetensors", self.name, if long { "long" } else { "short" }));
        let python: Vec<String> = if have_cmd("uv") {
            vec![
                "uv".into(),
                "run".into(),
                "--project".into(),
                repo.join("tools").display().to_string(),
                "python".into(),
            ]
        } else {
            vec!["python3".into()]
        };
        let mut cmd = Command::new(&python[0]);
        cmd.args(&python[1..]);
        let mut desc;
        match self.a.reference.clone() {
            Some(r) if r.starts_with("http://") || r.starts_with("https://") => {
                let base = intarsia_bench::http::api_base(&r);
                let model = self.reference_model(&base)?;
                cmd.arg(repo.join("tools/golden_vllm.py"))
                    .args(["--base-url", &base, "--model", &model, "--out"])
                    .arg(&out);
                cmd.arg("--prompts").arg(repo.join("tools/prompts/golden.jsonl"));
                if long {
                    cmd.arg(repo.join("tools/prompts/golden_long.jsonl"));
                }
                desc = format!("made from the reference server {base} (top-20 prompt_logprobs)");
            }
            other => {
                let model_dir = other.map(PathBuf::from).unwrap_or(self.a.model.clone());
                let cfg = read_json(&model_dir.join("config.json"))?;
                let wf = facts::weight_facts(&model_dir, &cfg)?;
                if wf.moe || wf.hybrid || wf.quantized {
                    return Ok(Golden::None("no golden found, and a golden for a MoE / hybrid / quantized model is made from a reference server: pass --reference <vLLM url> or --golden FILE".into()));
                }
                let gb = wf.total_bytes as f64 / 1e9;
                if gb > self.a.make_golden_max_gb {
                    return Ok(Golden::None(format!("no golden found, and {gb:.1} GB of weights is too large to make an fp32 HF golden quickly (limit {} GB, --make-golden-max-gb): pass --golden FILE or --reference <vLLM url>", self.a.make_golden_max_gb)));
                }
                cmd.arg(repo.join("tools/golden.py"))
                    .args(["--model"])
                    .arg(&model_dir)
                    .args(["--dtype", "float32", "--device", "cpu", "--out"])
                    .arg(&out);
                cmd.arg("--prompts").arg(repo.join("tools/prompts/golden.jsonl"));
                if long {
                    cmd.arg(repo.join("tools/prompts/golden_long.jsonl"));
                }
                desc = format!("made with tools/golden.py from {} (HF transformers fp32 on cpu)", model_dir.display());
            }
        }
        if let Some(d) = out.parent() {
            std::fs::create_dir_all(d)?;
        }
        self.say(&format!("making the {} golden ({desc})", if long { "long" } else { "short" }));
        self.cmds.push(format!("{:?}", cmd));
        let r = proc::run_captured(
            cmd,
            &self.path(&format!("logs/make-golden-{}", if long { "long" } else { "short" })),
            Duration::from_secs(self.a.stage_timeout_min * 60),
        )?;
        if !r.ok() || !out.exists() {
            return Ok(Golden::None(format!(
                "no golden found and making one failed ({}): {}",
                r.describe(),
                r.tail(3)
            )));
        }
        desc.push_str(&format!(" in {:.0}s", r.wall_s));
        Ok(self.have_golden(key, out, desc))
    }

    fn reference_model(&self, base: &str) -> Result<String> {
        if let Some(m) = &self.a.reference_model {
            return Ok(m.clone());
        }
        let agent = intarsia_bench::http::agent(Some(Duration::from_secs(15)));
        let mut resp = agent
            .get(&format!("{base}/models"))
            .call()
            .map_err(|e| anyhow!("asking the reference server for /v1/models: {e}"))?;
        let text =
            resp.body_mut().read_to_string().map_err(|e| anyhow!("reading /v1/models of the reference server: {e}"))?;
        let v: Value =
            serde_json::from_str(&text).map_err(|e| anyhow!("/v1/models of the reference server is not JSON: {e}"))?;
        v["data"][0]["id"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| anyhow!("the reference server lists no model; pass --reference-model"))
    }
}

/// `accuracy --golden` on the loaded model; returns the verdict and the numbers read back from the baseline JSON the
/// run wrote (`modes[].overall`).
struct AccRun {
    verdict: Verdict,
    reason: String,
    modes: Vec<Value>,
    gate_class: Value,
    out: PathBuf,
    err: PathBuf,
    baseline: PathBuf,
    run_wall: f64,
}

impl Ctx {
    fn accuracy(
        &mut self,
        stem: &str,
        golden: &Path,
        mode: &str,
        long: bool,
        tp: usize,
        extra: &[String],
    ) -> Result<AccRun> {
        let baseline = self.path(&format!("baselines/{stem}.baseline.json"));
        if let Some(d) = baseline.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut args = vec!["accuracy".to_string()];
        args.extend(self.load_args(tp));
        args.extend([
            "--golden".into(),
            golden.display().to_string(),
            "--mode".into(),
            mode.into(),
            "--decode-positions".into(),
            self.decode_positions(long).to_string(),
            "--max-seq".into(),
            self.acc_max_seq(long).to_string(),
            "--write-baseline".into(),
            baseline.display().to_string(),
        ]);
        args.extend(extra.iter().cloned());
        let r = self.exec(stem, args)?;
        let text = r.stdout();
        let (verdict, reason) = if r.timed_out || r.code.is_none() {
            (Verdict::Error, format!("accuracy {}", r.describe()))
        } else if r.ok() && text.contains("golden gate: PASS") {
            (Verdict::Pass, String::new())
        } else if text.contains("golden gate: FAIL") {
            let bad = acc_fail_digest(&text);
            (Verdict::Fail, if bad.is_empty() { "the golden gate failed".to_string() } else { bad.join("; ") })
        } else {
            (Verdict::Error, format!("accuracy {}: {}", r.describe(), r.tail(5)))
        };
        let b = if baseline.exists() { read_json(&baseline).ok() } else { None };
        let modes = b
            .as_ref()
            .and_then(|b| b["modes"].as_array())
            .map(|a| a.iter().map(|m| m["overall"].clone()).collect())
            .unwrap_or_default();
        let gate_class = b.as_ref().map(|b| b["run"]["gate_class"].clone()).unwrap_or(Value::Null);
        Ok(AccRun {
            verdict,
            reason,
            modes,
            gate_class,
            out: r.out.clone(),
            err: r.err.clone(),
            baseline,
            run_wall: r.wall_s,
        })
    }

    fn acc_out(&self, a: AccRun, golden: &Path, how: &str, key_prefix: &str) -> Out {
        let m = json!({"golden": golden.display().to_string(), "golden_how": how, "golden_fingerprint": facts::file_fingerprint(golden), "gate_class": a.gate_class, "modes": a.modes, "run_s": a.run_wall});
        let mut o =
            Out::new(a.verdict, if a.verdict == Verdict::Pass { acc_summary(&a.modes) } else { a.reason.clone() })
                .metrics(m)
                .art(self.rel(&a.out));
        if a.baseline.exists() {
            o = o.art(self.rel(&a.baseline)).fact(&format!("{key_prefix}_baseline"), self.rel(&a.baseline));
        }
        let _ = a.err;
        o
    }
}

/// The failed checks of an `accuracy` run, one short phrase each ("prefill: top-1 99.08% (need >= 99.5)").
fn acc_fail_digest(text: &str) -> Vec<String> {
    let mut out = vec![];
    for l in text.lines() {
        if let Some((mode, rest)) = l.split_once(" floor [") {
            for seg in rest.split(" | ") {
                let seg = seg.trim();
                if seg.ends_with("=> FAIL") {
                    continue; // the verdict of the whole line, not a check
                }
                if let Some(i) = seg.find(" FAIL") {
                    let what = seg[..i].rsplit_once("]: ").map(|(_, w)| w).unwrap_or(&seg[..i]);
                    out.push(format!("{}: {}", mode.trim().split(" (").next().unwrap_or("").trim(), what.trim()));
                }
            }
        }
        if l.contains("baseline verdict") && l.contains(": FAIL") && !l.contains("not requested") {
            out.push("baseline gate failed".to_string());
        }
        if l.contains("prefix-reuse verdict: FAIL") {
            out.push("prefix reuse differs from a cold run".to_string());
        }
    }
    out.dedup();
    out
}

fn acc_summary(modes: &[Value]) -> String {
    modes
        .iter()
        .map(|m| {
            format!(
                "{} top-1 {:.2}% KL {:.2e}",
                m["mode"].as_str().unwrap_or("?"),
                m["top1_pct"].as_f64().unwrap_or(f64::NAN),
                m["mean_kl"].as_f64().unwrap_or(f64::NAN)
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn st_acc_short(c: &mut Ctx) -> Result<Out> {
    let (g, how) = match c.golden(false)? {
        Golden::Have(g, h) => (g, h),
        Golden::None(why) => return Ok(Out::skip(why)),
    };
    // A model whose golden is a reference server's (MoE / hybrid / quantized: no fp32 HF golden can be made) is gated
    // against the backend's recorded baseline and noise envelope; without one that applies it cannot PASS: the static
    // 4-bit floor alone sits 2x above where every build reads and lets a 1 % scale bug through (fx1,
    // docs/results/2026-10-22-fx1-oracle-gate.md). Other models keep the floor-only gate.
    let needs = c.needs_baseline(&g);
    let (extra, note) = match (c.recorded_baseline(&g, false), needs) {
        (Ok((args, note)), true) => (args, note),
        (Ok((_, note)), false) => (vec![], format!("{note}; not applied (the class floor of a full-precision golden gates this model)")),
        (Err(why), _) => (vec![], why),
    };
    let missing = needs && extra.is_empty();
    let a = c.accuracy("acc-short", &g, "both", false, 1, &extra)?;
    let o = c.acc_out(a, &g, &how, "short");
    let mut o = baseline_verdict(o, note, missing, &c.name, "acc-short");
    if o.verdict == Verdict::Fail {
        if let Some(f) = c.state.get("acc-format").filter(|f| f.verdict == Verdict::Pass) {
            o.verdict = Verdict::Warn;
            o.reason = format!(
                "report-only: {} (a third-party quant format; acc-format PASS against its own reference ({}), so this is the format's loss against bf16, not engine error; user decision 2026-10-21 17:13)",
                o.reason,
                f.reason
            );
        }
    }
    Ok(o)
}

/// The engine's error alone on a quantized model: `accuracy` against a golden made by the format's own reference
/// implementation from the same quantized weights (`reference_kind = format`, docs/accuracy.md "Format-reference
/// goldens"). `accuracy` gates it at the activation-precision class, and a `ref_noise` record of that golden (sidecar or
/// metadata) applies as on any golden. It runs before `acc-short`: when it passes, a failing `acc-short` (the bf16-relative
/// check: quant loss plus engine error) is report-only, a WARN (user decision 2026-10-21 17:13).
fn st_acc_format(c: &mut Ctx) -> Result<Out> {
    let (g, how) = match c.a.format_golden.clone() {
        Some(g) if !g.exists() => return Ok(Out::error(format!("--format-golden {} does not exist", g.display()))),
        Some(g) => (g, "given with --format-golden".to_string()),
        None => {
            if c.facts.get("quantized") != Some(&json!(true)) {
                return Ok(Out::skip("the weights are not quantized: no format reference applies"));
            }
            match facts::find_format_golden(&c.name, &default_golden_dirs(&c.a)) {
                Some(g) => (g, "found by model name".to_string()),
                None => {
                    return Ok(Out::skip(format!(
                        "no format-reference golden for '{}' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)",
                        c.name
                    )))
                }
            }
        }
    };
    if let Err(e) = std::fs::File::open(&g) {
        return Ok(Out::error(format!("{}: {e}", g.display())));
    }
    let kind = facts::golden_meta(&g, "reference_kind");
    if kind.as_deref() != Some("format") {
        return Ok(Out::error(format!(
            "{}: not a format-reference golden (metadata reference_kind = {}, want \"format\")",
            g.display(),
            kind.as_deref().unwrap_or("none")
        )));
    }
    let engine = facts::golden_meta(&g, "reference_engine").unwrap_or_else(|| "unrecorded".into());
    let how = format!("{how}; reference: {engine}");
    c.facts.insert("golden_format".into(), json!(g.display().to_string()));
    c.facts.insert("golden_format_how".into(), json!(how.clone()));
    let a = c.accuracy("acc-format", &g, "both", false, 1, &[])?;
    Ok(c.acc_out(a, &g, &how, "format"))
}

/// `acc-oracle`'s floor: mean KL(oracle || engine) over the oracle's top-64 + tail, scored prefill rows of the oracle
/// prompt set (the short set plus `long_docs_4k`). Measured, not chosen (docs/results/2026-10-22-fx1-oracle-gate.md):
/// clean builds read 9.9e-3 (Metal, HIP tp=1) to 1.1e-2 (HIP tp=2), bf16 activations against f32 / f64 math on the
/// same NVFP4 weights; 2e-2 is ~1.8x the largest. The floor guards absolute accuracy (a wrong kernel reads 10x and
/// more). It does NOT see a 1 % scale bug: those mutants read 9.1e-3 to 1.0e-2, inside the clean spread. The kill is
/// the drift check against the recorded same-backend dump.
pub const ORACLE_MAX_KL: f64 = 2.0e-2;

impl Ctx {
    /// The f64 CPU-oracle golden: `--oracle-golden`, `oracle/<model>-cpu-oracle...` in the golden directories (the repo
    /// commits them), one made earlier in this report directory, or, for a model with no full-precision golden, one made
    /// here with the cpu backend from the short golden's prompts.
    fn oracle_golden(&mut self) -> Result<Golden> {
        let key = "golden_oracle";
        if let Some(p) = self.fact_str(key) {
            if Path::new(&p).exists() {
                return Ok(Golden::Have(PathBuf::from(p), self.fact_str(&format!("{key}_how")).unwrap_or_default()));
            }
        }
        if let Some(g) = self.a.oracle_golden.clone() {
            if !g.exists() {
                return Ok(Golden::None(format!("--oracle-golden {} does not exist", g.display())));
            }
            return Ok(self.have_golden(key, g, "given with --oracle-golden".into()));
        }
        let dirs = default_golden_dirs(&self.a);
        if let Some(g) = facts::find_oracle_golden(&self.name, &dirs) {
            return Ok(self.have_golden(key, g, "found by model name".into()));
        }
        let made = self.path(&format!("goldens/{}-cpu-oracle-short.safetensors", self.name));
        if made.exists() {
            return Ok(self.have_golden(key, made, "made earlier in this report directory".into()));
        }
        let (short, _) = match self.golden(false)? {
            Golden::Have(g, h) => (g, h),
            Golden::None(why) => return Ok(Golden::None(format!("no oracle golden, and none can be made without the short golden's prompts: {why}"))),
        };
        // the same scope as acc-short's baseline requirement: a model whose golden is a full-precision reference needs no oracle
        if !self.needs_baseline(&short) {
            return Ok(Golden::None("the short golden is a full-precision reference of this model (or the model is dense, unquantized): no oracle needed".into()));
        }
        let mut args = vec!["accuracy".to_string(), "--model".into(), self.a.model.display().to_string(), "--backend".into(), "cpu".into()];
        if let Some(r) = &self.a.recipe {
            args.extend(["--recipe".into(), r.display().to_string()]);
        }
        for d in &self.a.recipe_dir {
            args.extend(["--recipe-dir".into(), d.display().to_string()]);
        }
        if let Some(m) = self.a.mem_budget {
            args.extend(["--mem-budget".into(), m.to_string()]);
        }
        args.extend([
            "--golden".into(),
            short.display().to_string(),
            "--mode".into(),
            "prefill".into(),
            "--max-seq".into(),
            self.acc_max_seq(false).to_string(),
            "--prefill-chunk".into(),
            "256".into(),
            "--write-self-golden".into(),
            made.display().to_string(),
            "--allow-unstamped".into(),
            "--allow-dirty".into(),
        ]);
        self.say(&format!("making the CPU oracle golden from {} (cpu backend, prefill rows, top-64)", short.display()));
        let r = self.exec("make-oracle", args)?;
        if !made.exists() {
            return Ok(Golden::None(format!("making the CPU oracle golden failed ({}): {}", r.describe(), r.tail(3))));
        }
        Ok(self.have_golden(key, made, format!("made here with the cpu backend from {} in {:.0}s", short.display(), r.wall_s)))
    }

    /// The `accuracy` flags of the oracle gate: the floor on scored rows, no top-1 floor (near-tie rows flip between
    /// any two correct implementations), no reference-noise term (the oracle is deterministic).
    fn oracle_flags(&self) -> (Vec<String>, f64) {
        let kl = self.a.oracle_max_kl.unwrap_or(ORACLE_MAX_KL);
        let mut v: Vec<String> = vec![
            "--gate-class".into(),
            "bf16".into(),
            "--max-mean-kl".into(),
            format!("{kl:e}"),
            "--min-top1".into(),
            "0".into(),
            "--no-ref-noise".into(),
            // the dump's identity: the same name on every box and for every copy of the weights
            "--model-id".into(),
            self.name.clone(),
        ];
        (v, kl)
    }

    /// `<oracle stem>.<backend>[-tpN]`: the name of this backend's (and tp degree's) recorded dump on the oracle prompts.
    fn dump_tag(&self, oracle: &Path, tp: usize) -> String {
        let stem = oracle.file_stem().and_then(|s| s.to_str()).unwrap_or("oracle");
        if tp > 1 {
            format!("{stem}.{}-tp{tp}", self.a.backend)
        } else {
            format!("{stem}.{}", self.a.backend)
        }
    }

    /// The drift check: this build's prefill logits on the oracle prompts against the dump recorded from clean main on
    /// the same backend and tp degree (`golden/baselines/<oracle stem>.<backend>[-tpN].selfgolden.safetensors`), with the
    /// self-golden budgets (mean KL21 1e-4 ...). The engine is deterministic, so clean main reads 0; any change of the
    /// logits fails it, a 1 % partial-scale bug included, which no comparison with another implementation can resolve
    /// on this model (fx1). `Err(why)`: there is no dump for this configuration.
    fn drift_check(&mut self, oracle: &Path, tp: usize, stem: &str) -> Result<std::result::Result<(Verdict, String), String>> {
        let tag = self.dump_tag(oracle, tp);
        let mut dirs: Vec<PathBuf> = vec![];
        if let Some(d) = oracle.parent().and_then(|d| d.parent()) {
            dirs.push(d.join("baselines"));
        }
        if let Some(r) = facts::repo_root() {
            dirs.push(r.join("golden/baselines"));
            if let Some(m) = facts::main_checkout(&r) {
                dirs.push(m.join("golden/baselines"));
            }
        }
        let Some(dump) = dirs.iter().map(|d| d.join(format!("{tag}.selfgolden.safetensors"))).find(|p| p.is_file()) else {
            return Ok(Err(format!("no recorded dump {tag}.selfgolden.safetensors in golden/baselines")));
        };
        let mut args = vec!["accuracy".to_string()];
        args.extend(self.load_args(tp));
        args.extend([
            "--self-golden".into(),
            dump.display().to_string(),
            "--mode".into(),
            "prefill".into(),
            "--decode-positions".into(),
            self.decode_positions(false).to_string(),
            "--max-seq".into(),
            self.acc_max_seq(false).to_string(),
            "--model-id".into(),
            self.name.clone(),
            "--allow-same-code".into(),
        ]);
        let head = facts::golden_meta(&dump, "self_meta")
            .and_then(|m| serde_json::from_str::<Value>(&m).ok())
            .and_then(|m| m["code"]["head"].as_str().map(String::from))
            .unwrap_or_default();
        if head.is_empty() {
            args.push("--allow-unstamped".into());
        } else {
            args.extend(["--pre-change".into(), head]);
        }
        args.extend(self.stamp_flags());
        let r = self.exec(stem, args)?;
        let text = r.stdout();
        let what = format!("against the recorded dump {}", dump.display());
        Ok(if r.timed_out || r.code.is_none() {
            Ok((Verdict::Error, format!("drift check {}", r.describe())))
        } else if r.ok() && text.contains("golden gate: PASS") {
            Ok((Verdict::Pass, format!("logits unchanged {what}")))
        } else if text.contains("golden gate: FAIL") {
            let bad: Vec<String> = text.lines().filter(|l| l.trim_end().ends_with("FAIL") && !l.contains("golden gate")).map(|l| l.trim().to_string()).collect();
            Ok((Verdict::Fail, format!("the logits moved {what}: {}", if bad.is_empty() { "self-golden budgets exceeded".into() } else { bad.join("; ") })))
        } else if r.stderr().contains("different configuration") {
            Err(format!("the recorded dump {} does not apply: {}", dump.display(), r.tail(2)))
        } else {
            Ok((Verdict::Error, format!("drift check {}: {}", r.describe(), r.tail(4))))
        })
    }
}

/// The engine against the f64 CPU oracle on the short prompt set, prefill rows (fx1, review finding 1). The oracle runs
/// the same weights and model definition (recipe, scale interpretation, GDN) in f32 / f64, so it does not catch a recipe
/// error, but it is deterministic and close: a kernel, scale or TP regression shows at the resolution of the measured
/// clean spread instead of behind a reference server's run-to-run noise.
fn st_acc_oracle(c: &mut Ctx) -> Result<Out> {
    let (g, how) = match c.oracle_golden()? {
        Golden::Have(g, h) => (g, h),
        Golden::None(why) => return Ok(Out::skip(why)),
    };
    if facts::golden_meta(&g, "self_golden").as_deref() != Some("1")
        || !facts::golden_meta(&g, "self_meta").is_some_and(|m| m.contains("\"backend\":\"cpu\""))
    {
        return Ok(Out::error(format!("{}: not a CPU-oracle golden (want self_golden = 1 written by the cpu backend)", g.display())));
    }
    let (mut extra, kl) = c.oracle_flags();
    // this run's own dump: copied to golden/baselines/ it becomes the recorded dump of the drift check
    let rec = c.path(&format!("baselines/{}.selfgolden.safetensors", c.dump_tag(&g, 1)));
    extra.extend(["--write-self-golden".into(), rec.display().to_string()]);
    let a = c.accuracy("acc-oracle", &g, "prefill", false, 1, &extra)?;
    let mut o = c.acc_out(a, &g, &how, "oracle");
    o.notes.push(format!("floor: mean KL(oracle || engine) over the oracle's top-64 + tail, scored prefill rows, <= {kl:.1e} (measured, docs/results/2026-10-22-fx1-oracle-gate.md); top-1 not gated. The floor guards absolute accuracy; it does not see a 1 % scale bug, the drift check does"));
    if rec.exists() {
        o = o.art(c.rel(&rec));
    }
    if o.verdict != Verdict::Pass {
        return Ok(o);
    }
    o = match c.drift_check(&g, 1, "acc-oracle-drift")? {
        Ok((Verdict::Pass, why)) => {
            o.notes.push(format!("drift check: {why}"));
            o
        }
        Ok((v, why)) => Out::new(v, why).metrics(o.metrics).art(c.rel(&c.path("logs/acc-oracle-drift.out"))),
        Err(why) => {
            let mut m = Out::new(Verdict::MissingBaseline, format!("MISSING-BASELINE: {why}; the oracle floor passed ({}), but on this model only a recorded same-backend dump can see a 1 % scale bug. Record it from clean main: copy {} to golden/baselines/ and commit it", o.reason, c.rel(&rec)));
            m.metrics = o.metrics;
            m.artifacts = o.artifacts;
            m
        }
    };
    Ok(o)
}

// ---- serving + bench ----

struct BenchRun {
    json: Value,
    path: PathBuf,
    out: PathBuf,
    server_err: PathBuf,
    load_s: f64,
}

impl Ctx {
    /// The decode byte floor inputs: (GB read per token, bandwidth GB/s, where each came from).
    fn floor_inputs(&self, per_rank_gb: Option<f64>) -> (Option<f64>, Option<f64>, String) {
        let mut src = vec![];
        let gb = if let Some(g) = self.a.gb_per_token {
            src.push("gb/token: --gb-per-token".to_string());
            Some(g)
        } else if let Some(g) = per_rank_gb {
            src.push("gb/token: busiest tp rank's weights from the plan".to_string());
            Some(g)
        } else {
            self.fact_f64("active_gb").map(|g| {
                src.push(format!("gb/token: {} checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate)", if self.facts.get("moe") == Some(&json!(true)) { "estimated from" } else { "from" }));
                g
            })
        };
        let arch = self.fact_str("device_arch").unwrap_or_default();
        let bw = if let Some(b) = self.a.bandwidth_gbs {
            src.push("bandwidth: --bandwidth-gbs".into());
            Some(b)
        } else if arch == "gfx1201" {
            src.push("bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md".into());
            Some(627.0)
        } else if arch.starts_with("sm_121") {
            src.push("bandwidth: 221 GB/s, the GB10 measured figure".into());
            Some(221.0)
        } else if arch == "applegpu_g17s" {
            src.push("bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md".into());
            Some(614.0)
        } else {
            self.fact_f64("copy_gbs").map(|b| {
                src.push("bandwidth: doctor's measured device copy (read+write bytes counted)".into());
                b
            })
        };
        (gb, bw, src.join("; "))
    }

    /// Start a server, run `bench speed` against it, stop the server. `tp` > 1 serves sharded.
    fn bench_run(
        &mut self,
        stem: &str,
        sizes: &[usize],
        gen: usize,
        reps: usize,
        tp: usize,
        per_rank_gb: Option<f64>,
    ) -> Result<BenchRun> {
        let max_seq = server_max_seq(sizes, gen);
        let port = proc::free_port(self.a.port);
        let url = format!("http://127.0.0.1:{port}");
        let served = format!("qualify-{}", self.name);
        let mut sargs = vec!["serve".to_string()];
        sargs.extend(self.load_args(tp));
        sargs.extend([
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            "--served-model-name".into(),
            served.clone(),
            "--max-seq".into(),
            max_seq.to_string(),
            "--profile-every".into(),
            "0".into(),
            "--no-supervisor".into(),
            "--no-ui".into(),
        ]);
        let mut cmd = self.base_cmd();
        cmd.args(&sargs);
        self.cmds.push(format!("{} {}", self.bin.display(), sargs.join(" ")));
        let t = Instant::now();
        let mut server = proc::Server::start(cmd, &self.path(&format!("logs/{stem}-server")), url.clone())?;
        let load_s =
            server.wait_healthy(Duration::from_secs(self.a.stage_timeout_min * 60)).map_err(|e| anyhow!("{e:#}"))?;
        self.say(&format!("server up in {load_s:.0}s on {url} (max-seq {max_seq})"));
        let _ = t;
        let bench_dir = self.path("bench");
        let label = format!("qualify-{}-{}-{}", self.name, self.a.backend, stem);
        let (gb, bw, _src) = self.floor_inputs(per_rank_gb);
        let mut bargs: Vec<String> = vec![
            "bench".into(),
            "speed".into(),
            "--url".into(),
            url.clone(),
            "--model".into(),
            served,
            "--sizes".into(),
            sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","),
            "--gen".into(),
            gen.to_string(),
            "--reps".into(),
            reps.to_string(),
            "--warmup".into(),
            "1".into(),
            "--thinking".into(),
            "off".into(),
            "--label".into(),
            label.clone(),
            "--out-dir".into(),
            bench_dir.display().to_string(),
            "--tp".into(),
            tp.to_string(),
            "--fingerprint".into(),
            format!("model={}", self.name),
            "--fingerprint".into(),
            format!("backend={}", self.a.backend),
            "--fingerprint".into(),
            format!("build={}", BUILD_STAMP.unwrap_or("unstamped")),
            "--timeout-s".into(),
            "900".into(),
        ];
        if let Some(g) = gb {
            bargs.extend(["--gb-per-token".into(), format!("{g:.4}")]);
        }
        if let Some(b) = bw {
            bargs.extend(["--bandwidth-gbs".into(), format!("{b}")]);
        }
        let doctor = self.path("doctor.json");
        if doctor.exists() {
            bargs.extend(["--fingerprint-file".into(), doctor.display().to_string()]);
        }
        let r = self.exec(&format!("{stem}-bench"), bargs)?;
        let server_err = server.err.clone();
        drop(server);
        let path = bench_dir.join(format!("{}.json", intarsia_bench::speed::sanitize_label(&label)));
        if !path.exists() {
            if r.stderr().contains("server did not report usable prompt_tokens") {
                bail!("SKIP: bench speed cannot size prompts for this model: the server reports the same prompt length whatever the message says, i.e. the chat template ignores the user message (reranker / embedding style templates do); speed needs a raw-completion bench, which does not exist yet");
            }
            bail!("bench speed {} and wrote no result: {}", r.describe(), r.tail(6));
        }
        if r.timed_out || r.code.is_none() {
            bail!("bench speed {}", r.describe());
        }
        Ok(BenchRun { json: read_json(&path)?, path, out: r.out, server_err, load_s })
    }

    fn bench_out(&self, b: &BenchRun, tp: usize, per_rank_gb: Option<f64>) -> Out {
        let rows = b.json["rows"].as_array().cloned().unwrap_or_default();
        let (gb, bw, src) = self.floor_inputs(per_rank_gb);
        let mut slim = vec![];
        let mut notes = vec![];
        let mut bad = vec![];
        for r in &rows {
            if let Some(s) = r["skipped"].as_str() {
                notes.push(format!("size {} skipped: {s}", r["target_prompt_tokens"]));
                slim.push(json!({"size": r["target_prompt_tokens"], "skipped": s}));
                continue;
            }
            if r["reps_ok"].as_u64().unwrap_or(0) == 0 {
                bad.push(format!("size {}: every call failed", r["target_prompt_tokens"]));
            }
            if let Some(p) = r["pct_of_floor"].as_f64() {
                if p > 100.0 {
                    notes.push(format!("size {}: {p:.0}% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)", r["target_prompt_tokens"]));
                }
            }
            slim.push(json!({
                "size": r["target_prompt_tokens"], "prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_s"], "prefill_tok_s": r["prefill_tok_s"], "decode_tok_s": r["decode_tok_s"],
                "decode_stdev": r["decode_tok_s_stdev"], "floor_tok_s": r["bandwidth_floor_tok_s"], "pct_of_floor": r["pct_of_floor"], "reps_ok": r["reps_ok"], "errors": r["errors"], "graph": r["graph"],
            }));
        }
        for w in b.json["warnings"].as_array().cloned().unwrap_or_default() {
            if let Some(s) = w.as_str() {
                notes.push(format!("bench warning: {s}"));
            }
        }
        let measured = slim.iter().filter(|r| r["skipped"].is_null()).count();
        let m = json!({"rows": slim, "tp": tp, "gb_per_token": gb, "bandwidth_gbs": bw, "floor_inputs": src, "server_load_s": b.load_s, "label": b.json["label"]});
        let mut o = if measured == 0 {
            Out::error("every size was skipped: none fits the server's context window")
        } else if !bad.is_empty() {
            Out::fail(bad.join("; "))
        } else {
            let first = slim.iter().find(|r| r["skipped"].is_null()).unwrap();
            Out::pass(format!(
                "decode {:.1} tok/s at {} tokens{}",
                first["decode_tok_s"].as_f64().unwrap_or(0.0),
                first["size"],
                first["pct_of_floor"].as_f64().map(|p| format!(" ({p:.0}% of the byte floor)")).unwrap_or_default()
            ))
        };
        o = o.metrics(m).art(self.rel(&b.path)).art(self.rel(&b.out)).art(self.rel(&b.server_err));
        for n in notes {
            o = o.note(n);
        }
        o
    }
}

fn st_bench(c: &mut Ctx) -> Result<Out> {
    let sizes = c.quick_sizes();
    let b = c.bench_run("bench", &sizes, c.gen_tokens(false), c.reps(false), 1, None)?;
    Ok(c.bench_out(&b, 1, None))
}

// =====================================================================================================================
// IN-DEPTH stages
// =====================================================================================================================

fn st_acc_long(c: &mut Ctx) -> Result<Out> {
    let (g, how) = match c.golden(true)? {
        Golden::Have(g, h) => (g, h),
        Golden::None(why) => return Ok(Out::skip(why)),
    };
    // With a 4k golden the prefix-reuse gate runs there (`acc-4k`): the long golden's longest prompt is 700 tokens, so at
    // a prefill chunk above ~350 its reuse gate has nothing to run on, prints "nothing ran" and fails (a prompt that cannot
    // cross a chunk boundary is a reported skip, so the short prompts alone do not fail it). Without one (every other
    // model today) this stage is unchanged.
    let reuse_in_4k = matches!(c.golden_4k()?, Golden::Have(g4, _) if g4 != g);
    let a = c.accuracy("acc-long", &g, if reuse_in_4k { "both" } else { "all" }, true, 1, &[])?;
    let mut o = c.acc_out(a, &g, &how, "long");
    if reuse_in_4k {
        o.notes.push("prefix reuse is gated in acc-4k, on the 4k golden".into());
    }
    Ok(o)
}

fn st_acc_4k(c: &mut Ctx) -> Result<Out> {
    let (g, how) = match c.golden_4k()? {
        Golden::Have(g, h) => (g, h),
        Golden::None(why) => return Ok(Out::skip(why)),
    };
    // a model whose long golden is the 4k one (no `-long` file) already ran it, reuse gate included, in acc-long
    if let Golden::Have(gl, _) = c.golden(true)? {
        if gl == g {
            return Ok(Out::skip("the 4k golden is the long golden: covered by acc-long"));
        }
    }
    // The class floor asks top-1 >= 99.5%, which no build reaches on these prompts (the fp32 reference's top-2 gap is under
    // 0.1 nats at 4% of the positions and 90% of the misses sit there: bf16 noise, docs/results/2026-10-15-metal-golden-4k.md),
    // so there is no absolute top-1 floor here. The stage gates KL (class floor), tail mass, non-finite rows, prefix reuse, and,
    // when the backend has a recorded baseline for this golden, "no worse than the baseline beyond its noise envelope".
    // As in acc-short (fx1): a model without a full-precision golden reads MISSING-BASELINE instead of passing on the
    // KL / tail / reuse gates alone; other models keep the floor-only fallback.
    let needs = c.needs_baseline(&g);
    let (mut extra, note) = c.recorded_baseline(&g, true).unwrap_or_else(|why| (vec![], why));
    let have = !extra.is_empty();
    extra.splice(0..0, ["--min-top1".to_string(), "0".to_string()]);
    let a = c.accuracy("acc-4k", &g, "all", true, 1, &extra)?;
    let o = c.acc_out(a, &g, &how, "4k");
    Ok(baseline_verdict(o, note, needs && !have, &c.name, "acc-4k"))
}

/// Attach the baseline note. A gate that passed without the baseline its model class needs (`missing`) reads
/// MISSING-BASELINE (exit 1) instead, with the floor's own reading kept in the reason; a FAIL or ERROR stays what it is.
fn baseline_verdict(mut o: Out, note: String, missing: bool, model: &str, stage: &str) -> Out {
    if !(missing && o.verdict == Verdict::Pass) {
        o.notes.push(note);
    } else {
        o.verdict = Verdict::MissingBaseline;
        o.reason = format!(
            "MISSING-BASELINE: {note}. {model} has no full-precision golden (MoE / hybrid / quantized, golden from a reference server), so {stage}'s floor alone does not gate it (it read: {}); record one with the envelope stage and commit it to golden/baselines/",
            o.reason
        );
    }
    o
}

impl Ctx {
    /// `accuracy` flags that gate `golden` against the backend's recorded baseline (and noise envelope), and a note that
    /// says which baseline; `Err` says why there is none. A baseline that does not apply to this run (other chunk, decode
    /// window or golden file) is not used: `acc-4k` then gates without it, `acc-short` of a model that needs one reports
    /// MISSING-BASELINE. `long`: the decode window of the long / 4k goldens.
    fn recorded_baseline(&self, golden: &Path, long: bool) -> std::result::Result<(Vec<String>, String), String> {
        let mut dirs: Vec<PathBuf> = vec![];
        if let Some(d) = golden.parent() {
            dirs.push(d.join("baselines"));
        }
        if let Some(r) = facts::repo_root() {
            dirs.push(r.join("golden/baselines"));
            if let Some(m) = facts::main_checkout(&r) {
                dirs.push(m.join("golden/baselines"));
            }
        }
        let backend = self.a.backend.as_str();
        let Some((bp, ep)) = facts::find_baseline(golden, backend, &dirs) else {
            return Err(format!("no baseline for backend {backend} next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated"));
        };
        let none = |why: String| Err(format!("baseline {} not used ({why}): KL, tail, prefix-reuse gates only, top-1 not gated", bp.display()));
        let Ok(b) = read_json(&bp) else { return none("unreadable".into()) };
        let chunk = self.state.get("plan").and_then(|s| s.metrics["prefill_chunk"].as_u64());
        if b["run"]["backend"].as_str() != Some(backend) {
            return none("other backend".into());
        }
        if chunk.is_none() || b["run"]["prefill_chunk"].as_u64() != chunk {
            return none(format!("recorded at prefill chunk {}, this run uses {}", b["run"]["prefill_chunk"], chunk.map_or("?".into(), |c| c.to_string())));
        }
        if b["run"]["decode_positions"].as_u64() != Some(self.decode_positions(long) as u64) {
            return none("other decode window".into());
        }
        if b["golden"]["size"].as_u64() != std::fs::metadata(golden).ok().map(|m| m.len()) {
            return none("recorded against another golden file".into());
        }
        let head = b["code"]["head"].as_str().unwrap_or("");
        let mut args = vec!["--baseline".to_string(), bp.display().to_string()];
        if let Some(e) = &ep {
            args.extend(["--noise-envelope".into(), e.display().to_string()]);
        }
        // the baseline is a recorded earlier run: its commit is the declared pre-change code (or the same tree, rebuilt)
        args.push("--allow-same-code".into());
        if head.is_empty() {
            args.push("--allow-unstamped".into());
        } else {
            args.extend(["--pre-change".into(), head.to_string()]);
        }
        args.extend(self.stamp_flags());
        Ok((args, format!("gated against the recorded baseline {} (commit {}){}", bp.display(), if head.is_empty() { "unstamped" } else { head }, if ep.is_some() { " and its noise envelope" } else { ", no noise envelope" })))
    }

    /// The model has no full-precision golden (MoE / hybrid / quantized, the classes `make_golden` refuses) and `golden`
    /// is not one either (it came from a reference server, `dtype = server`, or names no float32 dtype): its static
    /// floor is loose, so its accuracy gate needs a recorded baseline.
    fn needs_baseline(&self, golden: &Path) -> bool {
        let class = ["moe", "hybrid", "quantized"].iter().any(|k| self.facts.get(*k) == Some(&json!(true)));
        class && facts::golden_meta(golden, "dtype").as_deref() != Some("float32")
    }
}

fn st_envelope(c: &mut Ctx) -> Result<Out> {
    let (g, _how) = match c.golden(false)? {
        Golden::Have(g, h) => (g, h),
        Golden::None(why) => return Ok(Out::skip(format!("no baseline can be recorded without a golden: {why}"))),
    };
    // The baseline is the quick short-accuracy run's (same golden, same flags): record it again only when that stage left none.
    let base_rel =
        c.state.get("acc-short").and_then(|s| s.facts.get("short_baseline")).and_then(|v| v.as_str()).map(String::from);
    let baseline = match base_rel.map(|r| c.path(&r)).filter(|p| p.exists()) {
        Some(p) => p,
        None => {
            let a = c.accuracy("baseline", &g, "both", false, 1, &[])?;
            if a.verdict == Verdict::Error {
                return Ok(Out::error(format!("recording the baseline: {}", a.reason)));
            }
            a.baseline
        }
    };
    let env = c.path("baselines/envelope.json");
    let _ = std::fs::remove_file(&env);
    let mut perturb = vec!["chunk".to_string()];
    // attn-simt switches off the flash / split-KV attention kernels, which only cuda and hip have; cpu and metal
    // measure the noise envelope with the chunk perturbation alone
    if matches!(c.a.backend.as_str(), "cuda" | "hip") {
        perturb.push("attn-simt".into());
    }
    let mut runs = vec![];
    for p in &perturb {
        let mut args = vec!["accuracy".to_string()];
        args.extend(c.load_args(1));
        args.extend([
            "--golden".into(),
            g.display().to_string(),
            "--mode".into(),
            "both".into(),
            "--decode-positions".into(),
            c.decode_positions(false).to_string(),
            "--max-seq".into(),
            c.acc_max_seq(false).to_string(),
            "--noise-ref".into(),
            baseline.display().to_string(),
            "--noise-perturb".into(),
            p.clone(),
            "--write-noise-envelope".into(),
            env.display().to_string(),
            "--allow-same-code".into(),
        ]);
        args.extend(c.stamp_flags());
        let r = c.exec(&format!("envelope-{}", facts::sanitize(p)), args)?;
        if !r.ok() || !env.exists() {
            return Ok(Out::error(format!("measuring the noise envelope with '{p}' {}: {}", r.describe(), r.tail(5))));
        }
        runs.push(json!({"perturbation": p, "wall_s": r.wall_s}));
    }
    // self-check: the pair must gate the same build (the point of writing them)
    let mut args = vec!["accuracy".to_string()];
    args.extend(c.load_args(1));
    args.extend([
        "--golden".into(),
        g.display().to_string(),
        "--mode".into(),
        "both".into(),
        "--decode-positions".into(),
        c.decode_positions(false).to_string(),
        "--max-seq".into(),
        c.acc_max_seq(false).to_string(),
        "--baseline".into(),
        baseline.display().to_string(),
        "--noise-envelope".into(),
        env.display().to_string(),
        "--allow-same-code".into(),
    ]);
    args.extend(c.stamp_flags());
    // a stamped build is its own "before": the baseline is declared as coming from this build's commit
    if let Some(head) = BUILD_STAMP.and_then(|s| s.split_whitespace().find_map(|t| t.strip_prefix("head="))) {
        args.extend(["--pre-change".to_string(), head.to_string()]);
    }
    let r = c.exec("baseline-selfcheck", args)?;
    let text = r.stdout();
    let verdict_line =
        text.lines().find(|l| l.contains("baseline verdict")).map(|l| l.trim().to_string()).unwrap_or_default();
    if r.timed_out || r.code.is_none() {
        return Ok(Out::error(format!("baseline self-check {}", r.describe())));
    }
    let published = c.path("baselines");
    let m = json!({"baseline": c.rel(&baseline), "envelope": c.rel(&env), "perturbations": runs, "selfcheck_verdict_line": verdict_line,
        "use": format!("intarsia accuracy --model <dir> --backend {} --golden {} --baseline {} --noise-envelope {}  (copy both files next to the golden, e.g. golden/baselines/)", c.a.backend, g.display(), c.rel(&baseline), c.rel(&env))});
    let _ = published;
    let o = if verdict_line.contains(": PASS") {
        Out::pass(format!(
            "baseline and envelope ({}) written; they gate this build: {}",
            perturb.join(" + "),
            verdict_line.split(':').next_back().unwrap_or("").trim()
        ))
    } else if verdict_line.is_empty() {
        Out::error(format!("baseline self-check printed no baseline verdict: {}", r.tail(4)))
    } else {
        Out::fail(format!("the recorded baseline does not gate its own build: {verdict_line}"))
    };
    Ok(o.metrics(m).art(c.rel(&baseline)).art(c.rel(&env)).art(c.rel(&r.out)))
}

fn st_bench_long(c: &mut Ctx) -> Result<Out> {
    let sizes = c.long_sizes();
    let b = c.bench_run("bench-long", &sizes, c.gen_tokens(true), c.reps(true), 1, None)?;
    Ok(c.bench_out(&b, 1, None))
}

/// Context window of the bench server: the largest prompt plus the generation, with the bench's own margin (it skips a
/// size whose estimate plus 2% and 16 tokens does not fit) and at least 2048 (its calibration probes need room).
fn server_max_seq(sizes: &[usize], gen: usize) -> usize {
    let biggest = sizes.iter().max().copied().unwrap_or(1024);
    (((biggest as f64 * 1.03) as usize + gen + 96).div_ceil(256) * 256).max(2048)
}

fn localeval_dir(c: &Ctx) -> PathBuf {
    c.a.localeval_dir.clone().unwrap_or_else(|| PathBuf::from("/opt/code/deepeval"))
}

fn list_dirs(p: &Path) -> Vec<String> {
    std::fs::read_dir(p)
        .map(|rd| {
            rd.flatten().filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().to_string()).collect()
        })
        .unwrap_or_default()
}

fn st_ab(c: &mut Ctx) -> Result<Out> {
    let Some(refr) = c.a.reference.clone() else {
        return Ok(Out::skip("no --reference server given"));
    };
    if !(refr.starts_with("http://") || refr.starts_with("https://")) {
        return Ok(Out::skip("--reference is a model directory, not a server URL: there is nothing to A/B against"));
    }
    let dir = localeval_dir(c);
    if !dir.join("pyproject.toml").exists() {
        return Ok(Out::skip(format!(
            "localeval is not at {} (--localeval-dir); the A/B must run through localeval, never an ad-hoc script",
            dir.display()
        )));
    }
    if !have_cmd("uv") {
        return Ok(Out::skip(
            "uv is not installed here, so localeval cannot run; run the A/B from a machine that has /opt/code/deepeval",
        ));
    }
    let base = intarsia_bench::http::api_base(&refr);
    let ref_model = c.reference_model(&base)?;
    let sizes = c.quick_sizes();
    let (gen, reps) = (c.gen_tokens(false), c.reps(false));
    // localeval calibrates with a probe of about 4.5k tokens whatever the sizes are, so the context must hold it
    let max_seq = server_max_seq(&sizes, gen).max(8192);
    let port = proc::free_port(c.a.port);
    let url = format!("http://127.0.0.1:{port}");
    let served = format!("qualify-{}", c.name);
    let mut sargs = vec!["serve".to_string()];
    sargs.extend(c.load_args(1));
    sargs.extend([
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--served-model-name".into(),
        served.clone(),
        "--max-seq".into(),
        max_seq.to_string(),
        "--profile-every".into(),
        "0".into(),
        "--no-supervisor".into(),
        "--no-ui".into(),
    ]);
    let mut cmd = c.base_cmd();
    cmd.args(&sargs);
    c.cmds.push(format!("{} {}", c.bin.display(), sargs.join(" ")));
    let mut server = proc::Server::start(cmd, &c.path("logs/ab-server"), url.clone())?;
    server.wait_healthy(Duration::from_secs(c.a.stage_timeout_min * 60)).map_err(|e| anyhow!("{e:#}"))?;
    let results = dir.join("results");
    let mut runs = vec![];
    for (arm, arm_url, arm_model) in
        [("ours", format!("{url}/v1"), served.clone()), ("reference", base.clone(), ref_model.clone())]
    {
        let before = list_dirs(&results);
        let mut args: Vec<String> = vec![
            "run".into(),
            "--project".into(),
            dir.display().to_string(),
            "localeval".into(),
            "speed".into(),
            "--url".into(),
            arm_url,
            "--model".into(),
            arm_model,
            "--label".into(),
            format!("qualify-{}-{arm}", c.name),
            "--thinking".into(),
            "off".into(),
            "--sizes".into(),
            sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","),
            "--gen".into(),
            gen.to_string(),
            "--reps".into(),
            reps.to_string(),
        ];
        if c.a.no_unim {
            args.push("--no-unim".into());
        }
        let mut cmd = Command::new("uv");
        cmd.args(&args);
        c.cmds.push(format!("uv {}", args.join(" ")));
        let r = proc::run_captured(
            cmd,
            &c.path(&format!("logs/ab-{arm}")),
            Duration::from_secs(c.a.stage_timeout_min * 60),
        )?;
        let new_dir = list_dirs(&results).into_iter().find(|d| !before.contains(d));
        let result = new_dir.as_ref().and_then(|d| read_json(&results.join(d).join("result.json")).ok());
        if !r.ok() {
            drop(server);
            return Ok(
                Out::error(format!("localeval speed ({arm}) {}: {}", r.describe(), r.tail(6))).art(c.rel(&r.err))
            );
        }
        let Some(res) = result else {
            drop(server);
            return Ok(Out::error(format!(
                "localeval speed ({arm}) exited 0 but left no new results/<run_id>/result.json under {}",
                results.display()
            )));
        };
        let rows: Vec<Value> = res["speed"]["rows"].as_array().cloned().unwrap_or_default().iter().map(|r| json!({"size": r["target_prompt_tokens"], "prefill_tok_s": r["prefill_tok_s"]["median"], "decode_tok_s": r["decode_tok_s"]["median"], "ttft_s": r["ttft_s"]["median"]})).collect();
        runs.push(json!({"arm": arm, "run_id": res["run_id"], "result": results.join(new_dir.unwrap()).join("result.json").display().to_string(), "model": ref_model_or(&res), "rows": rows}));
    }
    drop(server);
    let ratio: Vec<Value> = {
        let ours = runs[0]["rows"].as_array().cloned().unwrap_or_default();
        let theirs = runs[1]["rows"].as_array().cloned().unwrap_or_default();
        ours.iter()
            .filter_map(|o| {
                let t = theirs.iter().find(|t| t["size"] == o["size"])?;
                Some(json!({"size": o["size"], "decode_ratio": f(&o["decode_tok_s"])? / f(&t["decode_tok_s"])?, "prefill_ratio": f(&o["prefill_tok_s"]).zip(f(&t["prefill_tok_s"])).map(|(a, b)| a / b)}))
            })
            .collect()
    };
    Ok(Out::pass(format!("localeval speed on both servers; ours/reference decode {}", ratio.iter().map(|r| format!("{}:{:.2}x", r["size"], r["decode_ratio"].as_f64().unwrap_or(0.0))).collect::<Vec<_>>().join(" ")))
        .metrics(json!({"runs": runs, "ratios": ratio, "reference_url": base, "note": "one sequential round, not the interleaved rounds of scripts/phase1b-acceptance.sh; a difference under the noise floor (decode about 5%) is not a result"})))
}

fn ref_model_or(res: &Value) -> Value {
    res["speed"]["model_info"]["served_models"][0].clone()
}

fn st_tp(c: &mut Ctx) -> Result<Out> {
    let tp = c.a.tp;
    if tp <= 1 {
        return Ok(Out::skip("--tp not given (or 1)"));
    }
    // per-rank bytes for the floor: the busiest rank's weights from the tp plan
    let mut pargs = vec!["plan".to_string(), "show".into()];
    pargs.extend(c.load_args(tp));
    pargs.push("--json".into());
    let pr = c.exec("tp-plan", pargs)?;
    if !pr.ok() {
        return Ok(Out::fail(format!("the model does not load with --tp {tp}: {}", pr.tail(6))).art(c.rel(&pr.err)));
    }
    let pj: Value = serde_json::from_str(&pr.stdout()).map_err(|e| anyhow!("tp plan JSON: {e}"))?;
    let per_rank_gb = pj["tp"]["ranks"]
        .as_array()
        .map(|r| r.iter().filter_map(|x| x["weights_bytes"].as_u64()).max().unwrap_or(0) as f64 / 1e9)
        .filter(|g| *g > 0.0);
    let tp_plan = json!({"world": pj["tp"]["world"], "comm": pj["tp"]["comm"], "all_reduce_calls_per_decode_step": pj["tp"]["all_reduce_calls_per_decode_step"], "ranks": pj["tp"]["ranks"], "load_s": pr.wall_s});
    let sizes = c.quick_sizes();
    let b = c.bench_run(&format!("tp{tp}"), &sizes, c.gen_tokens(false), c.reps(false), tp, per_rank_gb)?;
    let mut o = c.bench_out(&b, tp, per_rank_gb);
    // accuracy with the sharded model against the short golden
    let acc = match c.golden(false)? {
        Golden::Have(g, how) => {
            let a = c.accuracy(&format!("tp{tp}-acc"), &g, "both", false, tp, &[])?;
            Some((g, how, a))
        }
        Golden::None(_) => None,
    };
    // and against the f64 CPU oracle: the gate that sees a 1 % partial-scale bug on one rank (fx1)
    let oracle = match c.oracle_golden()? {
        Golden::Have(g, _) => {
            let (mut extra, _) = c.oracle_flags();
            let rec = c.path(&format!("baselines/{}.selfgolden.safetensors", c.dump_tag(&g, tp)));
            extra.extend(["--write-self-golden".into(), rec.display().to_string()]);
            let a = c.accuracy(&format!("tp{tp}-oracle"), &g, "prefill", false, tp, &extra)?;
            let drift = if a.verdict == Verdict::Pass { Some(c.drift_check(&g, tp, &format!("tp{tp}-oracle-drift"))?) } else { None };
            Some((a, drift, rec))
        }
        Golden::None(_) => None,
    };
    let mut metrics = o.metrics.clone();
    metrics["tp_plan"] = tp_plan;
    if let Some((a, drift, rec)) = &oracle {
        metrics["oracle"] = json!({"gate_class": a.gate_class, "modes": a.modes, "verdict": a.verdict.label(), "reason": a.reason});
        match a.verdict {
            Verdict::Fail => o = Out::fail(format!("tp={tp} against the CPU oracle: {}", a.reason)).art(c.rel(&b.path)),
            Verdict::Error => o = Out::error(format!("tp={tp} against the CPU oracle: {}", a.reason)),
            _ => o.reason = format!("{}; oracle {}", o.reason, acc_summary(&a.modes)),
        }
        match drift {
            Some(Ok((Verdict::Pass, why))) => o.notes.push(format!("tp={tp} drift check: {why}")),
            Some(Ok((Verdict::Error, why))) => o = Out::error(format!("tp={tp} drift check: {why}")),
            Some(Ok((_, why))) if o.verdict != Verdict::Error => o = Out::fail(format!("tp={tp}: {why}")).art(c.rel(&b.path)),
            Some(Err(why)) if o.verdict == Verdict::Pass => {
                o = Out::new(Verdict::MissingBaseline, format!("MISSING-BASELINE: tp={tp}: {why}; record it from clean main: copy {} to golden/baselines/", c.rel(rec)))
            }
            _ => {}
        }
    }
    if let Some((g, how, a)) = acc {
        metrics["accuracy"] = json!({"golden": g.display().to_string(), "golden_how": how, "gate_class": a.gate_class, "modes": a.modes, "verdict": a.verdict.label(), "reason": a.reason});
        match a.verdict {
            Verdict::Fail if o.verdict == Verdict::Pass => {
                o = Out::fail(format!("tp={tp} accuracy: {}", a.reason)).art(c.rel(&b.path))
            }
            Verdict::Error => o = Out::error(format!("tp={tp} accuracy: {}", a.reason)),
            Verdict::Pass if o.verdict == Verdict::Pass => {
                o.reason = format!("{}; accuracy {}", o.reason, acc_summary(&a.modes))
            }
            _ => {}
        }
        o.artifacts.push(c.rel(&a.baseline));
    } else {
        o.notes.push("no golden: the tp accuracy comparison was skipped".into());
    }
    o.metrics = metrics;
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_context_covers_the_benchs_margin() {
        // bench speed skips a size when ~size + gen, plus 2% and 16 tokens, exceeds the context
        for (size, gen) in [(16384usize, 64usize), (4096, 128), (128, 128), (256, 16)] {
            let need = ((size + gen) as f64 * 1.02) as usize + 16;
            assert!(server_max_seq(&[size], gen) >= need, "{size}+{gen}");
        }
        assert!(server_max_seq(&[64], 16) >= 2048);
    }

    #[test]
    fn missing_kernels_are_listed_from_the_load_error() {
        // the text built in crates/intarsia-core/src/plan.rs
        let e = "Error: no kernel for 2 op(s) of this model have no kernel on hip/gfx1201:\n  - matmul: 3 shape(s) without a kernel, e.g. m=1 n=4096 k=4096 fp8; needed by layer.attn.q\n  - attn_decode: 1 shape(s) without a kernel, e.g. hd=256; needed by layer.attn, ...\n";
        let m = missing_kernel_lines(e);
        assert_eq!(m.len(), 2);
        assert!(m[0].starts_with("matmul: 3 shape(s) without a kernel"));
        assert!(m[1].starts_with("attn_decode:"));
    }

    #[test]
    fn a_pass_without_a_needed_baseline_is_missing_baseline() {
        let pass = || Out::pass("prefill: KL 5.8e-3");
        let o = baseline_verdict(pass(), "no baseline for backend hip".into(), true, "Qwen3.6-35B-A3B-NVFP4", "acc-short");
        assert_eq!(o.verdict, Verdict::MissingBaseline);
        assert!(o.reason.starts_with("MISSING-BASELINE: no baseline for backend hip") && o.reason.contains("KL 5.8e-3"));
        assert!(o.verdict.is_failing(), "MISSING-BASELINE fails the run (exit 1)");
        // with the baseline (or for a model with a full-precision golden) the floor verdict stands, note attached
        let o = baseline_verdict(pass(), "gated against the recorded baseline".into(), false, "m", "acc-short");
        assert_eq!((o.verdict, o.notes.len()), (Verdict::Pass, 1));
        // a FAIL stays a FAIL
        let o = baseline_verdict(Out::fail("KL 4e-2"), "no baseline".into(), true, "m", "acc-4k");
        assert_eq!(o.verdict, Verdict::Fail);
    }

    #[test]
    fn stage_table_is_consistent() {
        let ids: Vec<&str> = STAGES.iter().map(|s| s.id).collect();
        let mut dedup = ids.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(ids.len(), dedup.len(), "duplicate stage id");
        // quick stages come first, the dependencies of every stage come before it
        let first_full = STAGES.iter().position(|s| s.section == "full").unwrap();
        assert!(
            STAGES[..first_full].iter().all(|s| s.section == "quick")
                && STAGES[first_full..].iter().all(|s| s.section == "full")
        );
        for (i, s) in STAGES.iter().enumerate() {
            for n in s.needs {
                assert!(ids[..i].contains(n), "{} needs {n} which does not come before it", s.id);
            }
        }
        // the spec's order of the quick section
        assert_eq!(&ids[..9], &["doctor", "recipe", "plan", "kernels", "gate", "acc-format", "acc-short", "acc-oracle", "bench"]);
    }

    #[test]
    fn stamp_formats_as_utc() {
        let s = chrono_stamp();
        assert_eq!(s.len(), 15);
        assert_eq!(&s[8..9], "-");
    }

    #[test]
    fn accuracy_failures_are_digested() {
        let t = "prefill floor [bf16]: top-1 99.08% (need >= 99.5) FAIL | full-vocab KL 5.503e-4 (need <= 1.0e-3) ok | non-finite rows 0 ok => FAIL\ndecode (teacher-forced) floor [bf16]: top-1 100.00% (need >= 99.5) ok | full-vocab KL 1.089e-3 (need <= 1.0e-3) FAIL | non-finite rows 0 ok => FAIL\nfloor verdict (x): FAIL   prefix-reuse verdict: FAIL\n";
        assert_eq!(
            acc_fail_digest(t),
            vec![
                "prefill: top-1 99.08% (need >= 99.5)",
                "decode: full-vocab KL 1.089e-3 (need <= 1.0e-3)",
                "prefix reuse differs from a cold run"
            ]
        );
    }

    #[test]
    fn the_scored_row_floor_line_is_digested() {
        // the floor line names the row set of each metric and carries the all-rows KL after the gated one
        let t = "prefill floor [4bit]: top-1 (all rows) 91.00% not gated for this class ok | 21-way KL (scored rows, n=32349) 4.100e-2 (need <= 3.0e-2) FAIL [all rows, n=36887: 5.300e-2, reported only] | tail mass (all rows) ours 0.05 vs ref 0.05, excess +0.0001 (|.| <= 0.02) ok | non-finite rows 0 ok => FAIL\n";
        assert_eq!(acc_fail_digest(t), vec!["prefill: 21-way KL (scored rows, n=32349) 4.100e-2 (need <= 3.0e-2)"]);
    }

    #[test]
    fn line_value_finds_prefixed_lines() {
        assert_eq!(line_value("x\n  perplexity: 7.5 over 3\n", "perplexity:").unwrap().trim(), "7.5 over 3");
    }
}
