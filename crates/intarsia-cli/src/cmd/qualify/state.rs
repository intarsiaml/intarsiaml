//! The qualify state file (`state.json` in the report directory): one record per stage, written after every stage so
//! `--resume` skips what is finished. The same records feed the reports.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

pub const STATE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// The gate passed.
    Pass,
    /// The gate ran and the model did not meet it (exit code 1).
    Fail,
    /// The stage did not run (nothing to compare against, tool missing, blocked); the reason says why.
    Skip,
    /// The pipeline itself failed (a tool crashed, timed out, produced nothing readable): exit code 2.
    Error,
    /// The gate ran and failed, but is report-only for this model (a warning; the run does not fail). The reason says
    /// why: `acc-short` of a third-party quant format whose `acc-format` passed (user decision 2026-10-21 17:13).
    Warn,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Skip => "SKIP",
            Verdict::Error => "ERROR",
            Verdict::Warn => "WARN",
        }
    }
    /// A finished stage (PASS or FAIL: the gate ran) is not run again by `--resume`. SKIP and ERROR are: the input a
    /// SKIP was missing (a golden, a reference server, `--skip`) may be there now, and the tool an ERROR hit may be fixed.
    pub fn is_done(self) -> bool {
        matches!(self, Verdict::Pass | Verdict::Fail | Verdict::Warn)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StageRecord {
    pub id: String,
    pub title: String,
    /// `quick` or `full`.
    pub section: String,
    pub verdict: Verdict,
    pub reason: String,
    pub started_unix: u64,
    pub wall_s: f64,
    /// Where the numbers came from: binary, build stamp, command line, model and golden fingerprints, environment.
    pub provenance: Value,
    /// Files this stage produced, relative to the report directory.
    pub artifacts: Vec<String>,
    /// The stage's numbers (the reports and the metrics push read these).
    pub metrics: Value,
    /// Facts later stages need (golden path, device arch, bandwidth, ...); restored on `--resume`.
    pub facts: Map<String, Value>,
    pub notes: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct State {
    pub version: u32,
    /// What this report directory is about; `--resume` refuses a directory made for something else.
    pub key: Value,
    pub created_unix: u64,
    /// Path of quick-report.md once the quick section is written.
    pub quick_report: Option<String>,
    pub stages: Vec<StageRecord>,
    /// Outcomes of the publishing hooks (`--on-quick-done`, `--push-metrics`).
    pub hooks: Vec<Value>,
}

impl State {
    pub fn new(key: Value) -> State {
        State { version: STATE_VERSION, key, created_unix: now(), quick_report: None, stages: vec![], hooks: vec![] }
    }

    pub fn path(dir: &Path) -> PathBuf {
        dir.join("state.json")
    }

    pub fn load(dir: &Path) -> Result<Option<State>> {
        let p = Self::path(dir);
        if !p.exists() {
            return Ok(None);
        }
        let t = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let s: State =
            serde_json::from_str(&t).with_context(|| format!("{} is not a qualify state file", p.display()))?;
        if s.version != STATE_VERSION {
            bail!("{} has state version {}, this build writes {}", p.display(), s.version, STATE_VERSION);
        }
        Ok(Some(s))
    }

    /// Write atomically (temp file then rename) so an interrupted run never leaves half a state file.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let p = Self::path(dir);
        let tmp = dir.join("state.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)? + "\n")?;
        std::fs::rename(&tmp, &p).with_context(|| format!("writing {}", p.display()))?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&StageRecord> {
        self.stages.iter().find(|s| s.id == id)
    }

    /// The record `--resume` keeps for this stage instead of running it again.
    pub fn finished(&self, id: &str) -> Option<&StageRecord> {
        self.get(id).filter(|r| r.verdict.is_done())
    }

    pub fn put(&mut self, rec: StageRecord) {
        match self.stages.iter().position(|s| s.id == rec.id) {
            Some(i) => self.stages[i] = rec,
            None => self.stages.push(rec),
        }
    }
}

pub fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, v: Verdict) -> StageRecord {
        StageRecord {
            id: id.into(),
            title: id.into(),
            section: "quick".into(),
            verdict: v,
            reason: String::new(),
            started_unix: 1,
            wall_s: 0.5,
            provenance: Value::Null,
            artifacts: vec![],
            metrics: Value::Null,
            facts: Map::new(),
            notes: vec![],
        }
    }

    #[test]
    fn state_round_trips_and_put_replaces() {
        let dir = std::env::temp_dir().join(format!("qualify-state-{}", std::process::id()));
        let mut s = State::new(serde_json::json!({"model": "m"}));
        s.put(rec("doctor", Verdict::Error));
        s.put(rec("doctor", Verdict::Pass));
        assert_eq!(s.stages.len(), 1);
        s.save(&dir).unwrap();
        let back = State::load(&dir).unwrap().unwrap();
        assert_eq!(back.get("doctor").unwrap().verdict, Verdict::Pass);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_keeps_pass_and_fail_and_reruns_skip_and_error() {
        let mut s = State::new(serde_json::json!({"model": "m"}));
        for (id, v) in [
            ("doctor", Verdict::Pass),
            ("gate", Verdict::Fail),
            ("acc-short", Verdict::Skip),
            ("bench", Verdict::Error),
        ] {
            s.put(rec(id, v));
        }
        assert_eq!(s.finished("doctor").unwrap().verdict, Verdict::Pass);
        assert_eq!(s.finished("gate").unwrap().verdict, Verdict::Fail);
        // a stage skipped for a missing golden runs again once the golden is supplied
        assert!(s.finished("acc-short").is_none());
        assert!(s.finished("bench").is_none());
        assert!(s.finished("acc-long").is_none());
        // the re-run's record replaces the SKIP
        s.put(rec("acc-short", Verdict::Pass));
        assert_eq!(s.finished("acc-short").unwrap().verdict, Verdict::Pass);
        assert_eq!(s.stages.len(), 4);
    }
}
