#!/usr/bin/env python3
"""Lane ink-hip-specflip PR B (flip B, option (b)): apply to the worktree after the PR A suite. ADAPT = the D1 table string (argv[1])."""
import sys

ADAPT = sys.argv[1]
import os
W = os.environ.get("PRB_ROOT", "/opt/code/intarsiaml/.worktrees/ink-hip-specflip/").rstrip("/") + "/"


def sub(p, old, new, cnt=1):
    s = open(W + p).read()
    assert s.count(old) == cnt, (p, old[:80], s.count(old))
    s = s.replace(old, new)
    open(W + p, "w").write(s)


# 1. recipe keys
sub("crates/intarsia-core/src/recipe.rs", '''    pub sampled_dflash: Option<bool>,
}
''', '''    pub sampled_dflash: Option<bool>,
    /// Lane ink-hip-specflip (flip B, user decisions 2026-11-26 "Both, confirm first" and "Test DFlash2 sampled"): the DFlash2 drafter
    /// this backend's speculation default uses instead of the checkpoint's own DSpark draft when it is found, for greedy and sampled
    /// requests (one drafter loaded). An absolute directory, or a name resolved next to the checkpoint (`<checkpoint>/../<name>`: the
    /// drafter downloaded beside the model). Not found (or not a DFlash2 draft): the DSpark default as before, never an error.
    /// `INTARSIA_SPEC_DFLASH2_DEFAULT=0` restores the DSpark default. Only `recipes/inkling2.toml` sets it, on HIP.
    pub dflash2_draft: Option<String>,
    /// `--spec-k` of [`BackendTuning::dflash2_draft`]'s default (a lone `--spec-k N` wins). Unset: the block's drafts.
    pub dflash2_k: Option<usize>,
    /// [`BackendTuning::sampled_adapt`] for sampled requests drafting with a DFlash2 draft (its own cost table; the DSpark one is
    /// [`BackendTuning::sampled_adapt`]). Unset: `sampled_adapt`. `INTARSIA_SPEC_SAMPLED_ADAPT` wins when set.
    pub dflash2_sampled_adapt: Option<String>,
}
''')
sub("crates/intarsia-core/src/recipe.rs", '''    pub fn sampled_dflash(&self, backend: &str) -> Option<bool> {
        self.tuning.backends.get(backend).and_then(|b| b.sampled_dflash)
    }
''', '''    pub fn sampled_dflash(&self, backend: &str) -> Option<bool> {
        self.tuning.backends.get(backend).and_then(|b| b.sampled_dflash)
    }

    /// The recipe's `dflash2_draft` for `backend` (`[tuning.<backend>]`); resolved by `intarsia_models::load::dflash2_default_draft`.
    pub fn dflash2_draft(&self, backend: &str) -> Option<&str> {
        self.tuning.backends.get(backend).and_then(|b| b.dflash2_draft.as_deref())
    }

    /// The recipe's `dflash2_k` for `backend`.
    pub fn dflash2_k(&self, backend: &str) -> Option<usize> {
        self.tuning.backends.get(backend).and_then(|b| b.dflash2_k)
    }

    /// The recipe's `dflash2_sampled_adapt` for `backend`, else its `sampled_adapt`.
    pub fn dflash2_sampled_adapt(&self, backend: &str) -> Option<&str> {
        self.tuning.backends.get(backend).and_then(|b| b.dflash2_sampled_adapt.as_deref()).or_else(|| self.sampled_adapt(backend))
    }
''')
sub("crates/intarsia-models/tests/mtp.rs", "sampled_adapt: None, sampled_dflash: None", "sampled_adapt: None, sampled_dflash: None, dflash2_draft: None, dflash2_k: None, dflash2_sampled_adapt: None", 2)

# 2. the model: a DFlash2 draft counts as a measured draft for the recipe's sampled default, with its own table
sub("crates/intarsia-core/src/model/mtp.rs", '''                self.dflash.as_ref().is_some_and(|h| h.dspark.is_some()),''', '''                self.dflash_measured_draft(),''')
sub("crates/intarsia-core/src/model/mtp.rs", '''        let b = self.backend.name();
        crate::spec::AdaptMode::resolve(b, self.recipe.sampled_adapt(b), crate::spec::sampled_adapt_env())
    }''', '''        let b = self.backend.name();
        // lane ink-hip-specflip: a DFlash2 draft has its own cost table (`[tuning.<backend>] dflash2_sampled_adapt`)
        let table = if self.dflash_is_dflash2() { self.recipe.dflash2_sampled_adapt(b) } else { self.recipe.sampled_adapt(b) };
        crate::spec::AdaptMode::resolve(b, table, crate::spec::sampled_adapt_env())
    }

    /// The loaded DFlash-family draft is a DFlash2 draft (no DSpine, not Inkling-2's DSpark).
    pub fn dflash_is_dflash2(&self) -> bool {
        self.dflash.as_ref().is_some_and(|h| h.dspark.is_none() && h.v2.as_ref().is_some_and(|d| d.spine.is_none()))
    }

    /// The draft the recipe's `sampled_dflash` default covers (lane ink-hip-specflip): Inkling-2's own DSpark draft (flip A) or a DFlash2
    /// draft (flip B: sampled DFlash2 measured on HIP, docs/results/2026-11-26-inkling2-hip-specflip.md).
    fn dflash_measured_draft(&self) -> bool {
        self.dflash.as_ref().is_some_and(|h| h.dspark.is_some()) || self.dflash_is_dflash2()
    }''')

# 3. the loader
sub("crates/intarsia-models/src/load.rs", '''/// The checkpoint holds Inkling-2's DSpark draft: `mtp_config` in its config.json and `mtp.safetensors` beside it.''', '''/// What the recipe's DFlash2 default (`[tuning.<backend>] dflash2_draft`, lane ink-hip-specflip flip B) resolves to.
#[derive(Debug, PartialEq)]
pub enum Dflash2Default {
    /// The recipe names no DFlash2 drafter for this backend.
    NotSet,
    /// `INTARSIA_SPEC_DFLASH2_DEFAULT=0`: the DSpark default.
    Off,
    /// Named but not usable here (the directory, why): the DSpark default, never an error.
    Missing(PathBuf, String),
    /// The drafter's directory and k.
    Found(PathBuf, usize),
}

/// The recipe's DFlash2 drafter for `backend` beside the checkpoint `model_dir`: `env` is `INTARSIA_SPEC_DFLASH2_DEFAULT` (`0` turns the
/// default off); the value is an absolute directory or a name next to the checkpoint (`<model_dir>/../<name>`, the parent of the path
/// as given, symlinks kept); found = its config.json is a DFlash2 draft (`dflash_config`, `DFlash2DraftModel`) and `open_dflash` takes it.
/// k: the recipe's `dflash2_k`, else the block's drafts.
pub fn dflash2_default_draft(recipe: &intarsia_core::recipe::Recipe, backend: &str, model_dir: &Path, env: Option<&str>) -> Dflash2Default {
    let Some(name) = recipe.dflash2_draft(backend) else { return Dflash2Default::NotSet };
    if env.is_some_and(|v| v.trim() == "0") {
        return Dflash2Default::Off;
    }
    let p = Path::new(name);
    let d = if p.is_absolute() {
        p.to_path_buf()
    } else {
        let m = std::path::absolute(model_dir).unwrap_or_else(|_| model_dir.to_path_buf());
        m.parent().map_or_else(|| p.to_path_buf(), |par| par.join(p))
    };
    let cfg = match std::fs::read(d.join("config.json")) {
        Ok(b) => b,
        Err(e) => return Dflash2Default::Missing(d, format!("config.json: {e}")),
    };
    let v: Value = match serde_json::from_slice(&cfg) {
        Ok(v) => v,
        Err(e) => return Dflash2Default::Missing(d, format!("config.json: {e}")),
    };
    let dflash2 = v.get("dflash_config").is_some()
        && v.get("architectures").and_then(Value::as_array).is_some_and(|a| a.iter().any(|x| x.as_str() == Some("DFlash2DraftModel")));
    if !dflash2 {
        return Dflash2Default::Missing(d, "not a DFlash2 draft (dflash_config + DFlash2DraftModel)".into());
    }
    match open_dflash(Some(d.as_path()), false) {
        Ok(dl) if dl.config.v2 && dl.config.spine.is_none() && dl.config.dspark.is_none() => {
            Dflash2Default::Found(d, recipe.dflash2_k(backend).unwrap_or(dl.config.block.saturating_sub(1)).max(1))
        }
        Ok(_) => Dflash2Default::Missing(d, "not a DFlash2 draft".into()),
        Err(e) => Dflash2Default::Missing(d, e.to_string()),
    }
}

/// The checkpoint holds Inkling-2's DSpark draft: `mtp_config` in its config.json and `mtp.safetensors` beside it.''')
sub("crates/intarsia-models/src/load.rs", '''    let dspark_dflt_opts: EngineOptions;
    let mut opts = opts;
    let mut dspark_dflt = false;
''', '''    let dspark_dflt_opts: EngineOptions;
    let dflash2_dflt_opts: EngineOptions;
    let mut opts = opts;
    let mut dspark_dflt = false;
    // lane ink-hip-specflip (flip B): the recipe's DFlash2 drafter, when found, replaces the DSpark draft of the default below for
    // greedy and sampled requests (one drafter loaded); not found, or INTARSIA_SPEC_DFLASH2_DEFAULT=0: the DSpark default as before
    let mut dflash2_dflt = false;
    if dspark_default(opts, &recipe, backend_name, backends.len(), local).is_some() {
        match dflash2_default_draft(&recipe, backend_name, dir, std::env::var("INTARSIA_SPEC_DFLASH2_DEFAULT").ok().as_deref()) {
            Dflash2Default::Found(d, k2) => {
                let k2 = if opts.spec_k_given { opts.spec.k } else { k2 };
                eprintln!(
                    "--spec dflash --spec-draft {} --spec-k {k2}{}: the DFlash2 default of recipe {} on {backend_name} ([tuning.{backend_name}] \\
                     dflash2_draft; greedy and sampled requests at one sequence; INTARSIA_SPEC_DFLASH2_DEFAULT=0 restores the DSpark default, \\
                     --spec off or INTARSIA_SPEC_DEFAULT=0 turns speculation off)",
                    d.display(),
                    if opts.spec_k_given { " (k from --spec-k)" } else { "" },
                    recipe.model.family
                );
                let mut o = opts.clone();
                o.spec = intarsia_core::spec::SpecConfig { mode: intarsia_core::spec::SpecMode::Dflash, k: k2, k_min: None };
                o.spec_draft = Some(d);
                dflash2_dflt_opts = o;
                opts = &dflash2_dflt_opts;
                dflash2_dflt = true;
            }
            Dflash2Default::Off => {
                eprintln!("DFlash2 default of recipe {} on {backend_name} off (INTARSIA_SPEC_DFLASH2_DEFAULT=0): the DSpark default", recipe.model.family)
            }
            Dflash2Default::Missing(d, why) => eprintln!(
                "DFlash2 default of recipe {} on {backend_name} not applied: {}: {why}; the DSpark default",
                recipe.model.family,
                d.display()
            ),
            Dflash2Default::NotSet => {}
        }
    }
''')
sub("crates/intarsia-models/src/load.rs", '''    if let Some(k) = dspark_default(opts, &recipe, backend_name, backends.len(), local) {
        if dspark_draft_present(&cfg, dir) {''', '''    if let Some(k) = dspark_default(opts, &recipe, backend_name, backends.len(), local).filter(|_| !dflash2_dflt) {
        if dspark_draft_present(&cfg, dir) {''')
