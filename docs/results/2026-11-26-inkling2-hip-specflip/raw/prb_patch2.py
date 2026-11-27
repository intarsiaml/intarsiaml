#!/usr/bin/env python3
"""Lane ink-hip-specflip PR B part 2: recipe lines, docs rows, tests. argv[1] = the DFlash2 adapt string from D1 (sim.py), argv[2] = the
D2 numbers line for the recipe comment (may be filled later)."""
import sys

ADAPT = sys.argv[1]
import os
W = os.environ.get("PRB_ROOT", "/opt/code/intarsiaml/.worktrees/ink-hip-specflip/").rstrip("/") + "/"


def sub(p, old, new, cnt=1):
    s = open(W + p).read()
    assert s.count(old) == cnt, (p, old[:80], s.count(old))
    s = s.replace(old, new)
    open(W + p, "w").write(s)


sub("recipes/inkling2.toml", '''sampled_dflash = true
''', f'''sampled_dflash = true
# Flip B (lane ink-hip-specflip, user decisions 2026-11-26 "Both, confirm first" and "Test DFlash2 sampled"): the DFlash2 drafter
# (incoai/Inkling-2-Mini-DFlash2) replaces the DSpark draft in the default above when it is found next to the checkpoint
# (<checkpoint>/../Inkling-2-Mini-DFlash2, or an absolute path here), for greedy requests (k4: 1.112x DSpark k5 over the six cells,
# 4k 0.991 [0.987-0.994], two cards, docs/results/2026-11-26-inkling2-dflash2.md, docs/results/2026-11-26-inkling2-hip-specflip.md) and
# sampled ones (with its own draft length table below, calibrated on its traces; same seed-exact keyed rule). One drafter is loaded
# (KV capacity 175,424 tokens vs DSpark's 177,504 at the defaults; the default contexts do not move). Not found: the DSpark default,
# never an error. Revert: INTARSIA_SPEC_DFLASH2_DEFAULT=0 (the DSpark default), or delete these lines.
dflash2_draft = "Inkling-2-Mini-DFlash2"
dflash2_k = 4
dflash2_sampled_adapt = "{ADAPT}"
''')

# docs: recipes.md rows after sampled_dflash's
s = open(W + "docs/recipes.md").read()
i = s.index("| `[tuning.<backend>] sampled_dflash` |")
j = s.index("\n", i)
rows = (
    "\n| `[tuning.<backend>] dflash2_draft` | string | unset | Lane ink-hip-specflip (flip B, user decisions 2026-11-26): the DFlash2 drafter that replaces "
    "the checkpoint's own DSpark draft in the recipe's speculation default (the `mtp_default_k` default: no `--spec`, one sequence, one rank) when "
    "it is found, for greedy and sampled requests; an absolute directory or a name next to the checkpoint (`<checkpoint>/../<name>`). Found = its "
    "`config.json` is a DFlash2 draft (`dflash_config`, `DFlash2DraftModel`) that loads; otherwise the DSpark default, never an error (one line on "
    "stderr says why). `INTARSIA_SPEC_DFLASH2_DEFAULT=0` restores the DSpark default. Only `recipes/inkling2.toml` sets it, on `hip` "
    "(`Inkling-2-Mini-DFlash2`). |"
    "\n| `[tuning.<backend>] dflash2_k` | integer | unset (the draft's block - 1) | `--spec-k` of the `dflash2_draft` default (a lone `--spec-k N` wins); "
    "Inkling-2 on HIP: 4 (the k the ABBAs measured). |"
    "\n| `[tuning.<backend>] dflash2_sampled_adapt` | string | unset (`sampled_adapt`) | `sampled_adapt` for sampled requests that draft with a DFlash2 "
    "draft (its own cost table: a DFlash2 draft call and its verify rows cost what DSpark's do not); `INTARSIA_SPEC_SAMPLED_ADAPT` wins when set. "
    "Inkling-2 on HIP: the table calibrated on the drafter's traces (docs/results/2026-11-26-inkling2-hip-specflip.md). |"
)
s = s[:j] + rows + s[j:]
s = s.replace(
    "Another draft (`--spec-draft <DFlash2 dir>`) keeps sampled requests plain.",
    "A DFlash2 draft counts too (flip B: the `dflash2_draft` default, or `--spec-draft <DFlash2 dir>`; with `dflash2_sampled_adapt`'s table); "
    "any other draft keeps sampled requests plain.",
)
open(W + "docs/recipes.md", "w").write(s)

# docs: tuning.md row for the revert switch, after INTARSIA_SPEC_DEFAULT's
s = open(W + "docs/tuning.md").read().split("\n")
i = [n for n, l in enumerate(s) if l.startswith("| `INTARSIA_SPEC_DEFAULT` |")][0]
s.insert(i + 1, "| `INTARSIA_SPEC_DFLASH2_DEFAULT` | `crates/intarsia-models/src/load.rs:dflash2_default_draft` | unset: the recipe's "
         "`[tuning.<backend>] dflash2_draft` applies where it is found (Inkling-2 on HIP: `Inkling-2-Mini-DFlash2` next to the checkpoint) | "
         "`0`: the recipe's DSpark default instead of its DFlash2 default (greedy DSpark k5 and sampled DSpark with `sampled_adapt`, flip A; "
         "lane ink-hip-specflip flip B's revert). Other values: the DFlash2 default. Startup line `--spec dflash --spec-draft <dir> --spec-k 4: the "
         "DFlash2 default of recipe ...`, or `DFlash2 default ... not applied: <dir>: <why>; the DSpark default`. | no (greedy and seeded "
         "sampled ids equal `--spec off` with either draft) |")
open(W + "docs/tuning.md", "w").write("\n".join(s))
s = open(W + "docs/tuning.md").read()
s = s.replace("with the checkpoint's own DSpark draft (`spec::sampled_dflash_resolve`): **on for Inkling-2 on HIP**",
              "with the checkpoint's own DSpark draft or a DFlash2 draft (`spec::sampled_dflash_resolve`): **on for Inkling-2 on HIP**")
open(W + "docs/tuning.md", "w").write(s)

# tests appended to tests/inkling2_dflash2.rs
s = open(W + "crates/intarsia-models/tests/inkling2_dflash2.rs").read()
s += r'''
// ------------------------------------------------------------------------------------------------------------------
// lane ink-hip-specflip (flip B): the recipe's DFlash2 default (`[tuning.<backend>] dflash2_draft`), docs/results/2026-11-26-inkling2-hip-specflip.md
// ------------------------------------------------------------------------------------------------------------------

fn recipe_with(backend: &str, f: &dyn Fn(&mut intarsia_core::recipe::BackendTuning)) -> intarsia_core::recipe::Recipe {
    let mut r = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    f(r.tuning.backends.entry(backend.to_string()).or_default());
    r
}

#[test]
fn recipe_dflash2_default_finds_the_drafter_or_falls_back() {
    use intarsia_models::load::{dflash2_default_draft, Dflash2Default};
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    let ink = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    assert_eq!(ink.dflash2_draft("hip"), Some("Inkling-2-Mini-DFlash2"), "recipes/inkling2.toml [tuning.hip] dflash2_draft");
    assert_eq!(ink.dflash2_k("hip"), Some(4));
    assert!(ink.dflash2_sampled_adapt("hip").is_some_and(|a| a != ink.sampled_adapt("hip").unwrap()), "DFlash2's own table");
    let dd = draft_dir();
    let model_dir = dd.parent().unwrap().join("Inkling-2-Mini"); // need not exist: only its parent is read
    for b in ["metal", "cuda", "cpu"] {
        assert_eq!(dflash2_default_draft(&ink, b, &model_dir, None), Dflash2Default::NotSet, "{b}: no DFlash2 default");
    }
    let name = dd.file_name().unwrap().to_str().unwrap().to_string();
    let sib = recipe_with("cpu", &|t| {
        t.dflash2_draft = Some(name.clone());
        t.dflash2_k = Some(4);
    });
    assert_eq!(dflash2_default_draft(&sib, "cpu", &model_dir, None), Dflash2Default::Found(dd.clone(), 4), "a sibling of the checkpoint");
    assert_eq!(dflash2_default_draft(&sib, "cpu", &model_dir, Some("1")), Dflash2Default::Found(dd.clone(), 4));
    assert_eq!(dflash2_default_draft(&sib, "cpu", &model_dir, Some("0")), Dflash2Default::Off, "INTARSIA_SPEC_DFLASH2_DEFAULT=0 reverts");
    let abs = recipe_with("cpu", &|t| t.dflash2_draft = Some(dd.to_str().unwrap().to_string()));
    assert_eq!(dflash2_default_draft(&abs, "cpu", Path::new("/nowhere/at/all"), None), Dflash2Default::Found(dd.clone(), BLOCK - 1), "absolute; k = block - 1");
    // a missing drafter, or a directory that is not a DFlash2 draft: Missing (the DSpark default), never an error
    let missing = recipe_with("cpu", &|t| t.dflash2_draft = Some("no-such-drafter".into()));
    assert!(matches!(dflash2_default_draft(&missing, "cpu", &model_dir, None), Dflash2Default::Missing(..)));
    let tgt = recipe_with("cpu", &|t| t.dflash2_draft = Some(target_dir().unwrap().to_str().unwrap().to_string()));
    assert!(matches!(dflash2_default_draft(&tgt, "cpu", &model_dir, None), Dflash2Default::Missing(..)), "the target is no draft");
    let ds = golden().join("inkling2-tiny-dspark");
    if ds.join("config.json").exists() {
        let r = recipe_with("cpu", &|t| t.dflash2_draft = Some(ds.to_str().unwrap().to_string()));
        assert!(matches!(dflash2_default_draft(&r, "cpu", &model_dir, None), Dflash2Default::Missing(..)), "a DSpark checkpoint is no DFlash2 draft");
    }
}

/// The tiny target + the DFlash2 draft on `recipe` (verify passes for k).
fn model_on(recipe: &intarsia_core::recipe::Recipe, ddir: Option<&Path>, k: usize) -> (Model, Arc<Telemetry>) {
    let (_, dims) = recipe_dims();
    let src = SafeTensors::open_dir(&target_dir().unwrap()).unwrap();
    let dflash = ddir.map(|d| intarsia_models::load::open_dflash(Some(d), false).unwrap());
    let draft = dflash.is_some();
    let tel = Arc::new(Telemetry::new());
    let opts = LoadOptions { max_seq: 256, prefill_chunk: 16, graphs: false, dflash, ..Default::default() };
    let mut m = Model::load(backend(false), recipe, &dims, &src, &opts, tel.clone()).unwrap();
    if draft {
        m.enable_verify(k + 1).unwrap();
        m.configure_plan(&Default::default(), Some(&*intarsia_backend_cpu::CpuBackend::arc())).unwrap();
    }
    (m, tel)
}

/// Flip B's sampled half: where the recipe sets `sampled_dflash`, a DFlash2 draft drafts sampled requests by default (as the DSpark draft
/// does under flip A), with `dflash2_sampled_adapt`'s table, and every seeded request emits `--spec off`'s ids; `INTARSIA_SPEC_SAMPLED_DFLASH`
/// unset (run without it).
#[test]
fn sampled_requests_draft_with_dflash2_under_the_recipe_key() {
    if target_dir().is_none() {
        eprintln!("skipped: no tiny fixtures");
        return;
    }
    if ["INTARSIA_SPEC_SAMPLED_DFLASH", "INTARSIA_SPEC_SAMPLED_ADAPT"].iter().any(|v| std::env::var(v).is_ok_and(|x| !x.is_empty())) {
        eprintln!("skipped: run without INTARSIA_SPEC_SAMPLED_DFLASH / INTARSIA_SPEC_SAMPLED_ADAPT");
        return;
    }
    let b = backend(false);
    let name = b.name().to_string();
    let table = "d=0.3,r=0.3/0.5/0.7/0.9,g=0.95,margin=0.0,drop=0.05,probe=1,hold=16,hold_max=64";
    let r_on = recipe_with(&name, &|t| {
        t.sampled_dflash = Some(true);
        t.dflash2_sampled_adapt = Some(table.into());
    });
    let dd = draft_dir();
    let (off, _) = {
        let (m, tel) = model_on(&r_on, None, 1);
        (LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), eos_ids: vec![], ..Default::default() }, tel.clone()).unwrap(), tel)
    };
    for k in [2usize, 4] {
        let (m, tel) = model_on(&r_on, Some(&dd), k);
        assert!(m.dflash_is_dflash2());
        assert!(m.spec_sampled_dflash(), "k {k}: the recipe key covers a DFlash2 draft");
        assert_eq!(m.spec_sampled_adapt(), intarsia_core::spec::AdaptMode::resolve(&name, Some(table), None), "DFlash2's table");
        let spec = SpecConfig { mode: SpecMode::Dflash, k, k_min: None };
        let on = LocalEngine::new(m, Arc::new(ByteCodec), EngineMeta { id: "tiny".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone()).unwrap();
        for p in prompts() {
            for (t, seed) in [(1.0f32, 1u64), (1.0, 7), (0.7, 3)] {
                let req = || GenRequest { prompt: p.clone(), max_tokens: 30, temperature: t, top_k: 0, top_p: 1.0, seed: Some(seed), ignore_eos: true, ..Default::default() };
                let ids = |e: &LocalEngine| {
                    let mut v = vec![];
                    e.generate(req(), &mut |ev| match ev {
                        GenEvent::Token { id, .. } => v.push(id),
                        GenEvent::Error { message } => panic!("engine error: {message}"),
                        _ => {}
                    })
                    .unwrap();
                    v
                };
                assert_eq!(ids(&on), ids(&off), "k {k} T {t} seed {seed} prompt {p:?}");
            }
        }
        let s = tel.snapshot();
        assert!(s.spec_steps > 0, "k {k}: sampled requests never drafted (vacuous)");
        eprintln!("k {k}: passes {} proposed {} accepted {}", s.spec_steps, s.spec_proposed, s.spec_accepted);
    }
    // without the key a DFlash2 draft keeps sampled requests plain (main before flip A / B)
    let r_none = recipe_with(&name, &|t| t.sampled_dflash = None);
    let (m, _) = model_on(&r_none, Some(&dd), 4);
    assert!(!m.spec_sampled_dflash());
}
'''
open(W + "crates/intarsia-models/tests/inkling2_dflash2.rs", "w").write(s)
