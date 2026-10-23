#!/usr/bin/env python3
"""List every benchmark cell where IntarsiaML does not win (or could win a tie) and write a lane brief for each.

Reads the model-benchmarks composer data (read-only):
  composer/data/regimes.json   one entry per (setup, lane, phase, bucket) cell with every contender
  composer/data/catalog.json   techniques (how a winner is fast), matched to cells the way build_regimes.py does

Rules (composer/README.md "Rules"):
  * Same setup only, and a gap counts only against a rival on IntarsiaML's OWN quant. The cell's
    same_quant_* fields are relative to the WINNER's quant, so this script recomputes from contenders[]:
    for each IntarsiaML config, the best non-IntarsiaML contender with the same non-empty quant.
    margin = rival / intarsia - 1, as build_regimes.py does for a win (build_regimes.py:265, :267-269).
  * Noise = max(floor, both spreads); floors decode 5 %, prefill 10 % under 16k and 5 % from 16k,
    concurrent 10 % (build_regimes.py:47, :110-113, :266). Inside the noise the verdict is a tie.
  * The speculative-decoding lane is skipped unless --lane spec.
  * An empty quant ("") is unknown, never a match: such cells are listed apart.

Ranking: score = margin_pct x phase weight. The weights are a judgement, not a measurement:
  decode 1.0, prefill 1.0, concurrent 0.5 (aggregate decode over N streams repeats the decode lever and is
  noisier). Override with --weight phase=value.

Usage:
  python3 tools/composer/gap_to_lane.py                  # writes docs/composer/gaps.md
  python3 tools/composer/gap_to_lane.py --json           # JSON on stdout, nothing written
  python3 tools/composer/gap_to_lane.py --json out.json  # JSON to a file as well as the markdown
  python3 tools/composer/gap_to_lane.py --lane spec
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MB = Path("/opt/code/model-benchmarks")
DEFAULT_REGIMES = MB / "composer" / "data" / "regimes.json"
DEFAULT_CATALOG = MB / "composer" / "data" / "catalog.json"
DEFAULT_EXTRACTION = MB / "composer" / "catalog" / "extraction-2026-10-22.json"
DEFAULT_OUT = REPO / "docs" / "composer" / "gaps.md"

ENGINE = "IntarsiaML"
# build_regimes.py:47 and :110-113
FLOOR = {"decode": 5.0, "prefill_short": 10.0, "prefill_long": 5.0, "concurrent": 10.0}
LONG_BUCKETS = ("16k", "32k", "64k+")
DEFAULT_WEIGHTS = {"decode": 1.0, "prefill": 1.0, "concurrent": 0.5}

# backend crate / kernel dir by hardware prefix (docs/backends)
BACKEND_OF = [("DGX Spark", "cuda"), ("2x DGX Spark", "cuda"), ("Radeon", "hip"), ("M5", "metal"), ("Apple", "metal")]

# Cells the model-benchmarks session asked us not to build plain lanes against (coordinator message, 2026-10-22).
# Each rule: (reason, predicate(cell)).
EXCLUDE_RUN = "20261022-074221-adhoc-tc-off-r1"
EXCLUSIONS = [
    ("did not reproduce: its own reps ran 141.6-150.1 tok/s, a 28% spread (the same config repeated gives 141.2-141.8); "
     "extraction-2026-10-22.json, intarsiaml unexplained",
     lambda c: any(k["run_id"] == EXCLUDE_RUN for k in c["contenders"])),
    ("labelled plain but really spec decode (TensorFold DSV4.1 TP2 decode)",
     lambda c: c["family"] == "DeepSeek V4.1 Flash" and c["hardware"].startswith("2x DGX Spark")
     and c["phase"] == "decode"),
    ("labelled plain but really spec decode (Flare Bolt decode 1k/16k)",
     lambda c: c["family"].startswith("Nemotron 3.5 Bolt") and c["phase"] == "decode" and c["bucket"] in ("1k", "16k")),
]
STOP = set("the a an of and or to in for on with by is are at as from per vs it its this that into over than not "
           "only each via using use one two all".split())


# ----------------------------------------------------------------------------------------------- rules

def floor_for(phase, bucket):
    """build_regimes.py:110-113."""
    if phase == "prefill":
        return FLOOR["prefill_long"] if bucket in LONG_BUCKETS else FLOOR["prefill_short"]
    return FLOOR[phase]


def techniques_for(engine, cell, catalog):
    """Catalog entries that explain this engine's number in this cell.

    An entry with a `cells` field (exact regimes.json ids) matches when the cell id is listed and the entry is for the
    engine; an entry without `cells` falls back to the build_regimes.py:192-200 predicate (engine, phase, bucket,
    family, hardware), with the engine a parameter because the same-quant rival is not always the cell's winner."""
    out = []
    for e in catalog:
        if e.get("cells"):
            if cell["id"] in e["cells"] and e["engine"] == engine:
                out.append(e)
            continue
        hw = e.get("hardware") or ""
        if (e["engine"] == engine and e["phase"] == cell["phase"]
                and (not e.get("buckets") or cell["bucket"] in e["buckets"])
                and (not e.get("family") or e["family"] == cell["family"])
                and (not hw or cell["hardware"] in hw or hw in cell["hardware"])):
            out.append(e)
    return out


def exclusion(cell):
    for reason, pred in EXCLUSIONS:
        if pred(cell):
            return reason
    return None


def weigh(phase, weights):
    return weights.get(phase, 1.0)


def analyse(regimes, catalog, lane="plain", weights=None, extraction=None):
    """Returns dict with gaps, ties, unmatched_quant, unknown_quant, measure_first (all lists of dicts)."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    all_cells = [c for c in regimes if c["lane"] == lane]
    excluded = [{"cell": c["id"], "reason": r} for c in all_cells for r in [exclusion(c)] if r
                and any(k["engine"] == ENGINE for k in c["contenders"])]
    cells = [c for c in all_cells if not exclusion(c)]
    extraction = extraction or {}
    gaps, ties, unmatched, unknown, measure_first = [], [], [], [], []

    intarsia_setups = {c["setup"] for c in cells if any(k["engine"] == ENGINE for k in c["contenders"])}
    intarsia_hw = {c["hardware"] for c in cells if c["setup"] in intarsia_setups}
    intarsia_fam = {c["family"] for c in cells if c["setup"] in intarsia_setups}

    for c in cells:
        mine = [k for k in c["contenders"] if k["engine"] == ENGINE]
        rivals = [k for k in c["contenders"] if k["engine"] != ENGINE]
        if not mine:
            continue
        for me in mine:
            if not rivals:
                measure_first.append(_measure_entry(c, me, "solo", cells))
                continue
            if not me["quant"]:
                best = max(rivals, key=lambda k: k["value"])
                unknown.append(_base(c, me, best, None, catalog, weights, note="IntarsiaML quant is empty in the data", extraction=extraction))
                continue
            same = [k for k in rivals if k["quant"] == me["quant"]]
            if not same:
                best = max(rivals, key=lambda k: k["value"])
                if best["value"] > me["value"]:
                    unmatched.append(_base(c, me, best, None, catalog, weights,
                                           note=f"no rival on {me['quant']}; best rival uses {best['quant'] or 'unknown quant'}",
                                           extraction=extraction))
                continue
            rival = max(same, key=lambda k: k["value"])
            e = _base(c, me, rival, True, catalog, weights, extraction=extraction)
            if e["verdict"] == "behind":
                gaps.append(e)
            elif e["verdict"] == "tie":
                ties.append(e)
            # verdict "ahead": IntarsiaML wins on its own quant, nothing to list

    # unmeasured: a cell of an IntarsiaML setup where only competitors have data, plus setups on hardware and
    # families IntarsiaML has run where it has no cell at all
    by_setup = collections.defaultdict(list)
    for c in cells:
        by_setup[c["setup"]].append(c)
    for setup, cs in by_setup.items():
        has_me = setup in intarsia_setups
        if not has_me and not (cs[0]["hardware"] in intarsia_hw and cs[0]["family"] in intarsia_fam):
            continue
        for c in cs:
            if any(k["engine"] == ENGINE for k in c["contenders"]):
                continue
            best = max(c["contenders"], key=lambda k: k["value"])
            measure_first.append(_measure_entry(c, None, "unmeasured" if has_me else "unmeasured-setup", cells, best))

    key = lambda e: -e["score"]
    gaps.sort(key=key)
    ties.sort(key=key)
    unmatched.sort(key=key)
    unknown.sort(key=key)
    measure_first.sort(key=lambda e: (e["kind"], e["setup"], e["phase"], e["bucket"]))
    return {"lane": lane, "weights": weights, "excluded": excluded, "gaps": gaps, "ties": ties, "unmatched_quant": unmatched,
            "unknown_quant": unknown, "measure_first": measure_first}


def _base(c, me, rival, matched, catalog, weights, note=None, extraction=None):
    phase, bucket = c["phase"], c["bucket"]
    noise = max(floor_for(phase, bucket), me.get("spread") or 0.0, rival.get("spread") or 0.0)
    margin = round((rival["value"] / me["value"] - 1) * 100, 1)
    if matched:
        verdict = "behind" if margin > noise else ("ahead" if -margin > noise else "tie")
    else:
        verdict = "not-comparable"
    w = weigh(phase, weights)
    techs = techniques_for(rival["engine"], c, catalog)
    if c["winner_engine"] != rival["engine"]:
        techs += [t for t in techniques_for(c["winner_engine"], c, catalog) if t not in techs]
    return {
        "cell": c["id"], "setup": c["setup"], "hardware": c["hardware"], "family": c["family"], "lane": c["lane"],
        "phase": phase, "bucket": bucket, "verdict": verdict, "note": note,
        "margin_pct": margin, "noise_pct": round(noise, 1), "margin_over_noise": round(margin / noise, 2) if noise else None,
        "phase_weight": w, "score": round(max(margin, 0.0) * w, 2) if matched else 0.0,
        "intarsia": _contender(me), "rival": _contender(rival),
        "cell_winner": c["winner"], "cell_winner_value": c["winner_value"], "cell_winner_run": c["winner_run"],
        "cell_winner_quality": c["winner_quality"],
        "evidence_runs": sorted({me["run_id"], rival["run_id"], c["winner_run"]}),
        "techniques": [_technique(t) for t in techs],
        "extraction": _extraction_status(c, techs, extraction or {}),
    }


def load_extraction(path):
    """composer/catalog/extraction-*.json: per engine, `unexplained` cells (with the reason) and `rejected` entries."""
    try:
        x = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    unexplained, rejected = {}, []
    for pe in x.get("per_engine", []):
        for u in pe.get("unexplained", []):
            unexplained[u["cell"]] = {"engine": pe["engine"], "reason": u["reason"]}
        for r in pe.get("rejected", []):
            rejected.append({"engine": pe["engine"], "id": r["id"], "reason": r["reason"]})
    return {"unexplained": unexplained, "rejected": rejected, "workflow": x.get("workflow"), "date": x.get("date")}


def _extraction_status(cell, techs, extraction):
    """What the extraction run says about a cell with no catalog technique: listed as unexplained (with its reason),
    rejected entries that name the cell's family, or simply never examined."""
    if techs or not extraction:
        return None
    cid = cell["id"].split(" | ")
    key = f'{cell["hardware"]} · {cell["family"]} | {cell["lane"]} | {cell["phase"]} | {cell["bucket"]}'
    u = extraction.get("unexplained", {}).get(key)
    rej = [r for r in extraction.get("rejected", []) if cell["family"].lower() in (r["id"] + " " + r["reason"]).lower()]
    return {"unexplained": u, "rejected_mentioning_family": [{"id": r["id"], "engine": r["engine"]} for r in rej[:4]],
            "status": "listed as unexplained by the extraction run" if u else
                      "not listed by the extraction run (no entry, no explanation: unexamined)"}


def _contender(k):
    keep = ("config", "engine", "quant", "tp", "value", "spread", "run_id", "version", "n_runs", "quality", "noisy")
    return {f: k.get(f) for f in keep}


def _technique(t):
    keep = ("id", "engine", "version", "hardware", "family", "arch", "phase", "buckets", "cells", "technique", "mechanism",
            "impact", "preconditions", "transfer_notes", "sources", "source_verified", "evidence_runs", "confidence",
            "verify_verdict", "verify_note")
    return {f: t.get(f) for f in keep}


def _measure_entry(c, me, kind, cells, best=None):
    setup_rivals = sorted({k["config"] for x in cells if x["setup"] == c["setup"] for k in x["contenders"]
                           if k["engine"] != ENGINE})
    if kind == "solo":
        why = ("IntarsiaML is the only contender in this cell"
               + (f"; other cells of the setup have {', '.join(setup_rivals)}" if setup_rivals
                  else "; no competitor has data anywhere in this setup"))
    elif kind == "unmeasured":
        why = f"IntarsiaML has other cells in this setup but none here; best competitor {best['config']} = {best['value']}"
    else:
        why = f"IntarsiaML has run on this hardware and this family but never in this setup; best competitor {best['config']} = {best['value']}"
    return {"kind": kind, "cell": c["id"], "setup": c["setup"], "hardware": c["hardware"], "family": c["family"],
            "phase": c["phase"], "bucket": c["bucket"], "why": why,
            "intarsia": _contender(me) if me else None,
            "competitors": [_contender(k) for k in c["contenders"] if k["engine"] != ENGINE],
            "competitor_has_data": bool(setup_rivals) if kind == "solo" else True}


# ------------------------------------------------------------------------------- file heuristics

def backend_for(hardware):
    for prefix, be in BACKEND_OF:
        if hardware.startswith(prefix):
            return be
    return None


PHASE_KW = {"prefill": ["prefill", "chunk", "gemm"], "decode": ["decode", "graph", "gemv"],
            "concurrent": ["batch", "concurrent", "stream", "scheduler"]}


def keywords(entry):
    """Words from the technique ids and one-liners plus the phase; a heuristic, not an analysis. A cell with no
    catalog technique falls back to a few words per phase."""
    if not entry["techniques"]:
        return PHASE_KW.get(entry["phase"], [entry["phase"]])
    text = " ".join([entry["phase"]] + [t["id"].replace("-", " ") + " " + (t.get("technique") or "")
                                        for t in entry["techniques"]])
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text)]
    words = [w for w in words if w not in STOP and w not in ("vllm", "gb10", "sm_121", "nvfp4", "intarsiaml")]
    return list(dict.fromkeys(words))[:14]


_FILES = {}


def source_files(repo):
    if repo not in _FILES:
        out = []
        for root in ("crates", "kernels", "recipes"):
            for dirpath, dirs, files in os.walk(repo / root):
                dirs[:] = [d for d in dirs if d not in ("target", ".git", "tests", "__pycache__")]
                for f in files:
                    if f.endswith((".rs", ".cu", ".cuh", ".hip", ".metal", ".toml", ".h")):
                        out.append(Path(dirpath) / f)
        _FILES[repo] = out
    return _FILES[repo]


def likely_files(entry, repo, top=6):
    """Heuristic: files in the cell's backend crate, its kernels, the model crate and core whose path or text
    mention the technique keywords. Labelled as a heuristic in the output."""
    be = backend_for(entry["hardware"])
    kws = keywords(entry)
    if not kws:
        return []
    scored = []
    for p in source_files(repo):
        rel = p.relative_to(repo).as_posix()
        in_scope = (be and (f"intarsia-backend-{be}" in rel or rel.startswith(f"kernels/{be}"))
                    or "intarsia-models" in rel or "intarsia-core" in rel or rel.startswith("recipes/"))
        if not in_scope:
            continue
        try:
            text = p.read_text(errors="ignore").lower()
        except OSError:
            continue
        s = 0
        hit = []
        for k in kws:
            n = text.count(k)
            pth = rel.lower().count(k)
            if n or pth:
                hit.append(k)
                s += 5 * pth + min(n, 20) ** 0.5
        if len(hit) >= 2:
            s += len(hit)
            scored.append((s, rel, hit))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"path": rel, "matched": hit[:6]} for _, rel, hit in scored[:top]]


def experiment(e):
    me, rv = e["intarsia"], e["rival"]
    steps = []
    if (me.get("quality") or "").startswith("drop"):
        steps.append(f"First settle quality: IntarsiaML is `{me['quality']}` on this setup, so re-run that bench before "
                     "treating the speed gap as a target (a faster path on a degraded output proves nothing).")
    t = e["techniques"][0] if e["techniques"] else None
    if e["phase"] == "concurrent" and backend_for(e["hardware"]) not in (None, "metal"):
        steps.append("Check batching first: `serve --max-batch` (continuous batching) is Metal only and refused on other "
                     "backends (crates/intarsia-cli/src/cmd/serve.rs, `--max-batch`), so on this backend streams queue behind "
                     "one sequence and the gap may be structural. Confirm by comparing aggregate tok/s at 1 and N streams.")
    if t:
        steps.append(f"Microbenchmark the lever first: {t.get('transfer_notes') or t['technique']}")
        steps.append(f"Then ship it opt-in (default bitwise unchanged) and A/B it with localeval, interleaved x3, "
                     f"in an exclusive fleetq slot at {e['phase']} {e['bucket']}, plus the accuracy gate if output bits change.")
    else:
        steps.append(f"No catalog technique explains {rv['engine']}'s lead. Cheapest: profile one {e['phase']} {e['bucket']} "
                     f"step of both engines on the same box (per-op time), find the op that differs, then add a catalog entry "
                     f"with a file:line source before writing code.")
    return steps


# ------------------------------------------------------------------------------------ rendering

def fmt_c(k):
    sp = f", spread {k['spread']}%" if k.get("spread") is not None else ""
    return f"{k['config']} = {k['value']}{sp}, quality `{k['quality']}`, run `{k['run_id']}`"


def brief(e, repo, n):
    lines = [f"### {n}. {e['setup']} · {e['phase']} {e['bucket']} ({e['verdict']}, score {e['score']})", ""]
    lines += [
        f"- **Setup:** {e['setup']} (lane {e['lane']}); cell `{e['cell']}`",
        f"- **Phase / bucket:** {e['phase']} / {e['bucket']}",
        f"- **Rival on IntarsiaML's quant ({e['intarsia']['quant']}):** {fmt_c(e['rival'])}",
        f"- **IntarsiaML:** {fmt_c(e['intarsia'])}",
        f"- **Margin vs noise:** rival ahead by {e['margin_pct']}% against a noise bound of {e['noise_pct']}% "
        f"({e['margin_over_noise']}x noise), verdict **{e['verdict']}**; phase weight {e['phase_weight']}, score {e['score']}",
    ]
    if e["cell_winner"] != e["rival"]["config"]:
        lines.append(f"- **Cell winner:** {e['cell_winner']} = {e['cell_winner_value']} (quality `{e['cell_winner_quality']}`, "
                     f"run `{e['cell_winner_run']}`), on a different quant or engine than the same-quant rival")
    lines.append(f"- **Quality status:** IntarsiaML `{e['intarsia']['quality']}`, rival `{e['rival']['quality']}`")
    lines.append("- **Evidence runs:** " + ", ".join(f"`{r}`" for r in e["evidence_runs"]))
    if e["techniques"]:
        lines.append("- **Catalog techniques:**")
        for t in e["techniques"]:
            src = "; ".join(s.get("path") or f"unanimis {s.get('record')}" + (f":{s['lines']}" if s.get("lines") else "")
                            for s in t["sources"])
            conf = ""
            if t.get("confidence") or t.get("verify_verdict"):
                conf = f" [confidence {t.get('confidence') or '?'}, verify {t.get('verify_verdict') or '?'}]"
            lines += [f"  - `{t['id']}` ({t['engine']}){conf}: {t['technique']}",
                      f"    - mechanism: {t['mechanism']}",
                      f"    - impact: {t['impact']}",
                      f"    - preconditions: {'; '.join(t.get('preconditions') or [])}",
                      f"    - sources: {src} ({t.get('source_verified') or 'unverified'})"]
            if t.get("verify_note"):
                lines.append(f"    - verify note: {t['verify_note']}")
    else:
        x = e.get("extraction") or {}
        lines.append("- **Catalog techniques:** none matches this cell; " + (x.get("status") or "extraction file not read"))
        if x.get("unexplained"):
            lines.append(f"  - extraction note ({x['unexplained']['engine']}): {x['unexplained']['reason']}")
        if x.get("rejected_mentioning_family"):
            lines.append("  - rejected entries naming this family (not usable): "
                         + ", ".join(f"`{r['id']}`" for r in x["rejected_mentioning_family"]))
    files = likely_files(e, repo)
    if files:
        lines.append("- **IntarsiaML files most likely involved** (heuristic: keyword hits in the backend, model and core "
                     "crates, not an analysis):")
        lines += [f"  - `{f['path']}` ({', '.join(f['matched'])})" for f in files]
    else:
        lines.append("- **IntarsiaML files most likely involved:** no keyword match (heuristic)")
    lines.append("- **Cheapest experiment:**")
    lines += [f"  {i}. {s}" for i, s in enumerate(experiment(e), 1)]
    lines.append("")
    return lines


def table(rows, header, fn):
    if not rows:
        return ["(none)", ""]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in fn(r)) + " |" for r in rows]
    return out + [""]


def render(res, repo, src):
    L = ["# IntarsiaML gaps by cell", "",
         f"Generated by `tools/composer/gap_to_lane.py` from `{src['regimes']}` (sha256 {src['regimes_sha']}) and "
         f"`{src['catalog']}` (sha256 {src['catalog_sha']}); rejected and unexplained items from `{src['extraction']}`. "
         f"Techniques match on an entry's `cells` (exact regimes.json ids) when it has them, else on engine, phase, bucket, family, "
         f"hardware. Lane: **{res['lane']}**."
         + (" The spec lane is skipped by default (`--lane spec`)." if res["lane"] == "plain" else ""),
         "", "Read-only on model-benchmarks. Refresh: rerun the script after `build_regimes.py` has been run there.", "",
         "## How cells are judged", "",
         "- Same setup only. A gap counts only against a rival on **IntarsiaML's own quant**; the cell's `same_quant_*` "
         "fields are relative to the winner's quant, so the margin is recomputed from `contenders[]`: "
         "`rival / IntarsiaML - 1`.",
         "- Noise = max(floor, both spreads); floors: decode 5%, prefill 10% under 16k and 5% from 16k, concurrent 10% "
         "(`build_regimes.py:47`). Inside the noise the cell is a tie.",
         "- Rank score = margin % x phase weight. Weights (a judgement, not a measurement): "
         + ", ".join(f"{k} {v}" for k, v in res["weights"].items())
         + ". Concurrent is half-weight because aggregate decode over N streams repeats the decode lever and is noisier.",
         "- An empty quant is unknown, never a match; those cells are listed apart.", "",
         "## Summary", "",
         f"- Gaps (rival ahead beyond noise, same quant): **{len(res['gaps'])}**",
         f"- Ties that IntarsiaML could win: **{len(res['ties'])}**",
         f"- Rival ahead on a different quant only (not comparable): {len(res['unmatched_quant'])}",
         f"- IntarsiaML quant unknown: {len(res['unknown_quant'])}",
         f"- Measure first (solo or unmeasured, a competitor has data): {len(res['measure_first'])}", ""]
    if res["gaps"]:
        L += ["## Top gaps", ""]
        L += table(res["gaps"][:10], ["#", "setup", "cell", "rival", "margin", "noise", "score", "IntarsiaML quality", "technique ids"],
                   lambda e: [res["gaps"].index(e) + 1, e["setup"], f"{e['phase']} {e['bucket']}", e["rival"]["config"],
                              f"{e['margin_pct']}%", f"{e['noise_pct']}%", e["score"], f"`{e['intarsia']['quality']}`",
                              ", ".join(f"`{t['id']}`" for t in e["techniques"]) or "none"])
    if res.get("excluded"):
        L += ["## Excluded cells", "", "Skipped on the model-benchmarks session's instruction (not reproducible, or labelled plain "
              "but really speculative):", ""]
        L += table(res["excluded"], ["cell", "reason"], lambda x: [f"`{x['cell']}`", x["reason"]])
    L += ["## Lane briefs: gaps", ""]
    for i, e in enumerate(res["gaps"], 1):
        L += brief(e, repo, i)
    if not res["gaps"]:
        L += ["(none)", ""]
    L += ["## Ties IntarsiaML could win", "",
          "The margin is inside the noise bound, so the cell is undecided. A cheaper measurement (more reps, an exclusive "
          "slot) may settle it before any engine work.", ""]
    L += table(res["ties"], ["setup", "cell", "rival", "margin (rival ahead if +)", "noise", "score", "techniques"],
               lambda e: [e["setup"], f"{e['phase']} {e['bucket']}", e["rival"]["config"], f"{e['margin_pct']}%",
                          f"{e['noise_pct']}%", e["score"], ", ".join(f"`{t['id']}`" for t in e["techniques"]) or "none"])
    L += ["## Rival ahead on a different quant only", "",
          "Not a same-quant comparison, so not an engine gap by the composer's rules; it is a quant lesson or a measure-first "
          "candidate (run the rival's engine on IntarsiaML's quant).", ""]
    L += table(res["unmatched_quant"], ["setup", "cell", "IntarsiaML", "best rival", "ratio", "note"],
               lambda e: [e["setup"], f"{e['phase']} {e['bucket']}", f"{e['intarsia']['config']} = {e['intarsia']['value']}",
                          f"{e['rival']['config']} = {e['rival']['value']}", f"{e['margin_pct']}%", e["note"]])
    L += ["## IntarsiaML quant unknown", ""]
    L += table(res["unknown_quant"], ["setup", "cell", "IntarsiaML", "best rival", "note"],
               lambda e: [e["setup"], f"{e['phase']} {e['bucket']}", f"{e['intarsia']['config']} = {e['intarsia']['value']}",
                          f"{e['rival']['config']} = {e['rival']['value']}", e["note"]])
    L += ["## Measure first", "",
          "Cells where IntarsiaML is solo or unmeasured and a competitor has data. A missing cell is a benchmark to run, "
          "not a loss (`composer/README.md`).", ""]
    L += table(res["measure_first"], ["kind", "setup", "cell", "why"],
               lambda e: [e["kind"], e["setup"], f"{e['phase']} {e['bucket']}", e["why"]])
    return "\n".join(L) + "\n"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regimes", default=str(DEFAULT_REGIMES))
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ap.add_argument("--extraction", default=str(DEFAULT_EXTRACTION), help="composer/catalog/extraction-*.json (rejected and unexplained)")
    ap.add_argument("--lane", default="plain", choices=["plain", "spec"])
    ap.add_argument("--weight", action="append", default=[], metavar="PHASE=W", help="override a phase weight (repeatable)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="markdown output (default docs/composer/gaps.md)")
    ap.add_argument("--json", nargs="?", const="-", default=None, metavar="PATH",
                    help="write JSON to PATH, or to stdout when no PATH is given (stdout JSON skips the markdown)")
    ap.add_argument("--repo", default=str(REPO), help="intarsiaml checkout used for the file heuristic")
    a = ap.parse_args(argv)
    weights = {}
    for w in a.weight:
        k, _, v = w.partition("=")
        weights[k] = float(v)
    regimes = json.loads(Path(a.regimes).read_text())
    catalog = json.loads(Path(a.catalog).read_text())
    res = analyse(regimes, catalog, a.lane, weights, load_extraction(a.extraction))
    src = {"regimes": a.regimes, "regimes_sha": sha(a.regimes), "catalog": a.catalog, "catalog_sha": sha(a.catalog),
           "extraction": a.extraction}
    repo = Path(a.repo)
    res["source"] = src
    for e in res["gaps"] + res["ties"] + res["unmatched_quant"] + res["unknown_quant"]:
        e["likely_files_heuristic"] = likely_files(e, repo)
        e["cheapest_experiment"] = experiment(e)
    if a.json == "-":
        json.dump(res, sys.stdout, indent=1)
        print()
        return 0
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res, repo, src))
    print(f"wrote {out}: {len(res['gaps'])} gaps, {len(res['ties'])} ties, {len(res['unmatched_quant'])} quant-only, "
          f"{len(res['unknown_quant'])} unknown-quant, {len(res['measure_first'])} measure-first")
    return 0


if __name__ == "__main__":
    sys.exit(main())
