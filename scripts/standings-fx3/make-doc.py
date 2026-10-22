#!/usr/bin/env python3
"""fx3: fill docs/results/2026-10-22-fx3-spark-standings.tmpl.md from the raw data and write the results doc.

usage: make-doc.py   (from the repo root)
Writes raw/aggregate-{35b,35b-window,8b}.md and docs/results/2026-10-22-fx3-spark-standings.md.
"""
import os, re, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "docs/results/2026-10-22-fx3-spark-standings/raw")
AGG = os.path.join(ROOT, "scripts/standings-fx3/aggregate.py")


def agg(sub, visits):
    env = dict(os.environ, VISITS=visits) if visits else dict(os.environ)
    env.pop("VISITS", None) if not visits else None
    return subprocess.run([sys.executable, "-I", AGG, os.path.join(RAW, sub)], env=env, capture_output=True, text=True, check=True).stdout


def sections(txt):
    parts = re.split(r"(?m)^### ", txt)
    return {p.split("\n", 1)[0]: "### " + p for p in parts if p.strip()}


def head(txt, drop=()):
    """everything except the named sections (by title prefix)"""
    return "\n".join(v for k, v in sections(txt).items() if not any(k.startswith(d) for d in drop)).strip()


t35 = agg("35b", "^p")
t35r = agg("35b", "^r")
t8 = agg("8b", "^p")
for name, t in (("35b", t35), ("35b-window", t35r), ("8b", t8)):
    open(os.path.join(RAW, f"aggregate-{name}.md"), "w").write(t)

s35, s8 = sections(t35), sections(t8)
qual = []
for model, s in (("Qwen3.6-35B-A3B NVFP4 (spark1)", s35), ("Qwen3-8B BF16 (spark2)", s8)):
    q = [v for k, v in s.items() if k.startswith("Quality")]
    if q:
        body = q[0].split("\n", 2)[2] if q[0].count("\n") > 2 else q[0]
        qual.append(f"**{model}**\n\n{body.strip()}")


def conc_row(sec, arm):
    for l in sec.splitlines():
        if l.startswith(f"| {arm} |"):
            cells = [c.strip() for c in l.strip("|").split("|")]
            vals = [c.split("(")[0].strip() for c in cells[2:]]
            return " / ".join(vals)
    return "-"


c8 = [v for k, v in s8.items() if k.startswith("Concurrency")][0]
sub = {
    "@@T35@@": head(t35, drop=("Quality",)),
    "@@T35R@@": head(t35r, drop=("Quality", "Concurrency (")) if False else head(t35r, drop=("Quality",)),
    "@@T8@@": head(t8, drop=("Quality",)),
    "@@Q@@": "\n\n".join(qual),
    "@@C8I@@": conc_row(c8, "intarsia"),
    "@@C8V@@": conc_row(c8, "vllm"),
    "@@C8S@@": conc_row(c8, "sglang"),
    "@@C8L@@": conc_row(c8, "llamacpp16"),
}
note = os.path.join(ROOT, "docs/results/2026-10-22-fx3-spark-standings.t8note.md")
sub["@@T8NOTE@@"] = open(note).read().strip() if os.path.exists(note) else ""
tmpl = open(os.path.join(ROOT, "docs/results/2026-10-22-fx3-spark-standings.tmpl.md")).read()
for k, v in sub.items():
    tmpl = tmpl.replace(k, v)
open(os.path.join(ROOT, "docs/results/2026-10-22-fx3-spark-standings.md"), "w").write(tmpl)
print("left placeholders:", re.findall(r"@@\w+@@", tmpl))
