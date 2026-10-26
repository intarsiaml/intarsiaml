#!/usr/bin/env python3
"""Lane dfh: per-category table of a cat-window (scripts/dfh/cat-window.sh) from localeval results and the server's
speculation counters. usage: python3 -I tools/dfh/cat_table.py <raw window dir> [more raw dirs ...] (results under /opt/code/deepeval/results;
several dirs pool their rounds, e.g. two speed windows and the radiance reference run)
Checks every case for exactly GEN completion tokens and every arm for the same set of rows; prints decode tok/s pooled per
category (tokens / sum of decode times), x vs off, and tokens per verify pass from the counters."""
import glob, json, os, re, statistics as st, sys
raws = sys.argv[1:]; res = "/opt/code/deepeval/results"
W = "+".join(os.path.basename(r.rstrip("/")) for r in raws); GEN = 512
runs = []
for raw in raws:
    steps = open(os.path.join(raw, "steps.log")).read()
    runs += re.findall(r"round (\d+) arm (\S+) (\S+): localeval exit (\d+) (results/\S+)", steps)
data, problems = {}, []
for r, arm, cat, rc, path in runs:
    cases = json.load(open(os.path.join(os.path.dirname(res), path)))["datasets"][0]["cases"]
    for c in cases:
        n = c["speed"]["completion_tokens"]
        if n != GEN:
            problems.append(f"{arm} {cat} r{r} {c['name']}: {n} tokens")
    data.setdefault((arm, cat), []).append(cases)
spec = {}
for raw in raws:
    mt = os.path.join(raw, "spec-metrics.txt")
    if not os.path.exists(mt):
        continue
    lines = open(mt).read().splitlines()
    for a, b in zip(lines[0::2], lines[1::2]):
        arm, cat, r = a.split()[:3]
        g = lambda s, k: float(re.search(re.escape(k) + r"_total(?:\{[^}]*\})? (\S+)", s).group(1)) if re.search(re.escape(k) + r"_total(?:\{[^}]*\})? (\S+)", s) else 0.0
        d = {k: g(b, k) - g(a, k) for k in ("intarsia_spec_steps", "intarsia_spec_proposed_tokens", "intarsia_spec_accepted_tokens")}
        # radiance (the reference engine): every decode step is a speculative pass, decode tokens / decode steps is tokens per pass
        if arm == "radiance":
            steps_, toks = g(b, "radiance:engine_steps_decode") - g(a, "radiance:engine_steps_decode"), g(b, "radiance:decode_tokens") - g(a, "radiance:decode_tokens")
            d = {"intarsia_spec_steps": steps_, "intarsia_spec_accepted_tokens": toks - steps_, "intarsia_spec_proposed_tokens": 0.0}
        spec.setdefault((arm, cat), []).append(d)
arms = sorted({a for a, _ in data}, key=lambda a: (a != "off", a))
cats = sorted({c for _, c in data})
names = {k: sorted(x["name"] for x in v[0]) for k, v in data.items()}
for c in cats:
    ref = names.get(("off", c))
    for a in arms:
        if names.get((a, c)) != ref:
            problems.append(f"{a} {c}: rows differ from off ({len(names.get((a, c)) or [])} vs {len(ref or [])})")
def pooled(cases):
    toks = sum(x["speed"]["completion_tokens"] - 1 for x in cases)
    secs = sum((x["speed"]["completion_tokens"] - 1) / x["speed"]["decode_tok_s"] for x in cases if x["speed"]["decode_tok_s"])
    return toks / secs
print(f"window {W}: {len(runs)} localeval runs; problems: {len(problems)}")
for p in problems[:20]:
    print("  PROBLEM", p)
print(f"{'category':12} " + " ".join(f"{a:>22}" for a in arms))
for c in cats:
    off = st.median(pooled(x) for x in data[("off", c)]) if ("off", c) in data else None
    row = []
    for a in arms:
        if (a, c) not in data:
            row.append(f"{'-':>22}"); continue
        v = st.median(pooled(x) for x in data[(a, c)])
        tp = ""
        if (a, c) in spec and spec[(a, c)][0]["intarsia_spec_steps"] > 0:
            s = spec[(a, c)][0]; tp = f" {(s['intarsia_spec_accepted_tokens'] + s['intarsia_spec_steps']) / s['intarsia_spec_steps']:.2f}t/p"
        row.append(f"{v:6.1f} {v / off:5.2f}x{tp:>9}" if off else f"{v:6.1f}")
    print(f"{c:12} " + " ".join(f"{x:>22}" for x in row))
