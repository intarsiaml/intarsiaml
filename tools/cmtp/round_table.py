#!/usr/bin/env python3
"""Lane cmtp: per-round decode tok/s of a scripts/cmtp/window.sh window, per category and arm, and each arm's ratio to spec-off of the same round.
usage: python3 -I tools/cmtp/round_table.py <raw window dir>   (results under /opt/code/deepeval/results)
Pooled per category and round (tokens / sum of decode times, as tools/dfh/cat_table.py); the gate is every category's ratio >= 0.99 in the
median of the rounds, the table also shows the worst single round."""
import json, os, re, statistics as st, sys
raw = sys.argv[1]; res = "/opt/code/deepeval/results"
steps = open(os.path.join(raw, "steps.log")).read()
runs = re.findall(r"round (\d+) arm (\S+) (\S+): localeval exit (\d+) (results/\S+)", steps)
data = {}
for r, arm, cat, rc, path in runs:
    cases = json.load(open(os.path.join(os.path.dirname(res), path)))["datasets"][0]["cases"]
    n = sum(c["speed"]["completion_tokens"] - 1 for c in cases)
    s = sum((c["speed"]["completion_tokens"] - 1) / c["speed"]["decode_tok_s"] for c in cases)
    data[(arm, cat, int(r))] = n / s
arms = sorted({a for a, _, _ in data}, key=lambda a: (a != "off", a)); cats = sorted({c for _, c, _ in data}); rounds = sorted({r for _, _, r in data})
print(f"{'category':11} {'arm':5} " + " ".join(f"{'r%d tok/s' % r:>9}" for r in rounds) + f" {'median':>7} {'x off':>6} {'worst round x off':>18}")
for c in cats:
    for a in arms:
        v = [data[(a, c, r)] for r in rounds]; o = [data[("off", c, r)] for r in rounds]
        ratios = [x / y for x, y in zip(v, o)]
        print(f"{c:11} {a:5} " + " ".join(f"{x:9.1f}" for x in v) + f" {st.median(v):7.1f} {st.median(v) / st.median(o):6.2f} {min(ratios):18.3f}")
