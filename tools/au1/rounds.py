#!/usr/bin/env python3
"""Lane au1: pooled decode tok/s per category, arm and round of a window (the spread behind tools/au1/win_table.py).
usage: rounds.py <raw/W>"""
import json, os, re, sys
raw = sys.argv[1].rstrip("/"); res = "/opt/code/deepeval/results"
runs = re.findall(r"round (\d+) arm (\S+) (\S+): localeval exit (\d+) (results/\S+)", open(os.path.join(raw, "steps.log")).read())
d = {}
for r, arm, cat, rc, path in runs:
    cases = json.load(open(os.path.join(os.path.dirname(res), path)))["datasets"][0]["cases"]
    toks = sum(x["speed"]["completion_tokens"] - 1 for x in cases)
    secs = sum((x["speed"]["completion_tokens"] - 1) / x["speed"]["decode_tok_s"] for x in cases)
    d[(cat, arm, int(r))] = toks / secs
arms = sorted({a for _, a, _ in d}); rounds = sorted({r for _, _, r in d}); cats = sorted({c for c, _, _ in d})
print("| category | " + " | ".join(f"{a} r{r}" for a in arms for r in rounds) + " | auto / mtp2 per round | auto / best per round |")
print("|---|" + "---:|" * (len(arms) * len(rounds) + 2))
for c in cats:
    row = [f"{d[(c, a, r)]:.2f}" if (c, a, r) in d else "-" for a in arms for r in rounds]
    a1 = [d[(c, "auto", r)] / d[(c, "mtp2", r)] for r in rounds if (c, "auto", r) in d and (c, "mtp2", r) in d]
    a2 = [d[(c, "auto", r)] / max(d[(c, "mtp2", r)], d[(c, "dfl7", r)]) for r in rounds if all((c, x, r) in d for x in ("auto", "mtp2", "dfl7"))]
    print(f"| {c} | " + " | ".join(row) + " | " + " / ".join(f"{x:.3f}" for x in a1) + " | " + " / ".join(f"{x:.3f}" for x in a2) + " |")
