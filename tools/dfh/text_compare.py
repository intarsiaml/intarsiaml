#!/usr/bin/env python3
"""Lane asp: category texts of a cat-window against `off`: every case of every arm and round vs the same case of the same
round's `off` run. usage: REF=off python3 -I tools/dfh/text_compare.py <raw window dir> [more raw dirs] (REF: the reference arm); prints per arm matches / total and the
misses."""
import json, os, re, sys, collections
REF = os.environ.get("REF", "off")
res = "/opt/code/deepeval"
runs = []
for raw in sys.argv[1:]:
    steps = open(os.path.join(raw, "steps.log")).read()
    runs += [(os.path.basename(raw.rstrip("/")),) + r for r in re.findall(r"round (\d+) arm (\S+) (\S+): localeval exit (\d+) (results/\S+)", steps)]
txt = {}
for w, r, arm, cat, rc, path in runs:
    for c in json.load(open(os.path.join(res, path)))["datasets"][0]["cases"]:
        txt[(w, r, arm, cat, c["name"])] = c["actual_output"]
ok, tot, miss = collections.Counter(), collections.Counter(), []
for (w, r, arm, cat, name), t in sorted(txt.items()):
    if arm == REF:
        continue
    ref = txt.get((w, r, REF, cat, name))
    if ref is None:
        continue
    tot[arm] += 1
    if ref == t:
        ok[arm] += 1
    else:
        miss.append(f"{w} r{r} {arm} {cat} {name}")
for a in sorted(tot):
    print(f"{a}: {ok[a]} / {tot[a]} texts equal to {REF}")
for m in miss:
    print("MISS", m)
