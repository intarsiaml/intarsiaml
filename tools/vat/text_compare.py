#!/usr/bin/env python3
"""vat: category texts of every arm of a cat-window.sh run against the --spec off arm's text for the same case.

usage: python3 tools/vat/text_compare.py <raw window dir>   (reads its steps.log and /opt/code/deepeval/results)
The off reference is the first off round of the window (greedy decoding: the text of a case does not depend on the round)."""
import collections, json, os, re, sys

raw = sys.argv[1].rstrip("/")
res = "/opt/code/deepeval/results"
steps = open(os.path.join(raw, "steps.log")).read()
cases = {}  # (arm, cat, round) -> {case name: text}
for r, arm, cat, rc, path in re.findall(r"round (\d+) arm (\S+) (\S+): localeval exit (\d+) (results/\S+)", steps):
    d = json.load(open(os.path.join(os.path.dirname(res), path)))["datasets"][0]["cases"]
    cases[(arm, cat, int(r))] = {c["name"]: c["actual_output"] for c in d}
ref = {}
for (arm, cat, r), v in sorted(cases.items(), key=lambda kv: kv[0][2]):
    if arm == "off":
        ref.setdefault(cat, v)
stat = collections.defaultdict(lambda: [0, 0])
for (arm, cat, r), v in sorted(cases.items()):
    if arm.startswith("off") and arm != "off-old":
        continue
    for name, text in v.items():
        t = ref.get(cat, {}).get(name)
        if t is None:
            continue
        stat[arm][0] += 1
        if text == t:
            stat[arm][1] += 1
        else:
            k = next((i for i, (a, b) in enumerate(zip(text, t)) if a != b), min(len(text), len(t)))
            print(f"DIFF {arm} {cat} r{r} {name}: first differing char {k}")
for arm, (n, same) in sorted(stat.items()):
    print(f"{arm}: {same} / {n} cases identical to off")
