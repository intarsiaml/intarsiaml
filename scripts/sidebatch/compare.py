#!/usr/bin/env python3
"""sidebatch: compare reply shas across arms. compare.py <out-dir> <ref-arm>:<arm>[,<arm>...] ... : each argument is a group whose
first arm is the reference; prints per (wave, i) EQUAL / MISMATCH per arm of the group and a per-group verdict."""
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
def load(a):
    r = {}
    for l in open(d / f"{a}.jsonl"):
        j = json.loads(l)
        if "sha" in j: r[(j["wave"], j["i"])] = j
        elif "error" in j: r[(j["wave"], j["i"])] = j
    return r
for g in sys.argv[2:]:
    arms = g.split(","); ref = load(arms[0])
    for a in arms[1:]:
        x = load(a); bad = 0; n = 0; rows = []
        for k in sorted(set(ref) | set(x)):
            r, y = ref.get(k), x.get(k)
            ok = r and y and "sha" in r and "sha" in y and r["sha"] == y["sha"] and r["toks"] == y["toks"]
            n += 1; bad += (not ok)
            if not ok: rows.append(f"  MISMATCH {k}: {arms[0]}={r and (r.get('sha'), r.get('toks'), r.get('error'))} {a}={y and (y.get('sha'), y.get('toks'), y.get('error'))}")
        print(f"{arms[0]} vs {a}: {n - bad}/{n} identical" + ("  PASS" if not bad else "  FAIL"))
        print("\n".join(rows))
