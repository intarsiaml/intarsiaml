#!/usr/bin/env python3
"""Compare the ident.py outputs of several servers: every phase and stream's text must be byte-identical to the first file's.
  python3 scripts/hgv/ident_cmp.py base-tp1.json hgv-tp1.json [more.json ...]   (exit 1 on any difference)"""
import json, sys

files = sys.argv[1:]
ref = json.load(open(files[0]))
bad = 0
for f in files[1:]:
    cur = json.load(open(f))
    n = same = 0
    for ph, v in ref["phases"].items():
        for i, (r, c) in enumerate(zip(v["streams"], cur["phases"][ph]["streams"])):
            n += 1
            ok = r["ok"] and c["ok"] and r["text"] == c["text"] and r["completion"] == c["completion"] and r["completion"] == ref["gen"]
            same += ok
            if not ok:
                bad += 1
                k = next((j for j, (x, y) in enumerate(zip(r.get("text", ""), c.get("text", ""))) if x != y), None)
                print(f"DIFF {files[0]} vs {f}: {ph} stream {i}: first differing char {k}, completion {r.get('completion')} vs {c.get('completion')}")
    print(f"{files[0]} vs {f}: {same}/{n} streams byte-identical ({sum(len(s['text']) for v in ref['phases'].values() for s in v['streams'])} chars in the reference)")
sys.exit(1 if bad else 0)
