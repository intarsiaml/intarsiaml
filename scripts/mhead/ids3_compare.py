#!/usr/bin/env python3
"""Lane mhead: compare the ids3.py outputs. usage: ids3_compare.py DIR REF_ARM ARM... (files DIR/ids3-<arm>.json)
Every key of every arm must equal the REF_ARM's solo ("s1:<prompt>") ids of the same prompt. Prints one line per arm and exits 1 on a mismatch."""
import json, sys

d, ref, *arms = sys.argv[1:]
load = lambda a: json.load(open(f"{d}/ids3-{a}.json"))
R = load(ref)
bad = 0
for a in [ref] + arms:
    A = load(a)
    ok = n = 0
    for k, ids in A.items():
        p = k.split(":", 1)[1] if k.startswith("s1:") else k.split(":")[1].split("|")[int(k.split(":")[2])]
        n += 1
        if ids == R[f"s1:{p}"] and len(ids) > 0:
            ok += 1
        else:
            bad += 1
            first = next((i for i, (x, y) in enumerate(zip(ids, R[f"s1:{p}"])) if x != y), None)
            print(f"  MISMATCH {a} {k}: len {len(ids)} vs {len(R[f's1:{p}'])}, first difference at {first}")
    print(f"{a}: {ok} / {n} identical to {ref} solo")
sys.exit(1 if bad else 0)
