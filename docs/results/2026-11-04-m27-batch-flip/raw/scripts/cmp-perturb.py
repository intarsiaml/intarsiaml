#!/usr/bin/env python3
import json, math, os
def rm(r): return {int(x[0]): x[1] for x in r}
LEN = json.load(open(".scratch/logs/exp3/len.json")); LEN.update(json.load(open(".scratch/logs/exp4/nopeel-len.json")))
P = json.load(open(".scratch/logs/oracle/perturb.json"))
ROWS = {"258": [(100, None), (104, None), (105, None)], "260": [(138, None), (197, None)], "262": [(188, None)], "1028": [(63, None), (129, None), (195, None)]}
F = {}
if os.path.exists(".scratch/logs/oracle/ref-free.json"): F = json.load(open(".scratch/logs/oracle/ref-free.json"))
for k, rows in ROWS.items():
    S = LEN[k]["solo"]; B = LEN[k]["behind"]
    paths = {"solo(NAX)": S, "busy(tail4 or 2,..)": B}
    for tag in ("nax0", "kvf32", "tail3", "tail5", "tail7"):
        paths[tag] = P.get(f"{tag}/{k}")
    if k in F: paths["f64 ref"] = F[k]
    for row, _ in rows:
        ref_prefix = S["ids"][:row]
        top = sorted(rm(S["top"][row]), key=rm(S["top"][row]).get, reverse=True)[:2]
        print(f"plen {k} row {row} solo id {S['ids'][row]} top ids {top}")
        for n, v in paths.items():
            if not v or len(v["ids"]) < row + 1 or row >= len(v["top"]): print(f"   {n:20} (no row)"); continue
            fd = next((i for i, (a, b) in enumerate(zip(v["ids"], S["ids"])) if a != b), None)
            shared = fd is None or fd >= row
            m = rm(v["top"][row]); print(f"   {n:20} same-context={str(shared):5} first-diff-vs-solo={str(fd):5} " + " ".join(f"lp[{i}]={m.get(i, float('nan')):.4f}" for i in top))
        f = P.get(f"forced/{k}@{row}")
        if f:
            m = rm(f["top"][0]); print(f"   {'forced one-pass':20} " + " ".join(f"lp[{i}]={m.get(i, float('nan')):.4f}" for i in top))
