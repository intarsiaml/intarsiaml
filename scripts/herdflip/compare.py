#!/usr/bin/env python3
"""herdflip: compare the reply ids of the herd-shaped cells across arms (stdlib only).
Reads the JSON lines of scripts/flipev/herd_bench.py (CUDA: keys arm, seed, mode, stagger, ids, ttft_s) or scripts/mflip/burst.py (Metal: arm, seed, mode, gap, ids, ttft).
  compare.py results.jsonl --groups on:on_new,on_base,on_zero off:off_new,off_base [--max-tokens 48]
Within a group every reply of a prompt (seed, index) must be identical in every arm, burst and solo rows alike; a group is PASS when no prompt differs, every row
has all its requests ok and every reply has --max-tokens ids. Prints one TTFT line per arm and burst cell (sum / median / last of the 8 requests, client side)."""
import argparse, json, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("file"); ap.add_argument("--groups", nargs="+", required=True); ap.add_argument("--max-tokens", type=int, default=48)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.file) if l.strip()]
fails = 0
for g in a.groups:
    gname, arms = g.split(":"); arms = arms.split(",")
    ref = {}  # (seed, idx) -> (ids, arm, mode)
    bad = []; n = 0
    for r in rows:
        arm = r.get("visit") or r["arm"]
        if arm not in arms: continue
        ids = r["ids"]; errs = r.get("errors") or [None] * len(ids)
        for i, (x, e) in enumerate(zip(ids, errs)):
            n += 1
            if e or len(x) != a.max_tokens: bad.append(f"{arm} seed {r['seed']} {r['mode']} #{i}: error/short ({e or len(x)})"); continue
            k = (r["seed"], i)
            if k not in ref: ref[k] = (x, arm, r["mode"])
            elif ref[k][0] != x: bad.append(f"{arm} seed {r['seed']} {r['mode']} #{i} differs from {ref[k][1]} {ref[k][2]}")
    seen = sorted({r.get("visit") or r["arm"] for r in rows} & set(arms))
    ok = not bad and set(seen) == set(arms) and n > 0
    fails += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'} group {gname}: arms {seen}, {n} replies, {len(ref)} prompts, {len(bad)} differences")
    for b in bad[:10]: print("   ", b)
print("-- TTFT (burst cells; sum / p50 / last of the requests, seconds)")
for r in rows:
    if r["mode"] == "solo": continue
    t = r.get("ttft_s") or r.get("ttft"); t = [x for x in t if x is not None]
    cell = f"seed {r['seed']} stagger/gap {r.get('stagger', r.get('gap', 0))}"
    if t: print(f"{(r.get('visit') or r['arm']):10s} {cell}: n {len(t)} sum {sum(t):.2f} p50 {statistics.median(t):.2f} last {max(t):.2f}  cached {r.get('cached_tokens', r.get('cached'))}")
print(f"gate fails={fails}")
sys.exit(1 if fails else 0)
