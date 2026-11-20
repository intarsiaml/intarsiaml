#!/usr/bin/env python3
"""herdflip: compare the reply ids of the herd-shaped cells across arms (stdlib only).
Reads the JSON lines of scripts/flipev/herd_bench.py (CUDA: keys arm, seed, mode, stagger, ids, ttft_s) or scripts/mflip/burst.py (Metal: arm, seed, mode, gap, ids, ttft).
  compare.py results.jsonl --groups on:on_new,on_base,on_zero off:off_new,off_base [--max-tokens 48]
Within a group every reply of a prompt (seed, index, mode) must be identical in every arm (burst against burst, solo against solo); a group is PASS when no prompt differs, every row
has all its requests ok and every reply has --max-tokens ids. Informational: burst against solo within a group. --engaged / --inert check from the replies' cached_tokens
that the herd acted in the cell where all 8 requests arrive together (more tokens restored than any herd-off store-on arm; store-off arms restore nothing), so that identical ids cannot come from a herd that never ran. Prints one TTFT line per arm and burst cell (sum / median / last of the 8 requests, client side)."""
import argparse, json, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("file"); ap.add_argument("--groups", nargs="+", required=True); ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--engaged", default="", help="comma list of arms where the herd must have acted: the first (stagger 0) cell has followers with cached_tokens > 0")
ap.add_argument("--inert", default="", help="comma list of arms where it must not have: herd-off store-on arms set the bar, arms named off* (no store) must have cached_tokens 0")
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
            k = (r["seed"], i, r["mode"] == "solo")
            if k not in ref: ref[k] = (x, arm, r["mode"])
            elif ref[k][0] != x: bad.append(f"{arm} seed {r['seed']} {r['mode']} #{i} differs from {ref[k][1]} {ref[k][2]}")
    seen = sorted({r.get("visit") or r["arm"] for r in rows} & set(arms))
    ok = not bad and set(seen) == set(arms) and n > 0
    fails += 0 if ok else 1
    # informational: burst against solo, per prompt, over every arm of the group
    cross = [k for k in ref if k[2] is False and (k[0], k[1], True) in ref and ref[k][0] != ref[(k[0], k[1], True)][0]]
    both = sum(1 for k in ref if k[2] is False and (k[0], k[1], True) in ref)
    print(f"{'PASS' if ok else 'FAIL'} group {gname}: arms {seen}, {n} replies, {len(ref)} prompts, {len(bad)} differences")
    print(f"    info: burst reply == solo reply (first arm seen): {both - len(cross)} of {both} prompts")
    for b in bad[:10]: print("   ", b)
def first_cell_cached(arm):
    cs = [r for r in rows if (r.get("visit") or r["arm"]) == arm and r["mode"] != "solo"]
    if not cs: return None
    c = (cs[0].get("cached_tokens") or cs[0].get("cached") or [])
    return [x or 0 for x in c]
# engagement: the store alone already restores part of a prefix for a request admitted after the leader's first chunk (herd off, cell 1: cached 4096 for 7 of 8 on CUDA / Metal),
# so "the herd acted" is read as: its stagger-0 cell restores more cached_tokens in total than every herd-off store-on arm's; store-off arms must restore nothing
inert_sum = {}
for arm in filter(None, a.inert.split(",")):
    c = first_cell_cached(arm); ok = bool(c) and (sum(c) == 0 if arm.startswith("off") else True); fails += 0 if ok else 1
    inert_sum[arm] = sum(c or [])
    print(f"{'PASS' if ok else 'FAIL'} inert {arm}: cached_tokens of the stagger-0 cell {c} (sum {inert_sum[arm]}{', must be 0: no store' if arm.startswith('off') else ''})")
ref = max([v for k, v in inert_sum.items() if not k.startswith("off")] or [0])
for arm in filter(None, a.engaged.split(",")):
    c = first_cell_cached(arm); ok = bool(c) and sum(c) > ref; fails += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'} engaged {arm}: cached_tokens of the stagger-0 cell {c} (sum {sum(c or [])} > herd-off store-on max {ref})")
print("-- TTFT (burst cells; sum / p50 / last of the requests, seconds)")
for r in rows:
    if r["mode"] == "solo": continue
    t = r.get("ttft_s") or r.get("ttft"); t = [x for x in t if x is not None]
    cell = f"seed {r['seed']} stagger/gap {r.get('stagger', r.get('gap', 0))}"
    if t: print(f"{(r.get('visit') or r['arm']):10s} {cell}: n {len(t)} sum {sum(t):.2f} p50 {statistics.median(t):.2f} last {max(t):.2f}  cached {r.get('cached_tokens', r.get('cached'))}")
print(f"gate fails={fails}")
sys.exit(1 if fails else 0)
