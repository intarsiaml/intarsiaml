#!/usr/bin/env python3
"""storeids: tables from raw/ids.jsonl. usage: table.py ids.jsonl > tables.md"""
import json
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ids = {}  # (visit, mode, set) -> {(seed, idx): rec}
for r in rows:
    ids.setdefault((r["visit"], r["mode"], r["set"]), {})[(r["seed"], r["idx"])] = r
order = []
for r in rows:
    k = (r["visit"], r["mode"])
    if k not in order:
        order.append(k)
REFS = ["s_def", "s_spec0", "s_store", "s_store_spec0"]
SETS = sorted({r["set"] for r in rows})


def get(visit, mode, s):
    return ids.get((visit, mode, s), {})


def cmp(a, b):
    """(equal, n, first divergence indices) over the prompts both have, ids only."""
    eq = n = 0
    div = []
    for k, ra in a.items():
        rb = b.get(k)
        if rb is None or not ra.get("ok") or not rb.get("ok"):
            continue
        n += 1
        if ra["ids"] == rb["ids"]:
            eq += 1
        else:
            i = next((j for j, (x, y) in enumerate(zip(ra["ids"], rb["ids"])) if x != y), min(len(ra["ids"]), len(rb["ids"])))
            div.append(i)
    return eq, n, sorted(div)


print("# ids vs the single-sequence references (equal / compared; cold prompts only)\n")
print("| config (visit, mode) | set | cached_tokens != 0 | errors | " + " | ".join(f"= {r}" for r in REFS) + " |")
print("|---|---|---|---|" + "---|" * len(REFS))
for visit, mode in order:
    if mode == "lp":
        continue
    for s in SETS:
        a = get(visit, mode, s)
        if not a:
            continue
        warm = sum(1 for r in a.values() if r.get("cached_tokens"))
        err = sum(1 for r in a.values() if not r.get("ok"))
        cells = []
        for ref in REFS:
            b = get(ref, "seq", s)
            if not b:
                cells.append("-")
                continue
            eq, n, div = cmp(a, b)
            cells.append(f"{eq}/{n}" + (f" (first diff at token {min(div)}..{max(div)})" if div else ""))
        print(f"| {visit} {mode} | {s} | {warm} | {err} | " + " | ".join(cells) + " |")

print("\n# pairwise: burst vs burst (the 8 simultaneous prompts of a seed, both seeds)\n")
bursts = [v for v, m in order if m == "burst"]
print("| A | B | " + " | ".join(f"set {s}" for s in SETS) + " |")
print("|---|---|" + "---|" * len(SETS))
for i, x in enumerate(bursts):
    for y in bursts[i + 1:]:
        cells = []
        for s in SETS:
            eq, n, div = cmp(get(x, "burst", s), get(y, "burst", s))
            cells.append(f"{eq}/{n}" + (f" (tok {min(div)}..{max(div)})" if div else ""))
        print(f"| {x} | {y} | " + " | ".join(cells) + " |")

print("\n# first diverging token index per prompt: store off vs store on (solo, s_spec0 vs s_store_spec0; and burst, b_off vs b_store)\n")
for a_, b_, m in [("s_spec0", "s_store_spec0", "seq"), ("s_def", "s_store", "seq"), ("b_off", "b_store", "burst"), ("b_off_spec0", "b_store_spec0", "burst")]:
    for s in SETS:
        eq, n, div = cmp(get(a_, m, s), get(b_, m, s))
        if n:
            print(f"- {a_} vs {b_} ({m}), set {s}: {eq}/{n} equal; first-diff token index of the differing prompts: {div}")

print("\n# logprob cells (lp mode, 4 prompts per set, 4 tokens, top-5): token 0 = the prefill's last-row output. Per prompt: ids, top-1 logprob, top-1 minus top-2 margin (nats)\n")
lpv = [v for v, m in order if m == "lp"]
LP = {}
for s_ in SETS:
    print(f"\nset {s_}\n")
    print("| visit | " + " | ".join(f"p{i}" for i in range(4)) + " |")
    print("|---|" + "---|" * 4)
    for v in lpv:
        d = get(v, "lp", s_)
        cells = []
        for i in range(4):
            r = next((x for k, x in d.items() if k[1] == i), None)
            if not r or not r.get("ok"):
                cells.append("err" if r else "-")
                continue
            lp = r.get("token_logprobs") or []
            top = (r.get("top_logprobs_ids") or [[]])[0]
            marg = (top[0][1] - top[1][1]) if len(top) > 1 else float("nan")
            LP[(v, s_, i)] = (r["ids"], lp)
            cells.append(f"{r['ids']} {lp[0]:.5f} m={marg:.4f}")
        print(f"| {v} | " + " | ".join(cells) + " |")

print("\n# max |delta logprob| over the 4 generated positions, per set, between lp visits (nats; 0 = bitwise on these 4 rows)\n")
print("| A | B | " + " | ".join(f"set {s_}" for s_ in SETS) + " |")
print("|---|---|" + "---|" * len(SETS))
for i, x in enumerate(lpv):
    for y in lpv[i + 1:]:
        cells = []
        for s_ in SETS:
            m = 0.0
            same = 0
            for p in range(4):
                a, b = LP.get((x, s_, p)), LP.get((y, s_, p))
                if not a or not b:
                    continue
                same += a[0] == b[0]
                m = max(m, max((abs(u - w) for u, w in zip(a[1], b[1])), default=0.0))
            cells.append(f"{m:.2e} (ids equal {same}/4)")
        print(f"| {x} | {y} | " + " | ".join(cells) + " |")
