#!/usr/bin/env python3
"""mherd tables. analyze.py herd <results.jsonl> | sjf <results.jsonl>. Prints markdown."""
import json, statistics as st, sys, collections

kind, path = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path)]
f = lambda xs: f"{st.mean(xs):.2f}" if xs else "-"
rng = lambda xs: f"{min(xs):.2f}..{max(xs):.2f}"

if kind == "herd":
    print("| cell | arm | per-cell values (visit: sum / last / wall) | mean TTFT sum s | mean p50 s | mean last s | mean wall s | n |\n|---|---|---|---|---|---|---|---|")
    for mode in ("herd8", "distinct"):
        for arm in "AB":
            c = [r for r in rows if r["mode"] == mode and r["arm"] == arm and r["ttft_sum_s"] is not None]
            if not c: continue
            per = "; ".join(f'{r["visit"]}s{r["seed"]}: {r["ttft_sum_s"]:.1f} / {r["ttft_last_s"]:.1f} / {r["wall_s"]:.1f}' for r in c)
            print(f'| {mode} | {"off" if arm=="A" else "on"} | {per} | {f([r["ttft_sum_s"] for r in c])} | {f([r["ttft_p50_s"] for r in c])} | {f([r["ttft_last_s"] for r in c])} | {f([r["wall_s"] for r in c])} | {len(c)} |')
    print()
    for arm in "AB":
        c = [r for r in rows if r["mode"] == "herd8" and r["arm"] == arm]
        print(f"cached tokens per request, herd8, arm {arm}:", sorted({tuple(r["cached_tokens"]) for r in c}))
    # ids
    ref = {}
    for r in rows:
        if r["mode"] == "solo" and r["arm"] == "B": ref.setdefault(r["seed"], r["ids"])
    ok = bad = 0; errs = sum(1 for r in rows for e in r["errors"] if e); short = 0
    bycell = collections.Counter()
    for r in rows:
        if r["mode"] == "distinct": continue
        for i, ids in enumerate(r["ids"]):
            if r["seed"] not in ref: continue
            if ids == ref[r["seed"]][i]: ok += 1; bycell[(r["mode"], r["arm"], "eq")] += 1
            else: bad += 1; bycell[(r["mode"], r["arm"], "NEQ")] += 1; print("MISMATCH", r["visit"], r["mode"], r["seed"], i)
            if len(ids) != r["args"]["max_tokens"]: short += 1
    print(f"ids: {ok} of {ok+bad} replies equal the store-on solo reply of the same prompt (first B-arm solo cell of the seed); mismatches {bad}; errors {errs}; replies not {rows[0]['args']['max_tokens']} tokens {short}")
    print(dict(bycell))
    print("solo reply vs solo reply across arms/visits equal:", all(a["ids"] == b["ids"] for a in rows for b in rows if a["mode"] == b["mode"] == "solo" and a["seed"] == b["seed"]))
else:
    print("| arm | cell (visit s seed) | short TTFT s | long TTFT s | short end s | long end s |\n|---|---|---|---|---|---|")
    for arm in "AB":
        c = [r for r in rows if r["mode"] == "pair" and r["arm"] == arm]
        for r in c: print(f'| {"fifo" if arm=="A" else "shortest"} | {r["visit"]} s{r["seed"]} | {r["short"]["ttft_s"]} | {r["long"]["ttft_s"]} | {r["short"]["end_s"]} | {r["long"]["end_s"]} |')
    print("\n| arm | n | short TTFT mean (min..max) | long TTFT mean (min..max) |\n|---|---|---|---|")
    for arm in "AB":
        c = [r for r in rows if r["mode"] == "pair" and r["arm"] == arm]
        s = [r["short"]["ttft_s"] for r in c]; l = [r["long"]["ttft_s"] for r in c]
        print(f'| {"fifo" if arm=="A" else "shortest"} | {len(c)} | {f(s)} ({rng(s)}) | {f(l)} ({rng(l)}) |')
    ref = {}
    for r in rows:
        if r["mode"] == "solo" and r["arm"] == "A": ref.setdefault(r["seed"], r)
    ok = bad = 0
    for r in rows:
        if r["seed"] not in ref: continue
        for k in ("long", "short"):
            if r[k]["ids"] == ref[r["seed"]][k]["ids"]: ok += 1
            else: bad += 1; print("MISMATCH", r["visit"], r["mode"], r["seed"], k)
    print(f"ids: {ok} of {ok+bad} replies (pair and solo cells, both arms) equal the fifo-arm solo reply of the same prompt; mismatches {bad}")
    pa = {(r["arm"], r["seed"], k): r[k]["ids"] for r in rows if r["mode"] == "pair" for k in ("long", "short")}
    cmp_ = [(s, k) for (a, s, k) in pa if a == "A" and ("B", s, k) in pa]
    print("pair cells shortest == fifo (same seed, long/short):", sum(pa[("A", s, k)] == pa[("B", s, k)] for s, k in set(cmp_)), "of", len(set(cmp_)))
