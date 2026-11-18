#!/usr/bin/env python3
"""mflip tables. analyze.py <results.jsonl> [burst|sjf]. Prints markdown. Reference ids: the first A-arm (features off) solo cell of the same seed."""
import json, statistics as st, sys, collections
path = sys.argv[1]; kind = sys.argv[2] if len(sys.argv) > 2 else "burst"
rows = [json.loads(l) for l in open(path)]
m = lambda xs: f"{st.mean(xs):.2f}" if xs else "-"
rg = lambda xs: f"{min(xs):.1f}..{max(xs):.1f}" if xs else "-"
if kind == "sjf":
    print("| arm | n | short TTFT mean (min..max) | long TTFT mean (min..max) | short end | long end |\n|---|---|---|---|---|---|")
    for arm in "AB":
        c = [r for r in rows if r["mode"] == "pair" and r["arm"] == arm]
        s = [r["short"]["ttft_s"] for r in c]; l = [r["long"]["ttft_s"] for r in c]
        print(f'| {"fifo" if arm=="A" else "shortest"} | {len(c)} | {m(s)} ({rg(s)}) | {m(l)} ({rg(l)}) | {m([r["short"]["end_s"] for r in c])} | {m([r["long"]["end_s"] for r in c])} |')
    ref = {}
    for r in rows:
        if r["mode"] == "solo" and r["arm"] == "A": ref.setdefault(r["seed"], r)
    ok = bad = n48 = 0
    for r in rows:
        if r["seed"] not in ref: continue
        for k in ("long", "short"):
            if r[k]["ids"] == ref[r["seed"]][k]["ids"]: ok += 1
            else: bad += 1; print("MISMATCH", r["visit"], r["mode"], r["seed"], k)
            n48 += len(r[k]["ids"])
    print(f"\nids: {ok} of {ok+bad} replies equal the fifo solo reply of the same prompt (pair and solo cells, both arms); mismatches {bad}; ids compared {n48}; errors {sum(1 for r in rows for k in ('long','short') if r[k]['error'])}")
    sys.exit()
groups = collections.defaultdict(list)
for r in rows:
    if r["mode"] == "burst": groups[(r["gap"], r["short_len"] > 0)].append(r)
for (gap, hasshort), g in sorted(groups.items()):
    print(f"\n**gap {gap} s{', short 1000-token prompt at +2 s' if hasshort else ''}**\n")
    print("| arm | n cells | herd TTFT sum s | p50 | last herd TTFT | short TTFT | wall s | per-request mean TTFT by arrival index (0..) |\n|---|---|---|---|---|---|---|---|")
    for arm in "AB":
        c = [r for r in g if r["arm"] == arm]
        if not c: continue
        n = c[0]["n"]
        hs = [sum(r["ttft"][:n]) for r in c]; p50 = [st.median(r["ttft"][:n]) for r in c]; last = [max(r["ttft"][:n]) for r in c]
        sh = [r["ttft"][n] for r in c if len(r["ttft"]) > n]
        per = " / ".join(f"{st.mean(r['ttft'][i] for r in c):.1f}" for i in range(len(c[0]["ttft"])))
        print(f'| {"off" if arm=="A" else "on"} | {len(c)} | {m(hs)} ({rg(hs)}) | {m(p50)} | {m(last)} ({rg(last)}) | {m(sh)} ({rg(sh)}) | {m([r["wall_s"] for r in c])} ({rg([r["wall_s"] for r in c])}) | {per} |')
    for arm in "AB":
        c = [r for r in g if r["arm"] == arm]
        if c: print(f"\ncached tokens, arm {arm}:", sorted({tuple(r["cached"]) for r in c}))
    # follower slower than off? per index, per seed
    seeds = sorted({r["seed"] for r in g})
    for s in seeds:
        a = [r for r in g if r["arm"] == "A" and r["seed"] == s]; b = [r for r in g if r["arm"] == "B" and r["seed"] == s]
        if a and b:
            d = [st.mean(r["ttft"][i] for r in b) - st.mean(r["ttft"][i] for r in a) for i in range(len(a[0]["ttft"]))]
            print(f"seed {s}: on - off mean TTFT per arrival index (s): " + ", ".join(f"{x:+.1f}" for x in d) + f"  (worst {max(d):+.2f})")
ref = {}
for r in rows:
    if r["mode"] == "solo" and r["arm"] == "A": ref.setdefault(r["seed"], r["ids"])
ok = bad = errs = nids = 0; per = collections.Counter()
for r in rows:
    errs += sum(1 for e in r["errors"] if e)
    if r["seed"] not in ref: continue
    for i, ids in enumerate(r["ids"]):
        if ids == ref[r["seed"]][i]: ok += 1; per[(r["mode"], r["arm"], "eq")] += 1
        else: bad += 1; per[(r["mode"], r["arm"], "NEQ")] += 1; print("MISMATCH", r["visit"], r["mode"], r["seed"], i)
        nids += len(ids)
print(f"\nids: {ok} of {ok+bad} replies equal the arm-A solo reply of the same prompt; mismatches {bad}; ids compared {nids}; errors {errs}; by cell {dict(per)}")
