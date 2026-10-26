"""Lane dc: per-category DFlash vs spec-off from localeval categories-rcp runs (label dc-cat-<cfg>-r<n>).
usage: python3 -I summarize_bench.py <localeval results dir> [rounds]   (stdlib only)"""
import glob, json, os, statistics as st, sys
res = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
def load(cfg, r):
    d = sorted(glob.glob(os.path.join(res, f"*-adhoc-dc-cat-{cfg}-r{r}")))
    return json.load(open(d[-1] + "/result.json"))["datasets"][0]["cases"] if d else None
def cat(name): return name.split("-")[0]
data = {(c, r): load(c, r) for c in ("off", "dflash") for r in range(1, rounds + 1)}
names = [c["name"] for c in data[("off", 1)]]
def pooled(cases, sel):
    toks = sum(c["speed"]["completion_tokens"] - 1 for c in cases if sel(c))
    secs = sum((c["speed"]["completion_tokens"] - 1) / c["speed"]["decode_tok_s"] for c in cases if sel(c) and c["speed"]["decode_tok_s"])
    return toks / secs if secs else float("nan")
def e2e(cases, sel):
    return sum(c["speed"]["completion_tokens"] for c in cases if sel(c)) / sum(c["speed"]["total_s"] for c in cases if sel(c))
print("per category, pooled over prompts, per round (decode tok/s = tokens / sum of decode times; e2e = tokens / total call time)")
print(f"{'category':8} {'n':>2} | " + " ".join(f"off r{r} dfl r{r} x{r}" for r in range(1, rounds + 1)) + " | median speedup | e2e speedup")
for k in ("recall", "code", "prose", "ALL"):
    sel = (lambda c: True) if k == "ALL" else (lambda c, k=k: cat(c["name"]) == k)
    n = sum(1 for c in data[("off", 1)] if sel(c)); row = []; sp = []; se = []
    for r in range(1, rounds + 1):
        o, d = pooled(data[("off", r)], sel), pooled(data[("dflash", r)], sel)
        row.append(f"{o:6.1f} {d:6.1f} {d/o:4.2f}"); sp.append(d / o)
        se.append(e2e(data[("dflash", r)], sel) / e2e(data[("off", r)], sel))
    print(f"{k:8} {n:2} | " + " ".join(row) + f" | {st.median(sp):5.2f}x ({min(sp):.2f}-{max(sp):.2f}) | {st.median(se):.2f}x")
print("\nper prompt (median over rounds): speedup = dflash decode / off decode, and whether the answer text is identical")
bad = 0; diff = 0
for i, n in enumerate(names):
    sp = []
    for r in range(1, rounds + 1):
        o, d = data[("off", r)][i], data[("dflash", r)][i]
        sp.append(d["speed"]["decode_tok_s"] / o["speed"]["decode_tok_s"])
        if o["actual_output"] != d["actual_output"]: diff += 1
    m = st.median(sp); bad += m < 1.0
    print(f"  {n:28} {m:5.2f}x ({min(sp):.2f}-{max(sp):.2f}) tokens {data[('off',1)][i]['speed']['completion_tokens']}")
print(f"\nprompts with median speedup < 1: {bad} of {len(names)}; answers differing between off and dflash: {diff} (of {len(names)*rounds})")
