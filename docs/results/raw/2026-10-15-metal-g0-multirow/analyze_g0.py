#!/usr/bin/env python3
"""Join gemv_g0 samples (CSV, epoch ms) to a macmon trace, drop rounds with the GPU clock below 1450 MHz, and print
the G0 tables (markdown) plus a JSON summary.

usage: analyze_g0.py <dir with g0-*.csv and macmon.jsonl> [--min-mhz 1450]
"""
import csv, glob, json, os, statistics, sys, datetime, collections

d = sys.argv[1]
MIN_MHZ = 1450
if "--min-mhz" in sys.argv:
    MIN_MHZ = float(sys.argv[sys.argv.index("--min-mhz") + 1])

# ---- macmon
mm = []  # (epoch_ms, mhz)
for line in open(os.path.join(d, "macmon.jsonl")):
    line = line.strip()
    if not line.startswith("{"):
        continue
    j = json.loads(line)
    t = datetime.datetime.fromisoformat(j["timestamp"].replace("Z", "+00:00")).timestamp() * 1000
    mm.append((t, j["gpu_usage"][0]))
mm.sort()
mm_t = [t for t, _ in mm]

import bisect
def freq_in(a, b):
    i, j = bisect.bisect_left(mm_t, a), bisect.bisect_right(mm_t, b)
    return [mm[k][1] for k in range(i, j)]

# ---- samples
rows = []
bitwise = []
for f in sorted(glob.glob(os.path.join(d, "g0-*.csv"))):
    run = os.path.basename(f)[3:-4]
    for r in csv.DictReader(open(f)):
        if r["cand"] == "bitwise":
            bitwise.append((run, r["case"], r["a8"], int(r["rows"]), r["sec_per_dispatch"] == "1"))
            continue
        rows.append(dict(run=run, case=r["case"], a8=int(r["a8"]), cand=r["cand"], rows=int(r["rows"]),
                         round=int(r["round"]), t=float(r["sec_per_dispatch"]), ep=float(r["epoch_ms"]),
                         bytes=int(r["bytes_per_dispatch"])))

# ---- rounds -> validity
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["run"], r["case"], r["round"])].append(r)
valid = {}
clock = {}
for key, rs in groups.items():
    # chain end epochs; a chain started (its duration * count) earlier; use the window [min ep - 60 ms, max ep]
    a = min(x["ep"] for x in rs) - 60
    b = max(x["ep"] for x in rs)
    fs = freq_in(a - 100, b + 100)
    clock[key] = (min(fs) if fs else None, statistics.mean(fs) if fs else None)
    valid[key] = bool(fs) and min(fs) >= MIN_MHZ

by = collections.defaultdict(lambda: collections.defaultdict(list))  # (case,a8) -> cand -> [(run,round,t)]
t1 = {}  # (run,case,a8,round) -> serving1 time
nrounds = collections.Counter()
nvalid = collections.Counter()
for r in rows:
    key = (r["run"], r["case"], r["round"])
    k2 = (r["case"], r["a8"])
    if r["cand"] == "serving1":
        nrounds[k2] += 1
        nvalid[k2] += valid[key]
    if not valid[key]:
        continue
    by[k2][r["cand"]].append((r["run"], r["round"], r["t"]))
    if r["cand"] == "serving1":
        t1[(r["run"], r["case"], r["a8"], r["round"])] = r["t"]

def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    x = p * (len(v) - 1)
    lo, hi = int(x), min(int(x) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (x - lo)

summ = {}
for k2, cands in by.items():
    case, a8 = k2
    s = {}
    for cand, lst in cands.items():
        ts = [t for _, _, t in lst]
        ratios = [t / t1[(run, case, a8, rd)] for run, rd, t in lst if (run, case, a8, rd) in t1]
        s[cand] = dict(n=len(ts), t_med=statistics.median(ts), t_p10=q(ts, .1), t_p90=q(ts, .9),
                       ratio_med=statistics.median(ratios), ratio_p10=q(ratios, .1), ratio_p90=q(ratios, .9))
    # best output-rows-per-threadgroup P per R (lowest median time); the R = 1 sanity instance is mr1x1
    for R in (1, 2, 3, 4, 5, 6, 7, 8, 16):
        opts = [(v["t_med"], c) for c, v in s.items() if c.startswith(f"mr{R}x")]
        if opts:
            _, c = min(opts)
            s[f"mr{R}"] = dict(s[c], best_p=c.split("x",1)[1])
    summ[f"{case}/{a8}"] = s

# ---- print
print(f"clock filter: rounds with min GPU clock >= {MIN_MHZ:.0f} MHz (macmon 100 ms)\n")
print("| case | a8 | rounds valid/total | clock mean (MHz) over valid rounds |")
print("|---|---|---|---|")
for k2 in sorted(nrounds):
    cl = [clock[k][1] for k in clock if k[1] == k2[0] and valid[k] and
          any(x["a8"] == k2[1] for x in groups[k])]
    print(f"| {k2[0]} | {k2[1]} | {nvalid[k2]}/{nrounds[k2]} | {statistics.mean(cl) if cl else float('nan'):.0f} |")
print()
order = ["qo", "down", "kv", "gu", "lm"]
Rs = [1, 2, 3, 4, 5, 6, 7, 8, 16]
for a8 in (0, 1):
    for case in order:
        s = summ.get(f"{case}/{a8}")
        if not s:
            continue
        T1 = s["serving1"]["t_med"]
        print(f"### {case} ({'8-byte-aligned weights' if a8 else '16-byte-aligned weights'}), valid rounds {s['serving1']['n']}\n")
        print("| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |")
        print("|---|---|---|---|---|---|---|")
        for R in Rs:
            m = s.get(f"mr{R}")
            ex = s.get(f"tiled{R}") or None
            exname = f"tiled GEMM" if ex else ""
            if R == 16:
                nx = s.get("nax16")
                if nx:
                    ex, exname = nx, "NAX GEMM"
                elif ex:
                    exname = "tiled GEMM (no NAX logits)"
            if not m and not ex:
                continue
            ms = f"{m['t_med']*1e6:.1f} (cfg {m['best_p']})" if m else "-"
            mr = f"{m['ratio_med']:.3f} [{m['ratio_p10']:.3f}-{m['ratio_p90']:.3f}]" if m else "-"
            es = f"{ex['t_med']*1e6:.1f}" if ex else "-"
            er = f"{ex['ratio_med']:.2f}" if ex else "-"
            print(f"| {R} | {T1*1e6:.1f} | {ms} | {mr} | {es} | {er} | {exname} |")
        # P sweep
        print("P sweep (output rows per threadgroup; ratio to T(1)): " + "; ".join(
            f"R={R}: " + ", ".join(f"P{c.split('x')[1]}={s[c]['ratio_med']:.2f}" for c in sorted(s) if c.startswith(f"mr{R}x"))
            for R in Rs if R > 1))
        # extra tiled sizes
        extra = [c for c in s if c.startswith("tiled") and int(c[5:]) not in Rs]
        if extra:
            print("\nother tiled-GEMM row counts: " + ", ".join(f"M={c[5:]}: {s[c]['t_med']*1e6:.1f} us ({s[c]['ratio_med']:.2f}x)" for c in sorted(extra, key=lambda c: int(c[5:]))))
        print()

# ---- layer / step aggregates (medians of medians)
def tmed(case, a8, cand):
    s = summ.get(f"{case}/{a8}")
    return s[cand]["t_med"] if s and cand in s else None

counts = {  # layers of Qwen3-8B per (shape, alignment): from the safetensors headers
    "qo": (28 + 29, 8 + 7),   # q_proj + o_proj
    "down": (29, 7),
    "kv": (28, 8),
    "gu": (29, 7),
}
def layer_time(a8, cand_fn, R):
    tot = 0.0
    for case, mult in (("qo", 2), ("down", 1), ("kv", 1), ("gu", 1)):
        t = cand_fn(case, a8, R)
        if t is None:
            return None
        tot += mult * t
    return tot

def mr_fn(case, a8, R):
    return tmed(case, a8, f"mr{R}")
def serving_fn(case, a8, R):
    return tmed(case, a8, "serving1")
def existing_fn(case, a8, R):
    if R == 16:
        return tmed(case, a8, "nax16")
    return tmed(case, a8, f"tiled{R}") if case != "kv" and case != "gu" else (
        None if tmed(case, a8, f"tiled{R}") is None else tmed(case, a8, f"tiled{R}"))

print("### Layer sum (2 x q/o + down + k/v pair + gate/up pair), GEMV time only\n")
print("| R | alignment | T_layer(1) serving (us) | T_layer(R) multi-row (us) | ratio | T_layer(R) existing path (us) | existing ratio |")
print("|---|---|---|---|---|---|---|")
res = {}
for a8 in (0, 1):
    base = layer_time(a8, serving_fn, 1)
    if base is None:
        continue
    for R in Rs[1:]:
        m = layer_time(a8, mr_fn, R)
        e = layer_time(a8, existing_fn, R)
        if m is None:
            continue
        res[(a8, R)] = (base, m, e)
        print(f"| {R} | {'8-byte' if a8 else '16-byte'} | {base*1e6:.1f} | {m*1e6:.1f} | {m/base:.3f} | " +
              (f"{e*1e6:.1f} | {e/base:.2f} |" if e else "- | - |") if m else f"| {R} | - |")
print()

# census-weighted layer (a8 fraction by shape class)
def weighted(R, fn):
    tot = 0.0
    for case, mult, (n16, n8) in (("qo", 1, (28, 8)), ("qo", 1, (29, 7)), ("down", 1, (29, 7)), ("kv", 1, (28, 8)), ("gu", 1, (29, 7))):
        t16, t8 = fn(case, 0, R), fn(case, 1, R)
        if t16 is None or t8 is None:
            return None
        tot += (n16 * t16 + n8 * t8) / 36.0
    return tot
print("### Census-weighted (Qwen3-8B: 7-8 of 36 layers per shape sit in the 8-mod-16 shard), per layer, GEMV time\n")
print("| R | T_layer(1) (us) | T_layer(R) multi-row | ratio | existing | existing ratio |")
print("|---|---|---|---|---|---|")
w1 = weighted(1, serving_fn)
for R in Rs[1:]:
    wm, we = weighted(R, mr_fn), weighted(R, existing_fn)
    if w1 and wm:
        print(f"| {R} | {w1*1e6:.1f} | {wm*1e6:.1f} | {wm/w1:.3f} | " + (f"{we*1e6:.1f} | {we/w1:.2f} |" if we else "- | - |"))
print()

# per-step (36 layers + lm_head), aligned
print("### Per verify step, GEMVs only: 36 x layer + lm_head (all R rows through the head), 16-byte-aligned weights\n")
print("| R | step(1) (ms) | step(R) multi-row (ms) | step(R)/step(1) | step(R) existing (ms) | existing/step(1) | per-token cost multi-row (ms/row) |")
print("|---|---|---|---|---|---|---|")
lm1 = tmed("lm", 0, "serving1")
_k = [k for k in res if k[0] == 0]
if lm1 and _k:
    base = res[_k[0]][0]
    s1 = 36 * base + lm1
    for R in Rs[1:]:
        if (0, R) not in res:
            continue
        b, m, e = res[(0, R)]
        lmm = tmed("lm", 0, f"mr{R}")
        lme = tmed("lm", 0, f"tiled{R}")
        sm = 36 * m + lmm if lmm else None
        se = 36 * e + lme if (e and lme) else None
        print(f"| {R} | {s1*1e3:.2f} | " + (f"{sm*1e3:.2f} | {sm/s1:.3f} | " if sm else "- | - | ") +
              (f"{se*1e3:.2f} | {se/s1:.2f} | " if se else "- | - | ") + (f"{sm*1e3/R:.2f} |" if sm else "- |"))
print()
json.dump({"summary": summ, "bitwise": bitwise, "min_mhz": MIN_MHZ}, open(os.path.join(d, "summary.json"), "w"), indent=1)
# census-weighted per-step (36 x census-weighted layer + lm_head aligned)
print("### Per verify step, GEMVs only, census-weighted layers (16-byte / 8-mod-16 mix of the real shards) + lm_head\n")
print("| R | step(1) (ms) | step(R) multi-row (ms) | step(R)/step(1) |")
print("|---|---|---|---|")
_w1 = weighted(1, serving_fn)
_lm1 = tmed("lm", 0, "serving1")
if _w1 and _lm1:
    for R in Rs[1:]:
        wm, lmm = weighted(R, mr_fn), tmed("lm", 0, f"mr{R}")
        if wm and lmm:
            print(f"| {R} | {(36*_w1+_lm1)*1e3:.2f} | {(36*wm+lmm)*1e3:.2f} | {(36*wm+lmm)/(36*_w1+_lm1):.3f} |")
print()
print("bitwise checks in the bench:", all(b[4] for b in bitwise), f"({len(bitwise)} checks)")
