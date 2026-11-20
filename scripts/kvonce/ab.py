#!/usr/bin/env python3
"""kvonce: ABBA table from the sampled.py JSONs of scripts/kvonce/window.sh (python3 scripts/kvonce/ab.py RAW_DIR OFF_WINDOWS ON_WINDOWS [SPECOFF_WINDOWS]).
Windows are comma lists (a1,a2,...); a window named in RAW_DIR/guard-fail-<w>.txt is dropped. Per cell: every rep of every window of an arm pooled,
mean decode tok/s, 2 SE of the difference of the means, per-window means, MTP tokens per verify pass, ms per verify pass (tokens per pass x 1000 / tok/s)."""
import json, math, pathlib, sys

raw = pathlib.Path(sys.argv[1]); arms = {"off": sys.argv[2].split(","), "on": sys.argv[3].split(",")}
specoff = sys.argv[4].split(",") if len(sys.argv) > 4 else []
SIZES = [4096, 32768, 65536]

def ok(w):
    return not (raw / f"guard-fail-{w}.txt").exists()

def rows(w, T):
    f = raw / f"{w}-T{T}-sampled.json"
    if not f.exists() or not ok(w):
        return {}
    return {r["size"]: r for r in json.load(open(f))["rows"] if "skipped" not in r}

def stats(v):
    m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, sd / math.sqrt(len(v)) if v else 0.0

for T, name in [("0.7", "sampled T=0.7"), ("0", "greedy")]:
    print(f"\n### {name}: decode tok/s, mean of all reps (n), per-window means\n")
    print("| ctx | off | on | on / off | 2 SE of diff | windows off | windows on | tok/pass off / on | ms/pass off / on |\n|---|---|---|---|---|---|---|---|---|")
    for s in SIZES:
        cells = {}
        for a, ws in arms.items():
            each, wm, tpv = [], [], []
            for w in ws:
                r = rows(w, T).get(s)
                if r:
                    each += r["decode_each"]; wm.append(sum(r["decode_each"]) / len(r["decode_each"]))
                    t = ((r.get("mtp") or {}).get("tokens_per_verify"))
                    if t: tpv.append(t)
            cells[a] = (each, wm, sum(tpv) / len(tpv) if tpv else None)
        (eo, wo, to), (en, wn, tn) = cells["off"], cells["on"]
        if not eo or not en:
            continue
        mo, so = stats(eo); mn, sn = stats(en)
        se2 = 2 * math.sqrt(so ** 2 + sn ** 2) / mo
        msp = lambda t, m: f"{t * 1000 / m:.1f}" if t else "-"
        print(f"| {s // 1024}k | {mo:.1f} ({len(eo)}) | {mn:.1f} ({len(en)}) | **{mn / mo:.3f}** | {se2:.1%} | {' / '.join(f'{x:.1f}' for x in wo)} | {' / '.join(f'{x:.1f}' for x in wn)} | "
              f"{to or 0:.2f} / {tn or 0:.2f} | {msp(to, mo)} / {msp(tn, mn)} |")
    if specoff:
        for s in SIZES:
            each = []
            for w in specoff:
                r = rows(w, T).get(s)
                if r: each += r["decode_each"]
            if each:
                m, se = stats(each)
                print(f"\n`--spec off` {name} {s // 1024}k: {m:.1f} tok/s ({len(each)} reps, SE {se:.1f})")
