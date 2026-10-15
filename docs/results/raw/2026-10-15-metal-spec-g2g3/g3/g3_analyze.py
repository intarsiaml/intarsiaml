#!/usr/bin/env python3
"""g3_analyze.py <runs dir> : per-row table (median of reps) and category summary. Run with python3 -I."""
import re, glob, os, sys, json, statistics as st, math, collections
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(d + "/*.err")):
    m = re.match(r"(off|on)-(.*)-(\d+)\.err$", os.path.basename(f))
    mode, row, rep = m[1], m[2], int(m[3])
    t = open(f).read()
    ids = re.search(r"^ids: (\[.*\])$", t, re.M)
    fin = re.search(r"^finish=(\S+) prompt=(\d+) \(cached (\d+)\) completion=(\d+) \| prefill (\d+) ms .*\| decode (\d+) ms \(([\d.]+) tok/s\)", t, re.M)
    sp = re.search(r"spec: verify passes=(\d+) proposed=(\d+) accepted=(\d+)", t)
    rows[row][mode].append(dict(ids=ids[1] if ids else None, fin=fin[1], prompt=int(fin[2]), comp=int(fin[4]), dec_ms=int(fin[6]), tps=float(fin[7]),
                               passes=int(sp[1]) if sp else 0, prop=int(sp[2]) if sp else 0, acc=int(sp[3]) if sp else 0, rep=rep))
out = []
P = out.append
P("| row | cat | prompt tok | gen tok | off tok/s (3 reps) | on tok/s (3 reps) | on/off (median) | on/off (pooled ms) | passes | proposed | accepted | acceptance | ids identical |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
cat = collections.defaultdict(list); cat_pooled = collections.defaultdict(list); allok = True; summary = {}
for row in sorted(rows, key=lambda r: (not r.startswith("grounded"), r)):
    off, on = rows[row]["off"], rows[row]["on"]
    ids = {x["ids"] for x in off + on}
    same = len(ids) == 1 and None not in ids
    allok &= same
    mo, mn = st.median(x["tps"] for x in off), st.median(x["tps"] for x in on)
    ratio = mn / mo
    pooled = sum(x["dec_ms"] for x in off) / sum(x["dec_ms"] for x in on)
    c = "grounded" if row.startswith("grounded") else "open"
    cat[c].append(ratio); cat_pooled[c].append(pooled)
    pas = [x["passes"] for x in on]; prop = [x["prop"] for x in on]; acc = [x["acc"] for x in on]
    summary[row] = dict(cat=c, off=[x["tps"] for x in off], on=[x["tps"] for x in on], ratio=ratio, pooled=pooled, same=same, gen=off[0]["comp"],
                        passes=pas[0], prop=prop[0], acc=acc[0])
    P(f"| {row} | {c} | {off[0]['prompt']} | {off[0]['comp']} | {mo:.1f} ({'/'.join(f'{x['tps']:.1f}' for x in off)}) | {mn:.1f} ({'/'.join(f'{x['tps']:.1f}' for x in on)}) | "
      f"**{ratio:.2f}x** | {pooled:.2f}x | {pas[0]} | {prop[0]} | {acc[0]} | {100*acc[0]/prop[0] if prop[0] else 0:.0f}% | {'yes' if same else '**NO**'} |")
P("")
P("| category | rows | mean of ratios | geomean | median | min | max | rows >= target |")
P("|---|---|---|---|---|---|---|---|")
tg = {"grounded": 1.3, "open": 0.97}
for c in ("grounded", "open"):
    r = cat[c]
    P(f"| {c} | {len(r)} | {st.mean(r):.3f}x | {math.exp(st.mean(math.log(x) for x in r)):.3f}x | {st.median(r):.3f}x | {min(r):.2f}x | {max(r):.2f}x | {sum(x >= tg[c] for x in r)}/{len(r)} (target {tg[c]}x) |")
P("")
P(f"Identical ids on every row (all reps, both modes): {'YES' if allok else 'NO'}")
print("\n".join(out))
json.dump(summary, open(os.path.join(d, "..", "g3-summary.json"), "w"), indent=1)
