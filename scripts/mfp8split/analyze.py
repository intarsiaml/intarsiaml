#!/usr/bin/env python3
"""Lane mfp8split (from mchunk4): ABBA tables from scripts/mchunk4/abba.sh output. usage: analyze.py <outdir>
Per cell and arm (A = INTARSIA_METAL_FP8_ROWS_SPLIT=0, B = split on, same binary, --max-batch 16): aggregate tok/s, TTFT p50, per-request decode tok/s p50 (each the mean
of the arm's visits, with the visits' spread), B / A ratios; for every cell the server step ms from the INTARSIA_CB_TRACE=1
lines inside the cell's time window: decode-iteration wall (prepare + verify + decode us of each Emit), mean gap between Emits
(includes prefill iterations), and the prefill iterations' mean ms."""
import collections, glob, gzip, json, os, re, statistics, sys

O = sys.argv[1]
cells = collections.defaultdict(lambda: collections.defaultdict(list))  # cell -> arm -> [summary...]
step = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sorted(glob.glob(f"{O}/*-*")):
    visit, srv = os.path.basename(d).split("-", 1)
    arm = visit[0]
    win = {}
    if os.path.exists(f"{d}/cells.txt"):
        for l in open(f"{d}/cells.txt"):
            m = re.match(r"([\d.]+) (start|end) (\S+)", l)
            if m:
                win.setdefault(m.group(3), [0, 0])[m.group(2) == "end"] = float(m.group(1))
    for f in glob.glob(f"{d}/*.json"):
        tag = os.path.basename(f)[:-5]
        if tag == "warm":
            continue
        cells[tag][arm].append(json.load(open(f))["summary"])
    err = f"{d}/server.err.gz" if os.path.exists(f"{d}/server.err.gz") else f"{d}/server.err"
    if not os.path.exists(err) or srv not in ("mb8", "mb16"):
        continue
    lines = (gzip.open(err, "rt", errors="replace") if err.endswith(".gz") else open(err, errors="replace")).read().splitlines()
    for tag, (t0, t1) in win.items():
        emits, pre = [], []
        for l in lines:
            m = re.match(r"cb-trace-us t=(\d+) (Emit|PrefillIter) \{(.*)\}", l)
            if not m:
                continue
            t = int(m.group(1)) / 1e6
            if not (t0 <= t <= t1):
                continue
            kv = dict(re.findall(r"(\w+): (\d+)", m.group(3)))
            if m.group(2) == "Emit":
                emits.append((t, (int(kv.get("prepare_us", 0)) + int(kv.get("verify_us", 0)) + int(kv.get("decode_us", 0))) / 1e3))
            else:
                pre.append(int(kv["prefill_us"]) / 1e3)
        if len(emits) > 2:
            gaps = [(b[0] - a[0]) * 1e3 for a, b in zip(emits, emits[1:])]
            step[tag][arm].append((statistics.median(e[1] for e in emits), statistics.mean(gaps), statistics.mean(pre) if pre else float("nan"), len(pre)))

mean = lambda v: sum(v) / len(v)
spread = lambda v: (max(v) - min(v)) / mean(v) * 100 if v else float("nan")
print("| cell | A agg tok/s | B agg tok/s | B / A | A TTFT p50 s | B TTFT p50 s | B / A | dec tok/s p50 A -> B | spread A / B (agg, %) | visits |")
print("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
for tag in sorted(cells, key=lambda t: (t != "null4k", int(re.sub(r"\D", "", t) or 0))):
    A, B = cells[tag]["A"], cells[tag]["B"]
    if not A or not B:
        continue
    g = lambda X, k: [x[k] for x in X]
    print(f"| {tag} | {mean(g(A,'agg_tps')):.2f} | {mean(g(B,'agg_tps')):.2f} | {mean(g(B,'agg_tps'))/mean(g(A,'agg_tps')):.3f} | "
          f"{mean(g(A,'ttft_p50')):.3f} | {mean(g(B,'ttft_p50')):.3f} | {mean(g(B,'ttft_p50'))/mean(g(A,'ttft_p50')):.3f} | "
          f"{mean(g(A,'dec_tps_p50')):.2f} -> {mean(g(B,'dec_tps_p50')):.2f} | {spread(g(A,'agg_tps')):.1f} / {spread(g(B,'agg_tps')):.1f} | {len(A)} / {len(B)} |")
print()
print("| cell | decode iteration ms p50 A -> B | mean Emit gap ms A -> B | prefill iteration ms A -> B (count per visit) |")
print("|---|---|---|---|")
for tag in sorted(step, key=lambda t: int(re.sub(r"\D", "", t) or 0)):
    A, B = step[tag]["A"], step[tag]["B"]
    if not A or not B:
        continue
    f = lambda X, i: mean([x[i] for x in X])
    print(f"| {tag} | {f(A,0):.1f} -> {f(B,0):.1f} | {f(A,1):.1f} -> {f(B,1):.1f} | {f(A,2):.1f} -> {f(B,2):.1f} ({f(A,3):.0f} / {f(B,3):.0f}) |")

# robust view (this Mac has outlier visits of +20..30 % wall from outside load, in either arm): per cell, the visits' values, median and best visit per arm,
# and the server-side decode-iteration p50 per visit (less hit by a stall than the cell's wall)
print()
print("| cell | A agg tok/s per visit | B agg tok/s per visit | median B / A | best-visit B / A | A step p50 ms per visit | B step p50 ms per visit | median step B / A |")
print("|---|---|---|---|---|---|---|---|")
med = statistics.median
for tag in sorted(cells, key=lambda t: (t != "null4k", int(re.sub(r"\D", "", t) or 0))):
    A, B = [x["agg_tps"] for x in cells[tag]["A"]], [x["agg_tps"] for x in cells[tag]["B"]]
    sA, sB = [x[0] for x in step[tag]["A"]], [x[0] for x in step[tag]["B"]]
    if not A or not B:
        continue
    fmt = lambda v, p: " ".join(f"{x:.{p}f}" for x in v)
    print(f"| {tag} | {fmt(A,0)} | {fmt(B,0)} | {med(B)/med(A):.3f} | {max(B)/max(A):.3f} | {fmt(sA,1)} | {fmt(sB,1)} | {(med(sB)/med(sA) if sA and sB else float('nan')):.3f} |")
