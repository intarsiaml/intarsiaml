#!/usr/bin/env python3
"""Lane hg: tables from scripts/hg/ab.sh's output directory (localeval result.json per round, stall.py steady-ITL json per visit).
Usage: ab_table.py <dir>   -- prints markdown: aggregate decode tok/s (median of rounds x visits, with min-max), TTFT, steady ITL."""
import glob, json, os, re, statistics, sys, collections

d = sys.argv[1]
agg = collections.defaultdict(list)  # (arm, label) -> [tok/s]
ttft = collections.defaultdict(list)
solo = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, "v*-*-*-r*.json"))):
    m = re.match(r"v(\d+)-(\w+)-(1k|4k)-r(\d+)\.json", os.path.basename(f))
    if not m:
        continue
    visit, arm, size, rnd = m.groups()
    r = json.load(open(f))["speed"]
    for row in r.get("rows", []):
        if size == "1k" and row.get("decode_tok_s"):
            solo[(arm, "1 stream")].append(row["decode_tok_s"]["median"])
    for c in r.get("concurrency", []):
        key = (arm, f"{c['streams']} streams @{size}")
        if c.get("aggregate_decode_tok_s"):
            agg[key].append(c["aggregate_decode_tok_s"])
        if c.get("ttft_s"):
            ttft[key].append(c["ttft_s"]["median"])
itl = collections.defaultdict(list)
steady = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, "v*-*-itl*.json"))):
    m = re.match(r"v(\d+)-(\w+)-itl(\d+)\.json", os.path.basename(f))
    visit, arm, n = m.groups()
    j = json.load(open(f))
    itl[(arm, int(n))].append((j["steady_itl_ms"]["p50"], j["steady_itl_ms"]["p99"]))
    steady[(arm, int(n))].append(j["steady_agg_tok_s"])


def fmt(xs, nd=1):
    if not xs:
        return "-"
    return f"{statistics.median(xs):.{nd}f} ({min(xs):.{nd}f}-{max(xs):.{nd}f}, n={len(xs)})"


arms = ["base", "pk", "pkpack"]
names = {"base": "origin/main", "pk": "packed-prefill build, knob unset", "pkpack": "INTARSIA_PACK_PREFILL=1"}
print("| load | " + " | ".join(names[a] for a in arms) + " |")
print("|---|" + "---|" * len(arms))
for label in ["1 stream"] + [f"{n} streams @1k" for n in (4, 16)] + ["4 streams @4k"]:
    cells = []
    for a in arms:
        xs = solo[(a, label)] if label == "1 stream" else agg[(a, label)]
        cells.append(fmt(xs))
    print(f"| {label}: tok/s | " + " | ".join(cells) + " |")
print()
print("| burst TTFT median s | " + " | ".join(names[a] for a in arms) + " |")
print("|---|" + "---|" * len(arms))
for label in [f"{n} streams @1k" for n in (4, 16)] + ["4 streams @4k"]:
    print(f"| {label} | " + " | ".join(fmt(ttft[(a, label)], 2) for a in arms) + " |")
print()
print("| steady decode (stall.py), n streams | " + " | ".join(names[a] for a in arms) + " |")
print("|---|" + "---|" * len(arms))
for n in (1, 4, 16):
    cells = []
    for a in arms:
        v = itl[(a, n)]
        cells.append("-" if not v else f"{statistics.median(steady[(a, n)]):.1f} tok/s, ITL p50 {statistics.median(x[0] for x in v):.1f} / p99 {statistics.median(x[1] for x in v):.1f} ms")
    print(f"| {n} | " + " | ".join(cells) + " |")
