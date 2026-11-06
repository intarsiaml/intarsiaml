#!/usr/bin/env python3
"""Lane fh: tables from scripts/fh/window.sh's output directory.  table.py <dir>
Per tp and stream count: arm A (packed default) against arm B (INTARSIA_PACK_PREFILL=0): aggregate tok/s (median over visits, min-max),
localeval TTFT median, ttft.py TTFT p50 / p95 pooled over visits, packed passes (A should be > 0 at 4+ streams at tp=1, B = 0), and
whether every stream's greedy text hash is the same in A and B (1 stream = the greedy-ids check)."""
import collections, glob, json, os, re, statistics, sys

d = sys.argv[1]
agg, lttft, p50, p95, passes, sha = (collections.defaultdict(list) for _ in range(6))
for f in sorted(glob.glob(os.path.join(d, "tp*-v*-*-le.json"))):
    tp, arm = re.match(r"tp(\d)-v\d+-(\w)-le", os.path.basename(f)).groups()
    for c in json.load(open(f))["speed"].get("concurrency", []):
        if c.get("aggregate_decode_tok_s"):
            agg[(tp, c["streams"], arm)].append(c["aggregate_decode_tok_s"])
        if c.get("ttft_s"):
            lttft[(tp, c["streams"], arm)].append(c["ttft_s"]["median"])
for f in sorted(glob.glob(os.path.join(d, "tp*-v*-*-ttft.json"))):
    tp, arm = re.match(r"tp(\d)-v\d+-(\w)-ttft", os.path.basename(f)).groups()
    for r in json.load(open(f))["rows"]:
        k = (tp, r["streams"], arm)
        p50[k].append(r["ttft_p50"]); p95[k].append(r["ttft_p95"]); passes[k].append(r["pack_passes"]); sha[k].append(json.dumps(r["sha"], sort_keys=True))

def med(v):
    return f"{statistics.median(v):.2f} ({min(v):.2f}-{max(v):.2f}, n={len(v)})" if v else "-"

print("| tp | streams | tok/s A | tok/s B | A vs B | TTFT p50 A / B (s) | TTFT p95 A / B (s) | LE TTFT median A / B (s) | packed passes A / B | text A == B |")
print("|---|---|---|---|---|---|---|---|---|---|")
for tp in sorted({k[0] for k in agg}, reverse=True):
    for n in sorted({k[1] for k in agg if k[0] == tp}):
        a, b = agg[(tp, n, "A")], agg[(tp, n, "B")]
        rel = f"{(statistics.median(a) / statistics.median(b) - 1) * 100:+.1f} %" if a and b else "-"
        sa, sb = set(sha[(tp, n, "A")]), set(sha[(tp, n, "B")])
        same = "yes" if sa and sb and len(sa | sb) == 1 else ("NO" if sa and sb else "-")
        pa, pb = passes[(tp, n, "A")], passes[(tp, n, "B")]
        print(f"| {tp} | {n} | {med(a)} | {med(b)} | {rel} | {med(p50[(tp,n,'A')])[:4]} / {med(p50[(tp,n,'B')])[:4]} | {med(p95[(tp,n,'A')])[:4]} / {med(p95[(tp,n,'B')])[:4]} "
              f"| {med(lttft[(tp,n,'A')])[:4]} / {med(lttft[(tp,n,'B')])[:4]} | {sum(pa):.0f} / {sum(pb):.0f} | {same} |")
