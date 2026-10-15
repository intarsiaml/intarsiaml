#!/usr/bin/env python3
"""an.py PERFDIR OUT.md : per-rep TTFT (localeval) with GPU MHz (macmon) over the prefill window, base vs new, per size.
Rep line stamp = when localeval printed it (after the first token+decode); the prefill window is [stamp-decode_s-ttft, stamp-decode_s]."""
import sys, re, json, glob, statistics as st, datetime as dt, collections
D = sys.argv[1]; import os; GEN = int(os.environ.get("GEN", "32"))
mac = []
for l in open(f"{D}/macmon.jsonl"):
    try:
        j = json.loads(l)
        mac.append((dt.datetime.fromisoformat(j["timestamp"]).timestamp(), j["gpu_usage"][0]))
    except Exception:
        pass
def mhz(a, b):
    v = [m for t, m in mac if a <= t <= b]
    return sum(v) / len(v) if v else float("nan")
rows = []
for f in sorted(glob.glob(f"{D}/p*-localeval.txt")):
    pass_ = re.search(r"/(p\d+)-(\w+)-localeval", f)
    pn, build = pass_.group(1), pass_.group(2)
    for l in open(f):
        m = re.match(r"(\d+\.\d+) .*?\] +(\d+) tok rep (\d+)/\d+: prompt=(\d+) ttft=([\d.]+)s prefill=\d+ tok/s decode=([\d.]+) tok/s", l)
        if not m:
            continue
        ts, size, rep, plen, ttft, dec = float(m[1]), int(m[2]), int(m[3]), int(m[4]), float(m[5]), float(m[6])
        end = ts - GEN / dec
        rows.append(dict(pass_=pn, build=build, size=size, rep=rep, plen=plen, ttft=ttft, mhz=mhz(end - ttft, end)))
json.dump(rows, open(f"{D}/rows.json", "w"), indent=1)
g = collections.defaultdict(list)
for r in rows:
    g[(r["size"], r["build"])].append(r)
out = ["| size | build | n | prompt tok | TTFT s median (min-max) | GPU MHz mean | TTFT s median, clock >= 1450 MHz (n) | TTFT s scaled to 1600 MHz, median |", "|---|---|---|---|---|---|---|---|"]
for (s, b) in sorted(g):
    rs = g[(s, b)]
    fast = [r["ttft"] for r in rs if r["mhz"] >= 1450]
    sc = [r["ttft"] * r["mhz"] / 1600 for r in rs]
    out.append(f"| {s} | {b} | {len(rs)} | {min(r['plen'] for r in rs)}-{max(r['plen'] for r in rs)} | {st.median(r['ttft'] for r in rs):.3f} ({min(r['ttft'] for r in rs):.2f}-{max(r['ttft'] for r in rs):.2f}) | {st.mean(r['mhz'] for r in rs):.0f} | " + (f"{st.median(fast):.3f} ({len(fast)})" if fast else "-") + f" | {st.median(sc):.3f} |")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("\n".join(out))
# per-pass medians
print()
pp = collections.defaultdict(list)
for r in rows: pp[(r["size"], r["pass_"], r["build"])].append(r)
for s in sorted({k[0] for k in pp}):
    print(s, " ".join(f"{p}-{b}:{st.median(x['ttft'] for x in pp[(s,p,b)]):.2f}@{st.mean(x['mhz'] for x in pp[(s,p,b)]):.0f}" for (s2,p,b) in sorted(pp, key=lambda k:int(k[1][1:])) if s2==s))
