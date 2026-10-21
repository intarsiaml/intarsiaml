#!/usr/bin/env python3
"""e2sum.py <dir>: per arm (ctx-arm-round) cb-mode step period / busy / gap and the wake / encode / launch split, CPU load
and GPU MHz in the run (macmon, busy samples), medians per arm, plus greedy stdout identity per context."""
import collections, glob, gzip, hashlib, json, os, re, statistics as st, subprocess, sys

D = sys.argv[1]
H = os.path.dirname(os.path.abspath(__file__))
per = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(D, "*-*-[0-9].jsonl.gz"))):
    name = os.path.basename(f)[:-len(".jsonl.gz")]
    if name.endswith("macmon"):
        continue
    out = subprocess.run([sys.executable, os.path.join(H, "cbgap.py"), f, "128"], capture_output=True, text=True).stdout
    m = re.search(r"period ([\d.]+) ms .*busy ([\d.]+)\s+idle gap ([\d.]+)", out)
    g = re.search(r"wake ([\d.]+)\s+encode-first-buffer ([\d.]+)\s+commit->GPU start ([\d.]+)", out)
    mm = [json.loads(l) for l in gzip.open(os.path.join(D, f"{name}-macmon.jsonl.gz"), "rt") if l.startswith("{")]
    busy = [x for x in mm if x["gpu_usage"][1] >= 0.5]
    cpu = st.median(x["cpu_usage_pct"] for x in busy) if busy else float("nan")
    mhz = st.median(x["gpu_usage"][0] for x in busy) if busy else float("nan")
    ctx, arm, r = name.split("-")
    per[(ctx, arm)].append((float(m[1]), float(m[2]), float(m[3]), *(float(x) for x in g.groups()), cpu, mhz))
    print(f"{name:18s} period {m[1]} busy {m[2]} gap {m[3]} | wake {g[1]} enc {g[2]} launch {g[3]} | cpu {cpu:.2f} MHz {mhz:.0f}")
print("\nmedians per arm: period, busy, gap, wake, enc, launch, cpu, MHz")
for k, v in sorted(per.items()):
    print(k, " ".join(f"{st.median(x[i] for x in v):.3f}" for i in range(8)))
for ctx in sorted({k[0] for k in per}):
    hs = collections.Counter(hashlib.md5(open(o, "rb").read()).hexdigest()[:8] for o in glob.glob(os.path.join(D, f"{ctx}-*-[0-9].out")))
    print(f"greedy stdout {ctx}: {dict(hs)}")
