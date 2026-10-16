#!/usr/bin/env python3
"""serve-cold.sh summary: curl total time of the first (cold) 4062-token request after startup, per arm and gap, plus the GPU-side
commit-to-start latency of the request's first command buffer (new binaries only: 'hr' host timestamps). python3 servecoldsum.py <dir>"""
import re, sys, json, glob, statistics as st, collections
D = sys.argv[1]
t = collections.defaultdict(list)
for l in open(f"{D}/samples.txt"):
    m = re.match(r"(\w+) gap(\d+) r(\d+) load_s ([\d.]+) curl_s_code ([\d.]+) (\d+)", l)
    if m and m.group(6) == "200": t[(m.group(1), int(m.group(2)))].append((int(m.group(3)), float(m.group(5)) * 1000, float(m.group(4))))
f = lambda v: f"{st.median(v):.0f} [{min(v):.0f}..{max(v):.0f}]"
print("| gap s | arm | n | request wall ms (curl, 1 token) | load s |\n|---|---|---|---|---|")
for gap in (0, 3):
    for arm in ("base", "newoff", "new"):
        v = t.get((arm, gap), [])
        if v: print(f"| {gap} | {arm} | {len(v)} | {f([x[1] for x in v])} | {st.median([x[2] for x in v]):.1f} |")
print("\npaired by round (new - arm), request wall ms, median [min..max]\n")
for gap in (0, 3):
    for other in ("base", "newoff"):
        a = {r: w for r, w, _ in t.get(("new", gap), [])}; b = {r: w for r, w, _ in t.get((other, gap), [])}
        d = [a[r] - b[r] for r in a if r in b]
        if d: print(f"gap {gap}: new - {other}: {st.median(d):+.1f} [{min(d):+.1f}..{max(d):+.1f}] (n={len(d)})")
print("\nrequest's first command buffer: GPU start minus commit return (ms), arms with host timestamps\n")
for gap in (0, 3):
    for arm in ("newoff", "new"):
        v = []
        for fn in sorted(glob.glob(f"{D}/{arm}-gap{gap}-r*.trace.jsonl")):
            tr = [json.loads(x) for x in open(fn)]
            big = [r for r in tr if r["t"] >= 1000 and "hr" in r]
            if big: v.append(1000 * (big[0]["s"] - big[0]["hr"]))
        if v: print(f"gap {gap} {arm}: {f(v)} (n={len(v)})")
