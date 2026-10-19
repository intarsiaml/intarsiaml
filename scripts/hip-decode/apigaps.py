#!/usr/bin/env python3
"""Host side of the GPU idle gaps: rocprofv3 run_kernel_trace.csv + run_hip_api_trace.csv of one prefill (remote.sh pfapi).
usage: apigaps.py <dir with the csvs> [min_gap_us=100] [show=6]
For the main kernel burst: the HIP API calls the host made inside the window (counts and summed time), and for the gaps of at
least min_gap_us, the host calls between the launch of the kernel before the gap and the launch of the kernel after it."""
import collections
import csv
import sys

d = sys.argv[1]
min_gap = float(sys.argv[2]) * 1e3 if len(sys.argv) > 2 else 100e3
show = int(sys.argv[3]) if len(sys.argv) > 3 else 6
ks = []
with open(f"{d}/run_kernel_trace.csv") as f:
    for r in csv.DictReader(f):
        ks.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"][:60], int(r["Correlation_Id"])))
ks.sort()
api = []
with open(f"{d}/run_hip_api_trace.csv") as f:
    for r in csv.DictReader(f):
        api.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Function"], int(r["Correlation_Id"])))
api.sort()
launch_of = {c: (s, e, fn) for s, e, fn, c in api}
# main burst: the longest run of kernels with gaps < 5 ms
bursts, cur = [], [ks[0]]
for a, b in zip(ks, ks[1:]):
    if b[0] - a[1] > 5_000_000:
        bursts.append(cur)
        cur = []
    cur.append(b)
bursts.append(cur)
main = max(bursts, key=lambda s: s[-1][1] - s[0][0])
t0, t1 = main[0][0], main[-1][1]
inwin = [a for a in api if t0 <= a[0] <= t1]
agg = collections.defaultdict(lambda: [0, 0])
for s, e, fn, _ in inwin:
    agg[fn][0] += 1
    agg[fn][1] += e - s
print(f"main burst {len(main)} kernels, {(t1 - t0) / 1e6:.2f} ms; host API calls started inside it:")
for fn, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"  {fn:28s} n {n:6d}  host time {t / 1e6:9.2f} ms")
shown = 0
lag = []
for a, b in zip(main, main[1:]):
    g = b[0] - a[1]
    if g < min_gap:
        continue
    la, lb = launch_of.get(a[3]), launch_of.get(b[3])
    if not la or not lb:
        continue
    lag.append(b[0] - lb[1])
    if shown < show:
        shown += 1
        print(f"\ngap {g / 1e3:.1f} us: {a[2]} -> {b[2]}")
        print(f"  launch of next kernel returned {(lb[1] - a[1]) / 1e3:+.1f} us after the previous kernel ended; next kernel starts {(b[0] - lb[1]) / 1e3:.1f} us after that")
        for s, e, fn, _ in api:
            if la[0] <= s <= lb[1]:
                print(f"    {(s - a[1]) / 1e3:+10.1f} us  {fn:24s} {(e - s) / 1e3:9.1f} us")
if lag:
    lag.sort()
    print(f"\n{len(lag)} gaps >= {min_gap / 1e3:.0f} us: next kernel start minus its launch-call return, median {lag[len(lag) // 2] / 1e3:.1f} us")
