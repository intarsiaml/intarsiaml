#!/usr/bin/env python3
"""Lane TPO: how much of the all-reduce time (ar2_kernel) of one rank overlaps other kernels, from a rocprofv3 kernel_trace.csv[.gz].
usage: ovl.py trace.csv.gz [--dump N]"""
import csv, gzip, re, sys, collections
f = sys.argv[1]
op = gzip.open if f.endswith('.gz') else open
rows = collections.defaultdict(list)
for r in csv.DictReader(op(f, 'rt')):
    rows[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'], r['Stream_Id'], r['Queue_Id']))
for ag in sorted(rows):
    ev = sorted(rows[ag])
    ar = [e for e in ev if 'ar2_kernel' in e[2]]
    gem = [e for e in ev if re.search('a8_gemm_kernel', e[2])]
    other = [e for e in ev if e not in ar]
    span = ev[-1][1] - ev[0][0]
    # time of ar intervals covered by any other kernel interval
    oth = sorted((e[0], e[1]) for e in other)
    merged = []
    for s, t in oth:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t)
        else:
            merged.append([s, t])
    import bisect
    starts = [m[0] for m in merged]
    cov = 0; tot = 0
    for s, t, *_ in ar:
        tot += t - s
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0 and merged[i][1] > s:
            cov += max(0, min(t, merged[i][1]) - max(s, merged[i][0]))
            i -= 1
    print(f"{ag}: span {span/1e6:.1f} ms, kernels {len(ev)}, ar2 {len(ar)} launches {tot/1e6:.1f} ms total, of which overlapped by other kernels {cov/1e6:.1f} ms; streams {sorted(set(e[3] for e in ev))} queues {sorted(set(e[4] for e in ev))}")
    print(f"   a8 gemm {len(gem)} launches {sum(e[1]-e[0] for e in gem)/1e6:.1f} ms; ar2 streams {collections.Counter(e[3] for e in ar)}")
    if '--dump' in sys.argv:
        n = int(sys.argv[sys.argv.index('--dump') + 1]); t0 = ev[0][0]
        k = [i for i, e in enumerate(ev) if 'ar2_kernel' in e[2]][200]
        for e in ev[k - 6:k + n]:
            print(f"   {(e[0]-t0)/1e3:10.1f} +{(e[1]-e[0])/1e3:8.1f} s{e[3]} q{e[4]} {re.sub(r'[(].*','',e[2])[:44]}")
