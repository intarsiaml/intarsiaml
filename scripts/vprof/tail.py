#!/usr/bin/env python3
"""Lane vprof: cut the last K kernel clusters (idle gap >= 1 s) out of an nsys cuda_gpu_trace csv; writes them as csv.gz.
  tail.py trace_cuda_gpu_trace.csv K out.csv.gz"""
import csv, gzip, sys
rd = csv.reader(open(sys.argv[1])); hdr = next(rd); si, di = hdr.index('Start (ns)'), hdr.index('Duration (ns)'); ni = hdr.index('Name')
rows = [r for r in rd if r and not r[ni].startswith('[CUDA')]
rows.sort(key=lambda r: float(r[si]))
cl, cur, last = [], [], None
for r in rows:
    s = float(r[si])
    if last is not None and s - last >= 1e9: cl.append(cur); cur = []
    cur.append(r); last = max(last or 0, s + float(r[di]))
cl.append(cur)
print('clusters', [(len(c), round((float(c[-1][si]) - float(c[0][si])) / 1e9, 1)) for c in cl if len(c) > 200])
k = int(sys.argv[2]); keep = [r for c in cl[-k:] for r in c]
w = csv.writer(gzip.open(sys.argv[3], 'wt')); w.writerow(hdr); w.writerows(keep)
print('kept', len(keep), 'rows; header:', hdr)
