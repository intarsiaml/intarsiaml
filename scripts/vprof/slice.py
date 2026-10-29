#!/usr/bin/env python3
"""Lane vprof: dump K whole decode steps of a burst (same step finder as vsteps.py) as csv.gz: us from step start, us duration, stream, grid, block, name.
  slice.py trace.csv.gz <cluster_index> <first_step> <K> out.csv.gz"""
import csv, gzip, sys
src = open(__file__.replace('slice.py', 'vsteps.py')).read().split("sel = steps[lo:hi]")[0]
want_first, want_n, out_path = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], '0', '1']  # vsteps.py's own arguments
exec(src)
a, b = steps[want_first][0], steps[want_first + want_n - 1][1]
w = csv.writer(gzip.open(out_path, 'wt'))
w.writerow(['t_us', 'dur_us', 'stream', 'grid', 'block', 'name'])
for r in c[a:b]:
    w.writerow([round((r['s'] - c[a]['s']) / 1e3, 1), round(r['d'] / 1e3, 1), r['Strm'], f"{r['GrdX']}x{r['GrdY']}x{r['GrdZ']}", f"{r['BlkX']}x{r['BlkY']}", name(r)[:200]])
print('rows', b - a)
