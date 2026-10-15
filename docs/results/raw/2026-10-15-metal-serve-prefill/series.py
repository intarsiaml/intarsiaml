import sys
# usage: series.py LABEL COND_PREFIX rank : print macmon samples around the rank-th slowest (rank 0) request of a condition
lab, cond, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
mac = []
for l in open(lab + '-macmon.tsv'):
    p = l.strip().split('\t')
    if len(p) >= 6: mac.append(tuple(map(float, p)))
rows = [l.split('\t') for l in open(lab + '.tsv').read().splitlines()[1:] if l.split('\t')[1].startswith(cond)]
rows.sort(key=lambda r: -int(r[4]))
r = rows[rank]
t0 = float(r[8]); pf = int(r[4])
print("request", r[1], "prefill_ms", pf, "start", r[7])
print("t_rel_s  MHz  util%  temp  sysW  gpuW  pcpu%  ecpu%  cpuW")
for m in mac:
    if t0 - 2.0 <= m[0] <= t0 + pf / 1e3 + 0.6:
        print("%6.2f %5.0f %4.0f %5.1f %5.1f %5.1f %5.0f %5.0f %5.1f" % ((m[0] - t0,) + m[1:5] + m[5:6] + (m[6:9] if len(m) > 8 else (0, 0, 0))))
