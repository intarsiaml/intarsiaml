import sys, collections
# usage: mac_an.py LABEL  (reads LABEL.tsv from exp.py and LABEL-macmon.tsv)
lab = sys.argv[1]
mac = []
for l in open(lab + '-macmon.tsv'):
    p = l.strip().split('\t')
    if len(p) >= 6:
        mac.append(tuple(map(float, p)))
rows = [l.split('\t') for l in open(lab + '.tsv').read().splitlines()[1:]]
def window(a, b):
    return [m for m in mac if a <= m[0] <= b]
out = []
for r in rows:
    t0 = float(r[8]); pf = int(r[4])
    w = window(t0, t0 + pf / 1e3)
    pre = window(t0 - 3, t0)
    if not w: continue
    f = [m[1] for m in w]; p = [m[5] for m in w]; u = [m[2] for m in w]
    fp = [m[1] for m in pre]; pp = [m[5] for m in pre]
    out.append((r[1], pf, sum(f)/len(f), min(f), max(f), sum(u)/len(u), sum(p)/len(p), max(p),
                (sum(fp)/len(fp) if fp else 0), (sum(pp)/len(pp) if pp else 0), w[0][3], len(w)))
print("cond pf_ms | during: meanMHz minMHz maxMHz util% meanGpuW maxW | prev3s: MHz W | temp n")
for o in sorted(out, key=lambda x: (x[0], x[1])):
    print("%-16s %5d | %5.0f %5.0f %5.0f %4.0f %5.1f %5.1f | %5.0f %5.1f | %.0f %d" % o)
