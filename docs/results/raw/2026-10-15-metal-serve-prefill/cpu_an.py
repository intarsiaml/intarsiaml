import sys, statistics as st
# usage: cpu_an.py LABEL... : correlate prefill ms with CPU utilisation (macmon pcpu/ecpu %) and sys power during the request.
# Needs the 9-column macmon log (long1, more2, pvs1). B/G2 (idle-ramp) conditions excluded.
def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / ((sum((a - mx) ** 2 for a in x) ** .5) * (sum((b - my) ** 2 for b in y) ** .5))
allp = []
for lab in sys.argv[1:]:
    mac = []
    for l in open(lab + '-macmon.tsv'):
        p = l.strip().split('\t')
        if len(p) >= 9: mac.append(tuple(map(float, p)))
    rows = [l.split('\t') for l in open(lab + '.tsv').read().splitlines()[1:]]
    for r in rows:
        if r[1].startswith(('B_', 'G2', 'plain', 'serve_idle')): continue
        if r[1] == 'serve_b2b': t0 = float(r[4]); pf = int(r[2])
        else: t0 = float(r[8]); pf = int(r[4])
        w = [m for m in mac if t0 <= m[0] <= t0 + pf / 1e3]
        if len(w) < 4: continue
        allp.append((pf, st.mean(m[6] for m in w), st.mean(m[7] for m in w), st.mean(m[4] for m in w), st.mean(m[5] for m in w), st.mean(m[1] for m in w), lab))
pf = [a[0] for a in allp]
print(len(allp), "requests")
for i, n in ((1, 'P-cpu util %'), (2, 'E-cpu util %'), (3, 'sys power W'), (4, 'GPU power W'), (5, 'GPU MHz')):
    print("corr(prefill ms, %s during request) = %.2f" % (n, corr(pf, [a[i] for a in allp])))
bins = [(0, 1200), (1200, 1350), (1350, 1500), (1500, 9999)]
print("prefill-ms bin: n  mean P-cpu%  mean E-cpu%  mean sysW  mean GPU W  mean GPU MHz")
for lo, hi in bins:
    s = [a for a in allp if lo <= a[0] < hi]
    if s: print("%4d-%4d: %3d  %5.1f  %5.1f  %6.1f  %5.1f  %6.0f" % (lo, hi, len(s), st.mean(a[1] for a in s), st.mean(a[2] for a in s), st.mean(a[3] for a in s), st.mean(a[4] for a in s), st.mean(a[5] for a in s)))
