import sys, statistics as st
# usage: corr.py LABEL : correlation of prefill ms with trailing-window GPU power / sys power / temp (macmon)
lab = sys.argv[1]
mac = []
for l in open(lab + '-macmon.tsv'):
    p = l.strip().split('\t')
    if len(p) >= 6:
        mac.append(tuple(map(float, p)))
rows = [l.split('\t') for l in open(lab + '.tsv').read().splitlines()[1:]]
def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / ((sum((a - mx) ** 2 for a in x) ** .5) * (sum((b - my) ** 2 for b in y) ** .5))
pf = []; feats = {}
for r in rows:
    t0 = float(r[8])
    if r[1].startswith('B_') or r[1].startswith('G2'):
        continue  # idle-ramp conditions
    w1 = [m for m in mac if t0 - 1 <= m[0] < t0]
    if not w1: continue
    pf.append(int(r[4]))
    for W in (5, 15, 30, 60, 120):
        w = [m for m in mac if t0 - W <= m[0] < t0]
        feats.setdefault(('gpuW', W), []).append(sum(m[5] for m in w) / len(w))
        feats.setdefault(('sysW', W), []).append(sum(m[4] for m in w) / len(w))
    feats.setdefault(('temp', 1), []).append(sum(m[3] for m in w1) / len(w1))
print(len(pf), 'requests (B/G2 excluded)')
for k, v in feats.items():
    print(k, 'corr with prefill ms: %.2f' % corr(v, pf))
