import sys, statistics as st
# usage: ps_an.py LABEL : for each request window, sum of %CPU of the top-6 consumers other than intarsia/python/macmon
# (from LABEL-ps.txt), vs prefill ms; plus the busiest foreign process names.
lab = sys.argv[1]
samples = []  # (t, load1, [(pcpu, name)])
cur = None
for l in open(lab + '-ps.txt'):
    l = l.rstrip('\n')
    if l.startswith('T '):
        p = l.split()
        cur = [float(p[1]), float(p[4]), []]
        samples.append(cur)
    elif cur is not None and l.strip():
        p = l.split(None, 2)
        try:
            cur[2].append((float(p[0]), p[2]))
        except Exception:
            pass
rows = [l.split('\t') for l in open(lab + '.tsv').read().splitlines()[1:]]
out = []
for r in rows:
    if r[1].startswith('B_') or r[1].startswith('G2'): continue
    t0 = float(r[8]); pf = int(r[4])
    w = [s for s in samples if t0 - 6 <= s[0] <= t0 + 2]
    if not w: continue
    foreign = []
    for s in w:
        foreign.append(sum(c for c, n in s[2] if not any(k in n for k in ('intarsia', 'python', 'macmon', 'Python'))))
    out.append((r[1], pf, max(foreign), st.mean(foreign), max(s[1] for s in w)))
print("cond prefill_ms foreign_cpu_max% foreign_cpu_mean% load1")
for o in sorted(out, key=lambda x: x[1]):
    print("%-16s %5d %6.0f %6.0f %5.1f" % o)
fast = [o for o in out if o[1] < 1220]; slow = [o for o in out if o[1] > 1350]
print("fast n=%d mean foreign-cpu-mean=%.0f ; slow n=%d mean foreign-cpu-mean=%.0f" % (len(fast), st.mean([o[3] for o in fast]) if fast else 0, len(slow), st.mean([o[3] for o in slow]) if slow else 0))
