"""Server-side ITL / step histograms over one real-prompt level (VISIT-levels.txt spans, Mac time -> tatooine via VISIT-clock.txt)."""
import os, re, sys
out, visit = sys.argv[1], sys.argv[2]
mac0, tat, mac1 = (float(x) for x in open(os.path.join(out, f"{visit}-clock.txt")).read().split()[:3])
off = tat - (mac0 + mac1) / 2
S = {}
for line in open(os.path.join(out, f"{visit}.poll")):
    p = line.split()
    if len(p) == 3: S.setdefault(float(p[0]), {})[p[1]] = float(p[2])
ts = sorted(t for t in S if len(S[t]) > 10)
def at(t):
    b = None
    for x in ts:
        if x <= t: b = x
    return S[b]
def h(s, name):
    return {k: v for k, v in s.items() if k.startswith(name + "_bucket")}
for line in open(os.path.join(out, f"{visit}-levels.txt")):
    rep, n, a, b = line.split(); a = float(a) + off; b = float(b) + off + 0.3
    sa, sb = at(a), at(b)
    for name in ("intarsia_batch_itl_seconds", "intarsia_batch_step_seconds"):
        ha, hb = h(sa, name), h(sb, name)
        keys = sorted(hb, key=lambda k: float("inf") if "+Inf" in k else float(re.search(r'le="([^"]+)"', k).group(1)))
        prev, row = 0, []
        for k in keys:
            c = hb[k] - ha.get(k, 0)
            le = re.search(r'le="([^"]+)"', k).group(1)
            row.append(f"{le}:{int(c - prev)}"); prev = c
        print(f"N={n} {name.split('_')[2]:5s} " + " ".join(x for x in row if not x.endswith(':0')))
