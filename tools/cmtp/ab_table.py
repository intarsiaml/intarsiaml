#!/usr/bin/env python3
"""Lane cmtp: aggregate scripts/cmtp/ab.sh output. ab_table.py <dir> <base_arm> <dev_arm>
Per cell: base / dev median over rounds (each round = median of its reps; decode cells drop the first rep), ratio dev/base of the
medians, and the worst paired round ratio. Metrics: dec* = decode tok/s per stream; pf4k = TTFT ms (ratio is base/dev so >1 is faster);
b* = aggregate tok/s (tokens / wall, prefill included)."""
import glob, json, os, re, statistics as st, sys
d, base, dev = sys.argv[1], sys.argv[2], sys.argv[3]
cells = {}
for f in sorted(glob.glob(os.path.join(d, '*-r*-*.json'))):
    m = re.match(r'(.+?)-r(\d+)-(\w+)\.json', os.path.basename(f))
    if not m: continue
    arm, r, c = m.group(1), int(m.group(2)), m.group(3)
    reps = [json.loads(l) for l in open(f) if l.startswith('{')]
    if not reps: continue
    if c.startswith('dec'):
        use = reps[1:] if len(reps) > 1 else reps
        v = st.median(x['dec_toks_per_s_per_stream_med'] for x in use)
    elif c == 'pf4k':
        v = st.median(x['ttft_med'] for x in reps[1:] or reps) * 1e3
    else:
        v = st.median(sum(x['chunks']) / x['wall'] for x in reps)
    cells.setdefault(c, {}).setdefault(arm, {})[r] = v
print(f"{'cell':8s} {'metric':14s} {base:>10s} {dev:>10s} {'dev/base':>9s} {'worst round':>12s}")
for c in sorted(cells, key=lambda x: ['dec1k','dec4k','dec16k','pf4k','b1','b4','b8','b16'].index(x) if x in ['dec1k','dec4k','dec16k','pf4k','b1','b4','b8','b16'] else 99):
    a, b = cells[c].get(base, {}), cells[c].get(dev, {})
    if not a or not b: continue
    ma, mb = st.median(a.values()), st.median(b.values())
    rr = [(a[r] / b[r] if c == 'pf4k' else b[r] / a[r]) for r in a if r in b]
    ratio = (ma / mb) if c == 'pf4k' else (mb / ma)
    unit = 'ttft ms' if c == 'pf4k' else ('tok/s/stream' if c.startswith('dec') else 'agg tok/s')
    print(f"{c:8s} {unit:14s} {ma:10.1f} {mb:10.1f} {ratio:9.4f} {min(rr):12.4f}   rounds {sorted(a)}")
