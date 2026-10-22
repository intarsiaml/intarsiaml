"""Lane G2b: speed table from window 3's `intarsia bench speed` JSONs (arms tp1 / tp1eu / tp2, rounds r1..r3).
usage: python3 scripts/35b-tp2-hip/speed-table.py docs/results/2026-10-21-35b-tp2-hip/raw/w3"""
import json, sys, glob, os, statistics as st

d = sys.argv[1]
cells = {}  # (arm, size) -> list of (prefill, decode)
for f in sorted(glob.glob(os.path.join(d, 'g2b-*-r*.json'))):
    arm = os.path.basename(f)[4:].rsplit('-r', 1)[0]
    j = json.load(open(f))
    rows = j.get('rows') or j.get('results') or j.get('sizes') or []
    for r in rows:
        size = r.get('target_prompt_tokens')
        pf = r.get('prefill_tps') or r.get('prefill_tok_s') or r.get('prefill')
        dc = r.get('decode_tps') or r.get('decode_tok_s') or r.get('decode')
        if size is None or pf is None:
            continue
        cells.setdefault((arm, size), []).append((pf, dc))
if not cells:
    print('no rows parsed; keys of first json:', list(json.load(open(sorted(glob.glob(os.path.join(d, 'g2b-*-r*.json')))[0])).keys()))
    sys.exit(1)
sizes = sorted({s for _, s in cells})
arms = ['tp1', 'tp1eu', 'tp2']
med = lambda xs: st.median(xs)
print('| arm | ' + ' | '.join(f'prefill {s}' for s in sizes) + ' | ' + ' | '.join(f'decode {s}' for s in sizes) + ' |')
print('|---' * (1 + 2 * len(sizes)) + '|')
for a in arms:
    pf = [med([p for p, _ in cells.get((a, s), [])]) if cells.get((a, s)) else float('nan') for s in sizes]
    dc = [med([q for _, q in cells.get((a, s), [])]) if cells.get((a, s)) else float('nan') for s in sizes]
    spread = max(max(p for p, _ in cells[(a, s)]) - min(p for p, _ in cells[(a, s)]) for s in sizes if (a, s) in cells)
    print(f'| {a} (n={len(cells.get((a, sizes[0]), []))}) | ' + ' | '.join(f'{x:.0f}' for x in pf) + ' | ' + ' | '.join(f'{x:.1f}' for x in dc) + ' |')
for b in ['tp1', 'tp1eu']:
    r = []
    for kind in (0, 1):
        for s in sizes:
            x = med([v[kind] for v in cells[('tp2', s)]]); y = med([v[kind] for v in cells[(b, s)]])
            r.append(f'{(x / y - 1) * 100:+.1f} %')
    print(f'| tp2 vs {b} | ' + ' | '.join(r) + ' |')
print()
for a in arms:
    for s_ in sizes:
        v = cells.get((a, s_), [])
        if v:
            pf = [x for x, _ in v]; dc = [y for _, y in v]
            print(f'spread {a} {s_}: prefill {min(pf):.0f}-{max(pf):.0f} ({(max(pf) / min(pf) - 1) * 100:.1f} %), decode {min(dc):.1f}-{max(dc):.1f} ({max(dc) - min(dc):.1f} tok/s)')
