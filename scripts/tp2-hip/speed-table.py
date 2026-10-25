"""Lane tp2: speed table from window 2's `intarsia bench speed` JSONs (arms tp1 / tp2e / tp2g / tp2gf, rounds r1..r3; after
G2b's speed-table.py). Cells are medians over the rounds.
usage: python3 scripts/tp2-hip/speed-table.py docs/results/2026-10-25-tp2-hip/raw/w2"""
import json, sys, glob, os, statistics as st

d = sys.argv[1]
cells = {}  # (arm, size) -> list of (prefill, decode)
for f in sorted(glob.glob(os.path.join(d, 'h1-*-r*.json'))):
    arm = os.path.basename(f)[3:].rsplit('-r', 1)[0]
    j = json.load(open(f))
    for r in j.get('rows') or j.get('results') or j.get('sizes') or []:
        size = r.get('target_prompt_tokens')
        pf = r.get('prefill_tps') or r.get('prefill_tok_s') or r.get('prefill')
        dc = r.get('decode_tps') or r.get('decode_tok_s') or r.get('decode')
        if size is not None and pf is not None:
            cells.setdefault((arm, size), []).append((pf, dc))
if not cells:
    sys.exit('no rows parsed')
sizes = sorted({s for _, s in cells})
arms = [a for a in ['tp1', 'tp2e', 'tp2g', 'tp2gf'] if any((a, s) in cells for s in sizes)]
med = lambda xs: st.median(xs) if xs else float('nan')
get = lambda a, s, k: med([v[k] for v in cells.get((a, s), [])])
print('| arm | ' + ' | '.join(f'prefill {s}' for s in sizes) + ' | ' + ' | '.join(f'decode {s}' for s in sizes) + ' |')
print('|---' * (1 + 2 * len(sizes)) + '|')
for a in arms:
    print(f'| {a} (n={len(cells.get((a, sizes[0]), []))}) | ' + ' | '.join(f'{get(a, s, 0):.0f}' for s in sizes) + ' | '
          + ' | '.join(f'{get(a, s, 1):.1f}' for s in sizes) + ' |')
for x, y in [('tp2e', 'tp1'), ('tp2g', 'tp1'), ('tp2gf', 'tp1'), ('tp2g', 'tp2e'), ('tp2gf', 'tp2g')]:
    if x in arms and y in arms:
        r = [f'{(get(x, s, k) / get(y, s, k) - 1) * 100:+.1f} %' for k in (0, 1) for s in sizes]
        print(f'| {x} vs {y} | ' + ' | '.join(r) + ' |')
print()
for a in arms:
    for s in sizes:
        v = cells.get((a, s), [])
        if v:
            print(f'rounds {a} {s}: prefill ' + ', '.join(f'{p:.0f}' for p, _ in v) + '; decode ' + ', '.join(f'{q:.1f}' for _, q in v))
