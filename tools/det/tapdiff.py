#!/usr/bin/env python3
"""Per-row tap comparison (lane det): for every tap file in DIR whose bytes differ from the most common one, print each pass's first
differing layer, the differing rows of that layer, and the rank. Tap lines: `pos=P t=T rank=R h0,h1,...  (one field per layer, one hash
per row)`. A pass where only one rank differs, in a few rows, is a one-sided wrong read in the TP all-reduce.
usage: tapdiff.py DIR [PREFIX] [--passes N]"""
import collections, hashlib, os, sys

def load(f):
    out = []
    for l in open(f):
        p = l.split()
        out.append((tuple(p[:3]), [x.split(',') for x in p[3:]] if len(p) > 3 else None))
    return out

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    d, pre = args[0], (args[1] if len(args) > 1 else '')
    npass = int(sys.argv[sys.argv.index('--passes') + 1]) if '--passes' in sys.argv else 4
    files = sorted(f for f in os.listdir(d) if f.endswith('.tap') and f.startswith(pre))
    h = {f: hashlib.md5(open(os.path.join(d, f), 'rb').read()).hexdigest() for f in files}
    c = collections.Counter(h.values())
    if not c: return print('no taps')
    major = c.most_common(1)[0][0]
    ref = next(f for f in files if h[f] == major)
    print(f'{d}: {len(files)} taps, {len(c)} distinct, {c[major]} equal to {ref}')
    R = load(os.path.join(d, ref))
    for f in files:
        if h[f] == major: continue
        print('==', f)
        shown = 0
        for (hd, L), (hd2, L2) in zip(R, load(os.path.join(d, f))):
            if hd != hd2: print('  pass headers differ', hd, hd2); break
            if not L: continue
            dif = [(li, ri) for li, (a, b) in enumerate(zip(L, L2)) for ri, (x, y) in enumerate(zip(a, b)) if x != y]
            if not dif: continue
            l0 = min(x[0] for x in dif)
            rows = sorted({r for l, r in dif if l == l0})
            print(f'  {" ".join(hd)}: first layer {l0}, rows {rows} ({len(rows)} of {len(L[l0])}), {len({x[0] for x in dif})} layers differ')
            shown += 1
            if shown >= npass: break

if __name__ == '__main__':
    main()
