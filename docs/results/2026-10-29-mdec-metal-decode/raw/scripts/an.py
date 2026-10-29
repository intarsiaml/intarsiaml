#!/usr/bin/env python3
"""MDEC trace analyzer. an.py cb|ce1 FILE.jsonl[.gz] [steps]
cb : decode-region step period, GPU busy / idle per step, dispatches per step.
ce1: per-label GPU us per step, launches per step, bytes -> GB/s (in-stream, COMMIT_EVERY=1)."""
import gzip, json, sys, collections
mode, f = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 128
op = gzip.open if f.endswith('.gz') else open
R = [json.loads(l) for l in op(f, 'rt')]
R.sort(key=lambda r: r['s'])
last_pf = max([i for i, r in enumerate(R) if r['t'] > 64] or [-1])
D = R[last_pf + 1:]
span = D[-1]['e'] - D[0]['s']
busy = sum(r['e'] - r['s'] for r in D)
nd = sum(r['nd'] for r in D)
print(f'{f}: decode records {len(D)}, steps {steps}: period {span/steps*1e6:.0f} us, GPU busy {busy/steps*1e6:.0f} us, idle {(span-busy)/steps*1e6:.0f} us, dispatches/step {nd/steps:.1f}')
if mode == 'ce1':
    agg = collections.defaultdict(lambda: [0, 0.0, 0])
    pe = D[0]['s']
    for r in D:
        r['dur'] = max(0.0, r['e'] - max(r['s'], pe)); pe = max(pe, r['e'])
    for r in D:
        k = r['op'] + (f" {r['n']}x{r['k']}" if r['n'] else '')
        a = agg[k]; a[0] += 1; a[1] += r['dur']; a[2] += r['bytes']
    tot = sum(a[1] for a in agg.values())
    print(f"{'kernel':58s} {'n/step':>7s} {'us/step':>8s} {'share':>6s} {'MB/step':>8s} {'GB/s':>6s} {'us/launch':>9s}")
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        gbs = a[2] / a[1] / 1e9 if a[2] else 0
        print(f'{k[:58]:58s} {a[0]/steps:7.1f} {a[1]/steps*1e6:8.1f} {a[1]/tot*100:5.1f}% {a[2]/steps/1e6:8.1f} {gbs:6.0f} {a[1]/a[0]*1e6:9.1f}')
    print(f'sum {tot/steps*1e6:.0f} us/step, bytes {sum(a[2] for a in agg.values())/steps/1e6:.0f} MB/step')
