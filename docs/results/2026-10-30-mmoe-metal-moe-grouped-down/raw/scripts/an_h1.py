#!/usr/bin/env python3
"""usage: an_h1.py DIR CONFIG... -> per B, ce1 mean exclusive ms per step of MoE gate-up / down / gather (and step total)."""
import sys, collections
sys.path.insert(0, '/opt/code/intarsiaml/.worktrees/mmoe/.lane')
import mp
D = sys.argv[1]
for c in sys.argv[2:]:
    by = mp.by_b(f'{D}/ce1-{c}.jsonl', 'ce1', order=[9, 12, 16])
    for b, S in by.items():
        acc = collections.defaultdict(float)
        for st in S:
            for r in st:
                g, sub = mp.kind(r, b)
                if g == 'MoE routed experts': acc[sub] += r['dur']
                acc['total'] += r['dur']
        n = len(S)
        print(f"{c:5s} B={b:2d}: gate-up {1e3*acc['experts gate-up (fused silu*up)']/n:.2f}  down {1e3*acc['experts down']/n:.2f}  gather {1e3*acc['expert gather / grouping']/n:.2f}  step {1e3*acc['total']/n:.2f} ms (n={n})")
