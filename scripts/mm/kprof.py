#!/usr/bin/env python3
"""Per-kernel table of an INTARSIA_METAL_PROFILE=kernel trace from `real_35b_batch_step_real_traffic` (lane mm).

The test runs its phases (prefill, then one per B) 1.5 s apart: the trace is split at GPU gaps > 0.7 s; the first phase is
prefill. Usage: kprof.py TRACE STEPS B1,B2,...  -> per phase: per-step ms by kernel (sum of record durations / STEPS + 4
warm-up steps are included in the record count, so per step = sum / (STEPS + 4)), calls per step, and for MoeExperts the
mean distinct experts / pairs per call (labels n / k)."""
import json, sys, collections

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
steps = int(sys.argv[2]) + 4
bs = [int(b) for b in sys.argv[3].split(',')]
recs.sort(key=lambda r: r['s'])
phases, cur = [], [recs[0]]
for a, b in zip(recs, recs[1:]):
    if b['s'] - a['e'] > 0.7:
        phases.append(cur); cur = []
    cur.append(b)
phases.append(cur)
print(f"{len(phases)} phases (expect 1 prefill + {len(bs)})")
for b, ph in zip(bs, phases[-len(bs):]):
    agg = collections.defaultdict(lambda: [0, 0.0, 0, 0])
    for r in ph:
        a = agg[r['op']]
        a[0] += 1; a[1] += (r['e'] - r['s']) * 1e3
        if r['op'].startswith('moe_experts'):
            a[2] += r['n']; a[3] += r['k']
    tot = sum(a[1] for a in agg.values()) / steps
    print(f"\nB={b}: GPU sum {tot:.2f} ms/step ({len(ph)/steps:.0f} launches/step)")
    for op, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        extra = f"  distinct experts {a[2]/a[0]:.1f} of {a[3]/a[0]:.1f} pairs" if a[2] else ""
        print(f"  {op:<40} {a[0]/steps:7.1f} calls {a[1]/steps:8.3f} ms  {100*a[1]/steps/tot:5.1f}%{extra}")
