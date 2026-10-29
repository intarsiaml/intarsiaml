#!/usr/bin/env python3
"""MMTP: MTP cb trace, decode region: period, busy, idle per verify pass, idle split by the record that follows the gap.
passgaps.py FILE [passes]   (passes default: count of verify-start records = embedding with t>=2 after a t==1 chain or catch-up)"""
import gzip, json, sys, collections
f = sys.argv[1]
op = gzip.open if f.endswith('.gz') else open
R = sorted((json.loads(l) for l in op(f, 'rt')), key=lambda r: r['s'])
# decode region: after the last prefill buffer (t > 64)
last_pf = max([i for i, r in enumerate(R) if r['t'] > 64] or [-1])
D = R[last_pf + 1:]
span = D[-1]['e'] - D[0]['s']
busy = sum(r['e'] - r['s'] for r in D)
# a verify pass = the 'moe_experts_bf16_rows' run; count runs
passes = 0; prev = None
for r in D:
    if r['op'].startswith('moe_experts_bf16_rows') and not (prev or '').startswith('moe_experts_bf16_rows'):
        passes += 1
    prev = r['op']
if len(sys.argv) > 2: passes = int(sys.argv[2])
cat = collections.defaultdict(float); cnt = collections.Counter(); wake = collections.defaultdict(float)
pe = D[0]['e']; prevop = D[0]['op']
for r in D[1:]:
    g = max(0.0, r['s'] - pe)
    key = f"{prevop[:22]} -> {r['op'][:22]} t{r['t']}"
    cat[key] += g; cnt[key] += 1
    wake[key] += max(0.0, r['ho'] - pe)
    pe = max(pe, r['e']); prevop = r['op']
print(f"{f}: records {len(D)} passes {passes}: period {span/passes*1e3:.2f} ms, busy {busy/passes*1e3:.2f} ms, idle {(span-busy)/passes*1e3:.2f} ms/pass")
for k, v in sorted(cat.items(), key=lambda kv: -kv[1])[:14]:
    print(f"  {k:60s} n/pass {cnt[k]/passes:4.1f}  idle {v/passes*1e6:6.0f} us/pass  (host open after prev end {wake[k]/cnt[k]*1e6:6.0f} us avg)")
