#!/usr/bin/env python3
"""Lane PF27: instruction mix of every backward-branch loop of the kernels whose symbol matches a regex (hipcc -S dump)."""
import collections, re, sys
s = open(sys.argv[1]).read().split('\n'); pat = re.compile(sys.argv[2])
for i, l in enumerate(s):
    m = re.match(r'^(_Z\S*):\s', l + ' ')
    if not m or not pat.search(m.group(1)): continue
    j = i
    while j < len(s) and not s[j].startswith('.Lfunc_end'): j += 1
    body = s[i:j]; labels = {}
    for k, x in enumerate(body):
        mm = re.match(r'^(\.LBB\S+):', x.strip())
        if mm: labels[mm.group(1)] = k
    print('FUNC', m.group(1), len(body), 'lines')
    for k, x in enumerate(body):
        mm = re.search(r's_(?:cbranch_\w+|branch)\s+(\.LBB\S+)', x)
        if mm and mm.group(1) in labels and labels[mm.group(1)] < k:
            c = collections.Counter()
            for y in body[labels[mm.group(1)]:k + 1]:
                t = y.strip().split()
                if not t or t[0].startswith(('.', ';')) or t[0].endswith(':'): continue
                op = t[0]
                key = ('wmma' if op.startswith('v_wmma') else 'valu' if op.startswith('v_') else 'ds' if op.startswith('ds_') else
                       'vmem' if op.startswith(('global_', 'buffer_', 'flat_', 'scratch_')) else op if op.startswith(('s_wait', 's_barrier')) else 'salu' if op.startswith('s_') else op)
                c[key] += 1
            print('  loop', mm.group(1), dict(sorted(c.items())))
