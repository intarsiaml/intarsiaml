#!/usr/bin/env python3
"""Lane l8: compact table (TFLOP/s per shape) from an f8bench window.log. usage: summ.py window.log"""
import re, sys, collections

shape = None
data = collections.OrderedDict()
bad = []
for l in open(sys.argv[1]):
    m = re.match(r'^(.+?) M=(\d+) N=(\d+) K=(\d+)$', l.strip())
    if m:
        shape = m.group(1)
        data[shape] = collections.OrderedDict()
        continue
    m = re.match(r'^\s+(.+?)\s+([\d.e+]+) us\s+([\d.]+) TFLOPS (\S+)', l)
    if m and shape:
        data[shape][m.group(1)] = float(m.group(3))
        if m.group(4) != 'bitwise' and m.group(1) != 'ref' and not m.group(1).startswith('NEG'):
            bad.append((shape, m.group(1)))
print('non-NEG outputs differing from ref:', bad or 'none')
cols = [k for k in data if not k.startswith(('o M300', 'odd', 'K 1056'))]
names = list(next(iter(data.values())).keys())
print('%-24s' % 'variant', ' '.join('%9s' % c[:9] for c in cols))
for n in names:
    print('%-24s' % n, ' '.join('%9.0f' % data[c].get(n, 0) for c in cols))
