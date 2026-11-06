#!/usr/bin/env python3
"""Lane pb: which prompts prefilled together, from a serve log with INTARSIA_CB_TRACE=1. The scheduler ends at most one prefill per
iteration unless it packs (scheduler.rs `prefill_one`), so chunks of one iteration are one packed pass. Prints one line per
iteration: the chunk lengths (prompt tokens; the packed body is len - 1 for a last chunk) and the body total.
packs.py <serve.log> [prompt-len-to-name ...] (e.g. 15:p0 103:p1 17:p2 201:p3 15:p4 82:p5)"""
import re, sys
from collections import defaultdict
names = dict(a.split(':') for a in sys.argv[2:])
it = defaultdict(list)
for line in open(sys.argv[1], errors='replace'):
    m = re.search(r'Chunk \{ iter: (\d+), id: (\d+), start: (\d+), len: (\d+), last: (\w+)', line)
    if m:
        i, rid, s, n, last = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5] == 'true'
        it[i].append((rid, s, n, last))
for i in sorted(it):
    cs = it[i]
    body = sum(n - (1 if last else 0) for _, _, n, last in cs)
    desc = ' + '.join(f"{names.get(str(n), 'id' + str(rid))}({n}{'' if last else ',cont'}{'' if s == 0 else f',@{s}'})" for rid, s, n, last in cs)
    print(f"iter {i}: {'PACK ' if len(cs) > 1 else 'solo '}{desc}  body rows {body}")
