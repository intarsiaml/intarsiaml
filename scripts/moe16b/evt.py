#!/usr/bin/env python3
"""Lane msd2: tabulate EVT lines of timing_rows_evict (us per launch by form x M). evt.py micro.txt"""
import re, sys, collections
t = collections.OrderedDict()
for l in open(sys.argv[1], errors="replace"):
    m = re.match(r'EVT (\w+) (\S+) (\S+)\s+M=(\d+)\s+([\d.]+) us', l)
    if m:
        k, sh, form, M, us = m.groups()
        t.setdefault((k, sh, form), {})[int(M)] = float(us)
cur = None
for (k, sh, form), d in t.items():
    if (k, sh) != cur:
        print(f"--- {k} {sh}   M: " + " ".join(f"{m:>7}" for m in sorted(d)))
        cur = (k, sh)
    print(f"  {form:<11}" + " ".join(f"{d[m]:7.1f}" for m in sorted(d)))
