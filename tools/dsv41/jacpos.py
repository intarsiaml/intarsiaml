#!/usr/bin/env python3
"""Mean indexer selected-set Jaccard per position (and min over layers) between two INTARSIA_DSV41_DUMP_POS dumps.

    uv run --no-project --with numpy python tools/dsv41/jacpos.py DIR_A DIR_B
"""
import sys,os,re,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import chunkstudy as cs
a=cs.index(sys.argv[1]); b=cs.index(sys.argv[2])
by={}
for k,f in a.items():
    if k[0].endswith('.attn.indexer') and k[1]==0 and k in b:
        by.setdefault(k[2],[]).append(cs.jac(cs.load(f),cs.load(b[k])))
for p in sorted(by): print(p, len(by[p]), round(float(np.mean(by[p])),3), round(float(np.min(by[p])),3))
