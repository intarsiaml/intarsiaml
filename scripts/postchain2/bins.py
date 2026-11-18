#!/usr/bin/env python3
"""postchain2: (arm - null) per position bin of one request index, matched steps (same drafts), averaged over the given rounds.
usage: bins.py DIR REQ NULL ARM[,ARM..] [ROUNDS e.g. 1,2]  (REQ: 9 = MID0, 10 = MID1, 11 = POST0; bins of 100 positions)"""
import sys, glob, re, statistics as st, importlib.util, os
sys.argv, args = sys.argv[:2], sys.argv
spec = importlib.util.spec_from_file_location("an2", os.path.join(os.path.dirname(__file__), "an2.py"))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    an2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(an2)
D, REQ, NUL, ARMS = args[1], int(args[2]), args[3], args[4].split(",")
RS = [int(x) for x in args[5].split(",")] if len(args) > 5 else sorted({r for a, r in an2.runs})
for arm in ARMS:
    bins = {}
    for r in RS:
        if (arm, r) not in an2.runs or (NUL, r) not in an2.runs: continue
        A, B = an2.runs[(arm, r)][0][REQ], an2.runs[(NUL, r)][0][REQ]
        for k in A:
            if k in B and A[k][0] == B[k][0]:
                bins.setdefault(k[0] // 100, []).append(A[k][1] - B[k][1])
    print(f"{arm} - {NUL}, request {REQ}, rounds {RS}: " + "  ".join(f"{b*100}:{st.mean(v):+.2f}" for b, v in sorted(bins.items())))
