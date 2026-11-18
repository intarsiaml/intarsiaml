#!/usr/bin/env python3
"""postchain2: per request (index 2 = REF .. 20), arm minus REFARM over matched steps (positions >= 100), per round. usage: perreq2.py DIR REFARM [ROUND]"""
import sys, importlib.util, os, io, contextlib, statistics as st
args = sys.argv; sys.argv = args[:2]
spec = importlib.util.spec_from_file_location("an2", os.path.join(os.path.dirname(__file__), "an2.py"))
with contextlib.redirect_stdout(io.StringIO()):
    an2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(an2)
REF = args[2]; RS = [int(args[3])] if len(args) > 3 else sorted({r for a, r in an2.runs})
for r in RS:
    print(f"round {r}: requests 2(REF) 3-8(PRE) 9-10(MID) 11-20(POST); {REF} ms per step: " + " ".join(f"{an2.rmean(an2.runs[(REF, r)][0][i]):.2f}" for i in range(2, 21)))
    for (a, rr), (A, _) in sorted(an2.runs.items()):
        if rr != r or a == REF: continue
        print(f"  {a:5s} " + " ".join(f"{an2.d(A, an2.runs[(REF, r)][0], [i]):+.2f}" for i in range(2, 21)))
