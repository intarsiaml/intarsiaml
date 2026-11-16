#!/usr/bin/env python3
"""Summarise an INTARSIA_METAL_PROFILE=cb trace: prefill and decode phases, GPU busy / idle, dispatches, top ops.
usage: cb.py trace.jsonl [tokens_decoded]"""
import json, sys, collections
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 128
last_pre = max((i for i, r in enumerate(recs) if r["t"] >= 64), default=-1)
def phase(name, rs, per=None):
    if not rs: return
    span = rs[-1]["e"] - rs[0]["s"]; busy = 0.0; end = rs[0]["s"]
    ops = collections.Counter(); nd = 0
    for r in rs:
        s = max(r["s"], end); d = max(0.0, r["e"] - s); busy += d; end = max(end, r["e"]); ops[r["op"]] += d; nd += r["nd"]
    line = f"{name}: span {span*1e3:.1f} ms, gpu busy {busy*1e3:.1f} ms, idle {(span-busy)*1e3:.1f} ms ({100*(span-busy)/span:.1f} %), {len(rs)} cbs, {nd} dispatches"
    if per: line += f" | per token: {span*1e3/per:.2f} ms, idle {(span-busy)*1e3/per:.3f} ms, {nd/per:.0f} dispatches"
    print(line)
    print("   exclusive by cb's first op:", ", ".join(f"{o} {100*t/busy:.0f}%" for o, t in ops.most_common(8)))
phase("prefill", recs[:last_pre + 1])
phase("decode", recs[last_pre + 1:], ntok)
