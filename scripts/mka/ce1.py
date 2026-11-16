#!/usr/bin/env python3
"""Per-op decode table from a ce1 trace (one command buffer per dispatch); exclusive time = time after the previous buffer
ended. usage: ce1.py trace.jsonl [attn_layers]  -> totals over the decode phase, per launch and per pass (a pass = one
launch of the attention op per attention layer)."""
import json, sys, collections
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
layers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
# the prefill ends at the last flash (prefill attention) launch; MTP verify ops carry t = 3 (or 96 = 3 x 32 for folded rows)
last_pre = max(i for i, r in enumerate(recs) if r.get("t", 0) >= 1024 and r["t"] < 100000)
rs = recs[last_pre + 1:]
rs = rs[len(rs) // 8:]  # drop the warm-up eighth
t = collections.Counter(); c = collections.Counter(); end = rs[0]["s"]; idle = 0.0
for r in rs:
    s = max(r["s"], end); idle += max(0.0, r["s"] - end); t[r["op"]] += max(0.0, r["e"] - s); c[r["op"]] += r["nd"]; end = max(end, r["e"])
tot = sum(t.values())
# a pass = one launch of the main attention op per attention layer (the verify rows kernel when it runs, else the solo one)
rows_k = sum(v for k, v in c.items() if k.startswith("attn_prefill_bf16_splitkv_rows") or k.startswith("attn_decode_bf16_splitkv_rows"))
attn = rows_k if rows_k else sum(v for k, v in c.items() if k.startswith("attn_decode"))
passes = attn / layers
print(f"{sys.argv[1].split('/')[-1]}: {passes:.1f} passes, {tot*1e3/passes:.2f} ms busy / pass, {idle*1e3/passes:.2f} ms idle / pass, {sum(c.values())/passes:.0f} dispatches / pass")
print("| op | launches / pass | us / pass | share | us / launch |"); print("|---|---:|---:|---:|---:|")
for o, v in t.most_common(int(sys.argv[3]) if len(sys.argv) > 3 else 14):
    print(f"| {o} | {c[o]/passes:.1f} | {v*1e6/passes:.0f} | {100*v/tot:.1f} % | {v*1e6/c[o]:.1f} |")
