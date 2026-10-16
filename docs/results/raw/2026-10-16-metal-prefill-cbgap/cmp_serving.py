#!/usr/bin/env python3
"""Compare two harness outputs (chunkgolden --out jsonl) prompt by prompt: sampled token ids, every sampling-step logits row hash
(bitwise FNV-1a over the f32 bits), Model::forward last-row and all-rows hashes. python3 cmp_serving.py <a.jsonl> <b.jsonl> [label]
Exit 1 on any mismatch or on a prompt missing from either file."""
import json, sys
def load(p): return {r["name"]: r for r in map(json.loads, open(p))}
a, b = load(sys.argv[1]), load(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else f"{sys.argv[1]} vs {sys.argv[2]}"
bad, rows = 0, 0
for n in sorted(set(a) | set(b)):
    if n not in a or n not in b:
        print(f"MISSING {n}"); bad += 1; continue
    x, y = a[n], b[n]
    ok_ids = x["serve"]["ids"] == y["serve"]["ids"]
    hx, hy = x["serve"]["row_hashes"], y["serve"]["row_hashes"]
    ok_rows = hx == hy
    ok_last = x["forward"]["last_row_hash"] == y["forward"]["last_row_hash"]
    ok_all = x["forward"]["all_rows_hash"] == y["forward"]["all_rows_hash"]
    ok_cached = x["serve"]["cached"] == y["serve"]["cached"]
    rows += len(hx)
    if not (ok_ids and ok_rows and ok_last and ok_all and ok_cached):
        bad += 1
        print(f"DIFF {n} L={x['L']}: ids {ok_ids} rows {ok_rows} fwd-last {ok_last} fwd-all {ok_all} cached {ok_cached}")
print(f"{label}: {len(a)} prompts, {rows} sampled logits rows hashed: {'ALL IDENTICAL' if not bad else str(bad) + ' MISMATCHES'}")
sys.exit(1 if bad else 0)
