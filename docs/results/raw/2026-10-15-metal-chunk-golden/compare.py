#!/usr/bin/env python3
"""Compare serving/chunk256.jsonl vs chunk2048.jsonl: ids, per-row logits hashes (bitwise), Model::forward hashes."""
import json
def load(p): return {r["name"]: r for r in map(json.loads, open(p))}
a, b = load("serving/chunk256.jsonl"), load("serving/chunk2048.jsonl")
print("| prompt | L | L%256 | L%2048 | cached 256/2048 | serving: ids | serving: logits rows (33) bitwise | Model::forward last row | Model::forward all rows |")
print("|---|---|---|---|---|---|---|---|---|")
bad = 0
for n, x in sorted(a.items(), key=lambda kv: (kv[1]["L"], kv[0])):
    y = b[n]; L = x["L"]
    ids = x["serve"]["ids"] == y["serve"]["ids"]
    hs = [i for i, (p, q) in enumerate(zip(x["serve"]["row_hashes"], y["serve"]["row_hashes"])) if p != q]
    n_rows = len(x["serve"]["row_hashes"])
    rows = f"identical ({n_rows}/{n_rows})" if not hs and n_rows == len(y["serve"]["row_hashes"]) else f"DIFF rows {hs[:6]}"
    fl = "identical" if x["forward"]["last_row_hash"] == y["forward"]["last_row_hash"] else "DIFF"
    fa = "identical" if x["forward"]["all_rows_hash"] == y["forward"]["all_rows_hash"] else "DIFF"
    bad += (not ids) + bool(hs)
    print(f"| {n} | {L} | {L%256} | {L%2048} | {x['serve']['cached']}/{y['serve']['cached']} | {'same' if ids else 'DIFF'} | {rows} | {fl} | {fa} |")
print(f"\nserving-path mismatches: {bad}")
