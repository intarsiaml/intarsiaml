#!/usr/bin/env python3
"""compare_chunks.py A.jsonl B.jsonl : serving-path ids + per-row logits hashes (bitwise) and Model::forward hashes of two chunkgolden runs."""
import json, sys
def load(p): return {r["name"]: r for r in map(json.loads, open(p))}
pa, pb = sys.argv[1], sys.argv[2]
a, b = load(pa), load(pb)
print(f"A = {pa}, B = {pb}")
print("| prompt | L | cached A/B | serving: ids | serving: logits rows bitwise | Model::forward last row | Model::forward all rows |")
print("|---|---|---|---|---|---|---|")
bad = 0
missing = sorted(set(a) ^ set(b))
for n, x in sorted(a.items(), key=lambda kv: (kv[1]["L"], kv[0])):
    if n not in b: continue
    y = b[n]
    ids = x["serve"]["ids"] == y["serve"]["ids"]
    hs = [i for i, (p, q) in enumerate(zip(x["serve"]["row_hashes"], y["serve"]["row_hashes"])) if p != q]
    nr = len(x["serve"]["row_hashes"])
    rows = f"identical ({nr}/{nr})" if not hs and nr == len(y["serve"]["row_hashes"]) else f"DIFF rows {hs[:6]}"
    fl = "identical" if x["forward"]["last_row_hash"] == y["forward"]["last_row_hash"] else "DIFF"
    fa = "identical" if x["forward"]["all_rows_hash"] == y["forward"]["all_rows_hash"] else "DIFF"
    bad += (not ids) + bool(hs)
    print(f"| {n} | {x['L']} | {x['serve']['cached']}/{y['serve']['cached']} | {'same' if ids else 'DIFF'} | {rows} | {fl} | {fa} |")
print(f"\nprompts compared: {len(a) - len([m for m in missing if m in a])}, only in one file: {missing}")
print(f"serving-path mismatches: {bad}")
