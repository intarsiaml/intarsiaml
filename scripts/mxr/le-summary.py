#!/usr/bin/env python3
"""Lane mxr: summarise the w5 localeval runs (index.txt from scripts/mxr/le-window.sh).
Checks every arm: the 1-stream row has 512 completion tokens, every concurrency row has n == streams and no errors.
usage: le-summary.py <index.txt> [deepeval dir]"""
import json
import re
import statistics
import sys

idx = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "/opt/code/deepeval"
runs = {}
bad = []
for line in open(idx):
    m = re.match(r"mxr-(\w+)-tp(\d)-r(\d): (results/\S+?)/result\.json", line.strip().rstrip("`"))
    if not m:
        if line.strip():
            bad.append("unparsed / failed arm: " + line.strip())
        continue
    tree, tp, rnd, path = m.groups()
    r = json.load(open(f"{root}/{path}/result.json"))
    sp = r["speed"]
    row = sp["rows"][0]
    d = {"1": row["decode_tok_s"]["median"] if isinstance(row["decode_tok_s"], dict) else row["decode_tok_s"]}
    if row["completion_tokens"] != 512 if not isinstance(row["completion_tokens"], dict) else row["completion_tokens"].get("median") != 512:
        bad.append(f"{tree} tp{tp} r{rnd}: 1-stream completion_tokens {row['completion_tokens']}")
    for c in sp["concurrency"]:
        n = c["per_stream_decode_tok_s"]["n"]
        if n != c["streams"] or c["errors"]:
            bad.append(f"{tree} tp{tp} r{rnd}: {c['streams']} streams: n={n} errors={c['errors']}")
        d[str(c["streams"])] = c["aggregate_decode_tok_s"]
    for w in (r.get("warnings") or []) + (sp.get("warnings") or []):
        bad.append(f"{tree} tp{tp} r{rnd}: warning {w}")
    runs.setdefault((tp, tree), []).append((int(rnd), d, path))
cols = ["1", "4", "8", "16"]
print("| tp | tree | rounds | " + " | ".join(f"{c} stream{'s' if c != '1' else ''} tok/s (agg)" for c in cols) + " |")
print("|---|---|---|" + "---|" * len(cols))
mean = {}
for (tp, tree), lst in sorted(runs.items()):
    lst.sort()
    cells = []
    for c in cols:
        vals = [d[c] for _, d, _ in lst if c in d]
        mean[(tp, tree, c)] = statistics.mean(vals) if vals else float("nan")
        cells.append(f"{mean[(tp, tree, c)]:.1f} ({', '.join(f'{v:.1f}' for v in vals)})")
    print(f"| {tp} | {tree} | {len(lst)} | " + " | ".join(cells) + " |")
print()
for tp in sorted({k[0] for k in runs}):
    print(f"tp{tp} dev/base: " + ", ".join(f"{c}: {mean[(tp, 'dev', c)] / mean[(tp, 'base', c)] - 1:+.1%}" for c in cols if (tp, "dev", c) in mean and (tp, "base", c) in mean))
print()
print("checks:", "all arms complete, 512 tokens, no errors" if not bad else "\n  " + "\n  ".join(bad))
for (tp, tree), lst in sorted(runs.items()):
    for rnd, _, path in lst:
        print(f"  tp{tp} {tree} r{rnd}: {path}")
