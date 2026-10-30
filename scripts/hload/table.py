#!/usr/bin/env python3
"""Lane hload (from scripts/k4fp8/table.py): tables of an A/B window (scripts/hload/window.sh MODE=speed) from the localeval result files.
  python3 table.py <window dir> [<deepeval results dir>]
Reads <window dir>/steps.log for the result file of every "speed-<tp>-<base|new>-r<round>" step, prints per tp the aggregate decode tok/s
(streams 1, 4, 8, 16: stream 1 is the single-stream row) per round and arm, the median over rounds, and the new / base ratio.
With several prompt sizes (SIZES=1024,16384, MB=1, CONC=) each size's single-stream decode is a row "1@<size>"."""
import json, os, re, statistics, sys

win = sys.argv[1]
res = sys.argv[2] if len(sys.argv) > 2 else "/opt/code/deepeval/results"
runs = {}  # (tp, arm) -> {round: {streams: tok/s}}
for line in open(os.path.join(win, "steps.log")):
    m = re.search(r"speed-(tp\d)-(base|new)-r(\d+): localeval exit 0 .*results/([\w.-]+)/result.json", line)
    if not m:
        continue
    tp, arm, rd, rid = m[1], m[2], int(m[3]), m[4]
    d = json.load(open(os.path.join(res, rid, "result.json")))["speed"]
    rows = d["rows"]
    row = {1: rows[0]["decode_tok_s"]["median"]}
    if len(rows) > 1:  # several prompt sizes: one single-stream row per size
        row = {f"1@{r.get('target_prompt_tokens', i)}": r["decode_tok_s"]["median"] for i, r in enumerate(rows)}
    for c in d.get("concurrency", []):
        row[c["streams"]] = c["aggregate_decode_tok_s"]
    runs.setdefault((tp, arm), {})[rd] = row
for tp in sorted({k[0] for k in runs}):
    print(f"\n{tp}: aggregate decode tok/s (1 stream = that stream's decode tok/s), gen 512, 1k prompts")
    streams = sorted({s for arm in ("base", "new") for r in runs.get((tp, arm), {}).values() for s in r}, key=lambda s: (isinstance(s, str), s if isinstance(s, int) else int(str(s).split("@")[1])))
    print("| streams | " + " | ".join(f"{a} r{r}" for a in ("base", "new") for r in sorted(runs.get((tp, a), {}))) + " | base median | new median | new / base |")
    print("|---:|" + "---:|" * (len(sum((list(runs.get((tp, a), {})) for a in ("base", "new")), [])) + 3))
    for s in streams:
        cells, med = [], {}
        for a in ("base", "new"):
            vals = [runs[(tp, a)][r][s] for r in sorted(runs.get((tp, a), {})) if s in runs[(tp, a)][r]]
            cells += [f"{v:.1f}" for v in vals]
            med[a] = statistics.median(vals) if vals else float("nan")
        print(f"| {s} | " + " | ".join(cells) + f" | {med['base']:.1f} | {med['new']:.1f} | {med['new'] / med['base']:.3f} |")
