#!/usr/bin/env python3
"""G4X: summarise interleaved A/B rounds of scripts/g4x/ab.sh: summarize.py <ab-dir> <baseline-arm> <arm> [<arm> ...]
Reads <ab-dir>/<arm>-r<N>/g4x-*.json (bench speed). Per size: median over the rounds of decode and prefill tok/s for every arm, and the
ratio to the baseline arm; then the median over sizes of the decode ratio (the 'decode median') and the smallest prefill ratio."""
import glob, json, statistics, sys

d, base, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
def load(arm):
    out = {}
    for f in sorted(glob.glob(f"{d}/{arm}-r*/g4x-*.json")):
        for r in json.load(open(f))["rows"]:
            out.setdefault(r["target_prompt_tokens"], []).append((r["decode_tok_s"], r["prefill_tok_s"], r["prompt_tokens"]))
    return out
data = {a: load(a) for a in [base] + arms}
sizes = sorted(data[base])
print(f"rounds: " + ", ".join(f"{a}={len(next(iter(data[a].values())))}" for a in data))
print(f"{'size':>6} {'arm':>10} {'prompt':>7} {'decode':>8} {'x base':>7} {'prefill':>8} {'x base':>7}   per-round decode / prefill")
for a in arms:
    dec, pre = [], []
    for s in sizes:
        b, x = data[base][s], data[a][s]
        bd, bp = statistics.median(v[0] for v in b), statistics.median(v[1] for v in b)
        xd, xp = statistics.median(v[0] for v in x), statistics.median(v[1] for v in x)
        dec.append(xd / bd); pre.append(xp / bp)
        print(f"{s:>6} {base:>10} {statistics.median(v[2] for v in b):>7.0f} {bd:>8.2f} {'':>7} {bp:>8.1f}")
        print(f"{s:>6} {a:>10} {statistics.median(v[2] for v in x):>7.0f} {xd:>8.2f} {xd/bd:>7.3f} {xp:>8.1f} {xp/bp:>7.3f}   "
              + " ".join(f"{v[0]:.1f}" for v in x) + " / " + " ".join(f"{v[1]:.0f}" for v in x))
    print(f"{a}: decode ratio median over sizes {statistics.median(dec):.3f} (min {min(dec):.3f}, max {max(dec):.3f}); prefill ratio min {min(pre):.3f}, median {statistics.median(pre):.3f}")
