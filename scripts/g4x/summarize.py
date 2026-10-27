#!/usr/bin/env python3
"""G4X: summarise interleaved localeval speed rounds of scripts/g4x/le.sh:
   summarize.py <baseline-arm> <arm> [<arm> ...]       reads /opt/code/deepeval/results/*-adhoc-g4x-<arm>-r<N>/result.json
Per size: median over the rounds of decode and prefill tok/s for every arm and the ratio to the baseline arm; then, per arm, the median
over sizes of the decode ratio ('decode median') and the smallest prefill ratio. Per-round values are printed next to the medians."""
import glob, json, re, statistics, sys

base, arms = sys.argv[1], sys.argv[2:]
RES = "/opt/code/deepeval/results"

def load(arm):
    out = {}
    for f in sorted(glob.glob(f"{RES}/*-adhoc-g4x-{arm}-r[0-9]*/result.json")):
        rnd = int(re.search(r"-r(\d+)/", f).group(1))
        for r in json.load(open(f))["speed"]["rows"]:
            out.setdefault(r["target_prompt_tokens"], {})[rnd] = (r["decode_tok_s"]["median"], r["prefill_tok_s"]["median"], r["prompt_tokens"])
    return out

data = {a: load(a) for a in [base] + arms}
sizes = sorted(data[base])
rounds = sorted({r for v in data[base].values() for r in v})
print("rounds:", rounds, "  arms:", ", ".join(f"{a}={len(next(iter(data[a].values()), {}))}" for a in data))
print(f"{'size':>6} {'arm':>6} {'decode':>8} {'x base':>7} {'prefill':>8} {'x base':>7}   per-round decode | prefill")
for a in arms:
    dec, pre = [], []
    for s in sizes:
        b, x = data[base][s], data[a][s]
        bd, bp = statistics.median(v[0] for v in b.values()), statistics.median(v[1] for v in b.values())
        xd, xp = statistics.median(v[0] for v in x.values()), statistics.median(v[1] for v in x.values())
        dec.append(xd / bd); pre.append(xp / bp)
        print(f"{s:>6} {base:>6} {bd:>8.2f} {'':>7} {bp:>8.1f} {'':>7}   " + " ".join(f"{b[r][0]:.1f}" for r in sorted(b)) + " | " + " ".join(f"{b[r][1]:.0f}" for r in sorted(b)))
        print(f"{s:>6} {a:>6} {xd:>8.2f} {xd/bd:>7.3f} {xp:>8.1f} {xp/bp:>7.3f}   " + " ".join(f"{x[r][0]:.1f}" for r in sorted(x)) + " | " + " ".join(f"{x[r][1]:.0f}" for r in sorted(x)))
    print(f"==> {a}: decode ratio median over sizes {statistics.median(dec):.3f} (min {min(dec):.3f}, max {max(dec):.3f}); prefill ratio min {min(pre):.3f}, median {statistics.median(pre):.3f}")
