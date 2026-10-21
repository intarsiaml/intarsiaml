#!/usr/bin/env python3
"""absum.py <ab-dir>: interleaved A/B x3 (base-N / dev-N bench speed JSONs): per size, decode and prefill tok/s per run,
mean per arm, dev/base ratio of means and per-round paired deltas; GPU MHz / CPU load per run from the macmon logs."""
import glob, json, os, statistics as st, sys

D = sys.argv[1]
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "..", "scripts", "metal-speed-gate"))
from gate import macmon  # noqa: E402

runs = {}
for f in sorted(glob.glob(os.path.join(D, "*-[0-9].json"))):
    lab = os.path.basename(f)[:-5]
    d = json.load(open(f))
    mhz, cpu = macmon(os.path.join(D, f"{lab}-macmon.jsonl"))
    runs[lab] = ({r["target_prompt_tokens"]: r for r in d["rows"]}, mhz, cpu)
for lab, (_, mhz, cpu) in runs.items():
    print(f"{lab}: GPU {mhz:.0f} MHz, CPU {cpu:.2f}")
sizes = sorted(next(iter(runs.values()))[0])
for metric in ("decode_tok_s", "prefill_tok_s"):
    print(f"\n{metric}")
    print("| size | " + " | ".join(sorted(runs)) + " | base mean | dev mean | dev/base | paired deltas (round 1/2/3) |")
    print("|---:|" + "---:|" * (len(runs) + 4))
    for s in sizes:
        v = {k: runs[k][0][s][metric] for k in sorted(runs)}
        b = [v[k] for k in v if k.startswith("base")]
        d = [v[k] for k in v if k.startswith("dev")]
        pairs = [100 * (v[f"dev-{i}"] / v[f"base-{i}"] - 1) for i in (1, 2, 3) if f"dev-{i}" in v and f"base-{i}" in v]
        print(f"| {s} | " + " | ".join(f"{x:.1f}" for x in v.values()) + f" | {st.mean(b):.1f} | {st.mean(d):.1f} | {100 * (st.mean(d) / st.mean(b) - 1):+.1f} % | " + " / ".join(f"{p:+.1f}" for p in pairs) + " % |")
