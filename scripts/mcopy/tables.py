#!/usr/bin/env python3
"""mcopy: tables from ab.sh output. python3 tables.py <dir with <arm>-r<round>.jsonl> [base-arm]
Per cell and arm: median decode tok/s over all reps of all rounds [min-max], ratio to the base arm (median of the per-round ratios of
the round's median tok/s: base and arm of the same round ran back to back), and whether every rep's reply sha equals the base's."""
import collections, glob, json, os, re, statistics, sys

d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "base"
rows = collections.defaultdict(list)  # (arm, round, cell) -> [record]
for f in sorted(glob.glob(os.path.join(d, "*-r*.jsonl"))):
    arm, rnd = re.match(r"(.+)-r(\d+)\.jsonl", os.path.basename(f)).groups()
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows[(arm, int(rnd), r["cell"])].append(r)
arms = sorted({k[0] for k in rows}, key=lambda a: (a != base, a)); cells = []
for k in rows:
    if k[2] not in cells:
        cells.append(k[2])
rounds = sorted({k[1] for k in rows})
med = lambda xs: statistics.median(xs) if xs else float("nan")
print(f"| cell | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
worst = {}
for c in cells:
    out = []
    for a in arms:
        v = [r["dec_tps"] for rn in rounds for r in rows.get((a, rn, c), [])]
        if not v:
            out.append("-"); continue
        ratios = [med([r["dec_tps"] for r in rows[(a, rn, c)]]) / med([r["dec_tps"] for r in rows[(base, rn, c)]])
                  for rn in rounds if rows.get((a, rn, c)) and rows.get((base, rn, c))]
        # same round and rep: the first user message carries a per-round nonce, so replies of different rounds are different prompts
        same = all(r["sha"] == rows[(base, rn, c)][i]["sha"] for rn in rounds for i, r in enumerate(rows.get((a, rn, c), []))
                   if rows.get((base, rn, c)) and i < len(rows[(base, rn, c)]))
        if a == base:
            out.append(f"{med(v):.1f} [{min(v):.0f}-{max(v):.0f}]")
        else:
            x = med(ratios); worst[a] = min(worst.get(a, 9), x)
            out.append(f"{med(v):.1f} [{min(v):.0f}-{max(v):.0f}] **{x:.3f}x** ({', '.join(f'{q:.3f}' for q in ratios)}){'' if same else ' IDS DIFFER'}")
    print(f"| {c} | " + " | ".join(out) + " |")
print("\nlowest cell ratio per arm: " + ", ".join(f"{a} {x:.3f}" for a, x in worst.items()))
