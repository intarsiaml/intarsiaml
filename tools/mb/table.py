#!/usr/bin/env python3
"""Lane mb: per-category decode tok/s and draft acceptance of two arms of a dt1 window.
   tools/mb/table.py <window> <arm A> <arm B>   (e.g. mb2 mtp2-on mtp2-on3)
Decode tok/s: mean of localeval's per-request decode rates (results/*dt1-<window>-<arm>-<cat>-r*/result.json in
/opt/code/deepeval). Acceptance: /metrics deltas around each file (raw/<window>/spec-metrics.txt)."""
import collections, glob, json, math, re, sys

W, A, B = sys.argv[1:4]
RAW = f"docs/results/2026-11-06-27b-dflash-hip-tp1/raw/{W}"
rates, rounds = collections.defaultdict(list), collections.defaultdict(list)
for f in glob.glob(f"/opt/code/deepeval/results/*dt1-{W}-*/result.json"):
    d = json.load(open(f))
    m = re.search(rf"dt1-{W}-(mtp\d-\w+)-(.+)-r(\d)$", d["label"])
    if not m or m.group(1) not in (A, B):
        continue
    arm, cat, _ = m.groups()
    ds = d["datasets"][0]
    rates[(arm, cat)] += [c["speed"]["decode_tok_s"] for c in ds["cases"] if (c.get("speed") or {}).get("decode_tok_s")]
    rounds[(arm, cat)].append(ds["speed"]["decode_tok_s"]["median"])
acc = collections.defaultdict(lambda: [0, 0, 0])
keys = ["intarsia_spec_steps_total", "intarsia_spec_proposed_tokens_total", "intarsia_spec_accepted_tokens_total"]
lines = open(f"{RAW}/spec-metrics.txt").read().splitlines()
for before, after in zip(lines[::2], lines[1::2]):
    arm, cat = before.split()[:2]
    x, y = (dict(re.findall(r"(intarsia_\w+) (\d+)", l)) for l in (before, after))
    for j, k in enumerate(keys):
        acc[(arm, cat)][j] += int(y[k]) - int(x[k])
print(f"| category | {A} tok/s | {B} tok/s | gain | accepted per pass | draft acceptance | requests |")
print("|---|---|---|---|---|---|---|")
g = []
for cat in sorted({c for _, c in rates}):
    a, b = (sum(rates[(x, cat)]) / len(rates[(x, cat)]) for x in (A, B))
    g.append(b / a)
    (sa, pa, aa), (sb, pb, ab) = acc[(A, cat)], acc[(B, cat)]
    print(f"| {cat} | {a:.1f} | {b:.1f} | {100 * (b / a - 1):+.1f}% | {aa / sa:.2f} / {ab / sb:.2f} | "
          f"{100 * aa / pa:.1f}% / {100 * ab / pb:.1f}% | {len(rates[(A, cat)])} / {len(rates[(B, cat)])} |")
print(f"\ngeomean gain {100 * (math.exp(sum(map(math.log, g)) / len(g)) - 1):+.1f}%")
print("per-round medians:", {f"{k[0]} {k[1]}": [round(v, 1) for v in vs] for k, vs in sorted(rounds.items())})
