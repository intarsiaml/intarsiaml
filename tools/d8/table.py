#!/usr/bin/env python3
"""Lane d8: per-category table of scripts/d8/abab.sh windows. usage: python3 -I tools/d8/table.py <raw window dir> [more dirs]
Reads steps.log (round/arm/file -> localeval result.json), pools decode tok/s per (category, size, arm) as tokens / sum of decode times,
prints on/off per round and pooled, the prompt tokens, and (from spec-metrics.txt) tokens per verify pass of the `on` arm."""
import json, os, re, sys, collections
res = "/opt/code/deepeval/results"
data = collections.defaultdict(list)  # (cat,size,arm,round) -> cases
for raw in sys.argv[1:]:
    w = os.path.basename(raw.rstrip("/"))
    for line in open(os.path.join(raw, "steps.log")):
        m = re.search(r"round (\d+) arm (\S+) (\S+): localeval exit (\d+) results/(\S+)", line)
        if not m or m.group(4) != "0":
            continue
        rr, arm, f, _, path = m.groups()
        cat, size = f.split("-")
        cases = json.load(open(os.path.join(res, path.rstrip("/").replace("/result.json", ""), "result.json")))["datasets"][0]["cases"]
        data[(cat, size, arm, f"{w}:{rr}")] += cases
def pooled(cases):
    toks = sum(c["speed"]["completion_tokens"] - 1 for c in cases)
    t = sum((c["speed"]["completion_tokens"] - 1) / c["speed"]["decode_tok_s"] for c in cases if c["speed"]["decode_tok_s"])
    return toks / t if t else float("nan")
rounds = sorted({k[3] for k in data})
cells = sorted({(k[0], k[1]) for k in data}, key=lambda x: (x[0], x[1]))
print(f"{'cell':12} {'ptok':>6} {'ctok':>5} " + " ".join(f"{'off ' + r:>11} {'on ' + r:>11} {'x':>6}" for r in rounds) + "    pooled off    on      x")
allx = collections.defaultdict(list)
for cat, size in cells:
    row, offs, ons = [], [], []
    pt = ct = 0
    for r in rounds:
        o, n = data.get((cat, size, "off", r)), data.get((cat, size, "on", r))
        if not o or not n:
            row.append(f"{'-':>11} {'-':>11} {'-':>6}"); continue
        offs += o; ons += n
        pt = sum(c["speed"]["prompt_tokens"] for c in o) / len(o); ct = sum(c["speed"]["completion_tokens"] for c in o) / len(o)
        a, b = pooled(o), pooled(n)
        row.append(f"{a:11.1f} {b:11.1f} {b / a:6.3f}")
        allx[cat].append((b / a, len(o)))
    if offs:
        a, b = pooled(offs), pooled(ons)
        print(f"{cat + '-' + size:12} {pt:6.0f} {ct:5.0f} " + " ".join(row) + f"   {a:8.1f} {b:7.1f} {b / a:6.3f}")
        allx[cat + "*pooled"].append((b / a, 1))
print()
for k in sorted(allx):
    if "*" in k:
        print(f"{k.split('*')[0]:6} {cells and ''}pooled x per size:", " ".join(f"{x:.3f}" for x, _ in allx[k]))
# tokens per verify pass for `on`
mt = [os.path.join(r, "spec-metrics.txt") for r in sys.argv[1:] if os.path.exists(os.path.join(r, "spec-metrics.txt"))]
acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
for p in mt:
    L = open(p).read().splitlines()
    for a, b, c in zip(L[0::3], L[1::3], L[2::3]):
        tag, f = a.split()
        if not tag.startswith("on"):
            continue
        g = lambda s, k: float(re.search(re.escape(k) + r"_total(?:\{[^}]*\})? (\S+)", s).group(1)) if re.search(re.escape(k) + r"_total(?:\{[^}]*\})? (\S+)", s) else 0.0
        for i, k in enumerate(("intarsia_spec_steps", "intarsia_spec_proposed_tokens", "intarsia_spec_accepted_tokens")):
            acc[f][i] += g(c, k) - g(b, k)
print("\nspeculation counters of the `on` arm (verify passes, proposed, accepted; tokens per pass = 1 + accepted / passes):")
for f, (s, p, a) in sorted(acc.items()):
    print(f"  {f:10} passes {s:7.0f} proposed {p:8.0f} accepted {a:8.0f}  tok/pass {1 + a / s if s else float('nan'):.2f}")
