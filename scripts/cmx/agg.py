#!/usr/bin/env python3
"""Lane cmx: aggregate the localeval speed results of one le-window.sh run (raw/<out>/index.txt) per arm: median of the
per-round prefill tok/s medians (min-max over rounds), TTFT, marginal prefill, decode, and the paired per-round change vs a base arm.
usage: agg.py <raw/outdir> [base-arm]   (arm = name in s27d-<arm>-r<N>)"""
import collections, json, os, re, statistics as st, sys
d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else None
runs = collections.defaultdict(dict)
for line in open(os.path.join(d, "index.txt")):
    m = re.match(r"cmx-(.+)-r(\d+): .*results/([^/]+)/result.json", line)
    if m: runs[m.group(1)][int(m.group(2))] = m.group(3)
cells = {}
for arm, rs in runs.items():
    for r, rid in rs.items():
        j = json.load(open(f"/opt/code/deepeval/results/{rid}/result.json"))
        for row in j["speed"]["rows"]:
            c = cells.setdefault((arm, row["target_prompt_tokens"]), {})
            c[r] = dict(tok=row["prefill_tok_s"]["median"], ttft=row["ttft_s"]["median"], dec=row["decode_tok_s"]["median"],
                        ptok=row["prompt_tokens"], marg=row.get("marginal_prefill_tok_s"))
sizes = sorted({k[1] for k in cells}); arms = sorted(runs)
def col(a, s, k): return [v[k] for v in cells[(a, s)].values() if v.get(k) is not None]
print("| arm | " + " | ".join(f"prefill {s}-tok tok/s median (min-max) [prompt tok]" for s in sizes) + " | " + " | ".join(f"marginal {s}" for s in sizes) + " | decode tok/s |")
print("|---|" + "---|" * (2 * len(sizes) + 1))
for a in arms:
    print(f"| {a} | " + " | ".join(f"{st.median(col(a,s,'tok')):.0f} ({min(col(a,s,'tok')):.0f}-{max(col(a,s,'tok')):.0f}) [{col(a,s,'ptok')[0]}]" for s in sizes) + " | " +
          " | ".join((f"{st.median(col(a,s,'marg')):.0f}" if col(a, s, 'marg') else "-") for s in sizes) + " | " + " / ".join(f"{st.median(col(a,s,'dec')):.2f}" for s in sizes) + " |")
if base:
    for a in arms:
        if a == base: continue
        print(f"\n{a} vs {base}: median prefill change per size (and the paired per-round changes), decode ratio")
        for s in sizes:
            rr = sorted(set(cells[(a, s)]) & set(cells[(base, s)]))
            pr = [100 * (cells[(a, s)][r]["tok"] / cells[(base, s)][r]["tok"] - 1) for r in rr]
            dr = [cells[(a, s)][r]["dec"] / cells[(base, s)][r]["dec"] for r in rr]
            print(f"  {s:>6}: medians {100*(st.median(col(a,s,'tok'))/st.median(col(base,s,'tok'))-1):+.1f} %  paired rounds {', '.join(f'{p:+.1f} %' for p in pr)}  decode x{st.median(dr):.3f}")
