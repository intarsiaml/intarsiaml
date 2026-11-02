#!/usr/bin/env python3
"""Lane hgv: before/after of two scripts/hprof/analyze.py reports (JSON, key = rows) at 16 rows.
  profcmp.py before.json after.json [rows]"""
import json, sys

rows = sys.argv[3] if len(sys.argv) > 3 else "16"
a, b = (json.load(open(f))[rows] for f in sys.argv[1:3])
print(f"step wall {a['wall_ms']:.2f} -> {b['wall_ms']:.2f} ms, kernel busy {a['busy_ms']:.2f} -> {b['busy_ms']:.2f}, idle in step {a['idle_in_step_ms']:.2f} -> {b['idle_in_step_ms']:.2f}, "
      f"launches {a['kernels']} -> {b['kernels']}\n")
KEYS = ("gemv_kernel<16", "gemv_rows_grp_kernel", "shared_glu_rows", "shared_down_combine_rows", "ar2_kernel", "ar2ll_kernel")


def sel(r):
    out = {}
    for k in r["inst"]:
        n = k["name"]
        for key in KEYS:
            if key in n:
                out.setdefault(key, []).append((k["calls"], k["us_call"], k["ms_step"]))
    return out


sa, sb = sel(a), sel(b)
print("| kernel | before calls x us | before ms/step | after calls x us | after ms/step |\n|---|---|---:|---|---:|")
for key in KEYS:
    f = lambda v: ("; ".join(f"{int(c)} x {u:.2f}" for c, u, m in v) if v else "-", sum(m for c, u, m in v) if v else 0.0)
    (ta, ma), (tb, mb) = f(sa.get(key)), f(sb.get(key))
    if sa.get(key) or sb.get(key):
        print(f"| `{key}` | {ta} | {ma:.2f} | {tb} | {mb:.2f} |")
