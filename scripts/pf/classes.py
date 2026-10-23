#!/usr/bin/env python3
"""Lane pf: sum the kernel ranking (scripts/s1/rank.py output) of nsys profiles into kernel classes.
   classes.py <dir with <arm>-<chars>/rank.txt> arm1 arm2 ... --chars 14000"""
import re, sys
d = sys.argv[1]; chars = sys.argv[sys.argv.index("--chars") + 1]
arms = [a for a in sys.argv[2:] if not a.startswith("--") and a != chars]
CL = [("routed experts (moe_w4a16 + moe_mma)", r"^moe_w4a16|^moe_mma"), ("moe topk/group/reduce/combine", r"^moe_(topk|group|reduce|combine)"),
      ("fp8 projection GEMMs", r"gemm_reg|nvjet|sm89_xmma|cutlass|splitKreduce|gemvx"), ("activation quantiser", r"actq"),
      ("shared expert nvfp4 (nvfp4_reg + wq_gemm)", r"nvfp4_reg|wq::wq_gemm"), ("GDN recurrence", r"^gdn_delta"), ("GDN conv/gate", r"^gdn_conv|^gate_split|gated_rmsnorm"),
      ("flash attention", r"^flash"), ("rmsnorm", r"^rmsnorm"), ("rope/kv/sigmoid/other", r".")]
res = {}
for a in arms:
    tot = 0; cls = {c: 0.0 for c, _ in CL}; span = None
    for line in open(f"{d}/{a}-{chars}/rank.txt"):
        m = re.match(r"kernel total ([\d.]+) ms", line)
        if m: tot = float(m.group(1)); continue
        m = re.match(r"(\S.*?)\s{2,}([\d.]+) ms\s+[\d.]+ %\s+(\d+) calls", line)
        if not m: continue
        name, t = m.group(1), float(m.group(2))
        for c, pat in CL:
            if re.search(pat, name): cls[c] += t; break
    # the ranking lists the top 24 groups only: remainder is "other"
    cls["rope/kv/sigmoid/other"] += max(0.0, tot - sum(cls.values()))
    res[a] = (tot, cls)
print("| class | " + " | ".join(f"{a} ms (%)" for a in arms) + " |"); print("|---|" + "---|" * len(arms))
for c, _ in CL:
    print(f"| {c} | " + " | ".join(f"{res[a][1][c]:.1f} ({100*res[a][1][c]/res[a][0]:.1f})" for a in arms) + " |")
print("| **kernel total** | " + " | ".join(f"**{res[a][0]:.1f}**" for a in arms) + " |")
