#!/usr/bin/env python3
"""Lane GLR: rocprofv3 counter_collection.csv.gz files -> per kernel (fold GEMM launches only) the median over dispatches of the counter summed over
instances. usage: ctab.py <dir> [kernel-substring]"""
import csv, gzip, glob, re, collections, statistics, sys
d = sys.argv[1]; filt = sys.argv[2] if len(sys.argv) > 2 else ""
acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
for f in sorted(glob.glob(f"{d}/prof-*.csv.gz")):
    tag = re.search(r"prof-(.+?)-\d+\.csv", f).group(1)
    for r in csv.DictReader(gzip.open(f, "rt")):
        k = r["Kernel_Name"]
        if not (("a8_gemm_kernel" in k and "<true" in k) or "a8_line_kernel" in k) or filt not in k: continue
        m = re.search(r"(a8_\w+_kernel<[^>]*>)", k); name = (m.group(1).replace(", ik_bf16", "") if m else k[:50]) + " grid " + r["Grid_Size"]
        acc[(tag, name)][r["Counter_Name"]][r["Dispatch_Id"]] += float(r["Counter_Value"])
for key, cs in sorted(acc.items()):
    print(key[0], key[1])
    for cn, disp in sorted(cs.items()):
        v = list(disp.values())
        print(f"    {cn:28s} {statistics.median(v):16.0f}  (n={len(v)})")
