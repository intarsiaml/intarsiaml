#!/usr/bin/env python3
"""Lane mxr: VGPRs / spills / occupancy of the production int8 rows kernels from a -Rpass-analysis=kernel-resource-usage log.
usage: res.py <res.log>"""
import re
import sys

s = open(sys.argv[1]).read()
for blk in re.split(r"\n(?=\S*:\d+:\d+: remark: Function Name)", s):
    m = re.search(r"Function Name: (\S+)", blk)
    if not m:
        continue
    n = m.group(1)
    k = re.search(r"mxfp4_i8_rows_kernelILi(\d+)ELi(\d)E(\w+?)EEv", n)
    if not (k or "xprep_i8" in n or "mxfp4_gemv_rows_kernelILi8EfE" in n):
        continue
    g = lambda key: (re.search(key + r": (\d+)", blk) or [0, "?"])[1]
    label = "i8 M=%s XP=%s %s" % k.groups() if k else ("xprep_i8" if "xprep_i8" in n else "bf16 rows M=8 f32")
    print("%-26s VGPR %4s spill %3s occ %3s SGPR %3s" % (label, g("VGPRs"), g("VGPRs Spill"), g(r"Occupancy \[waves/SIMD\]"), g("SGPRs")))
