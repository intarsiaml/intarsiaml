#!/usr/bin/env python3
"""Lane mxr: static instruction mix + resource usage per kernel instance of scripts/mxr/mxr0.hip's device code.
usage: isa.py <mxr0.s> <mxr0-res.log>   (whole-function static counts; the group loop dominates)"""
import re
import sys

asm = open(sys.argv[1]).read()
res = open(sys.argv[2]).read()
usage = {}
for blk in re.split(r"\n(?=\S*:\d+:\d+: remark: Function Name)", res):
    m = re.search(r"Function Name: (\S+)", blk)
    if m:
        g = lambda k: (re.search(k + r": (\d+)", blk) or [0, "?"])[1]
        usage[m.group(1)] = (g("VGPRs"), g("VGPRs Spill"), g("Occupancy \\[waves/SIMD\\]"), g("LDS Size \\[bytes/block\\]"))
print("%-34s %6s %5s %5s %5s %5s %5s %5s %5s | %5s %5s %4s" % ("kernel", "VALU", "dot4", "fma", "perm", "vdual", "gload", "dsld", "call", "VGPR", "spill", "occ"))
for f in re.split(r"\n(?=_Z\S+:)", asm):
    head = re.match(r"(_Z\S+):", f)
    if not head:
        continue
    name = head.group(1)
    m = re.search(r"mxr_i8_rows_kernelILi(\d+)ELi(\d+)ELb(\d)EfLi(\d)E", name)
    m2 = re.search(r"mxr_i8_rows_kernelILi(\d+)ELi(\d+)ELi(\d+)ELb(\d)EfE", name)
    m3 = re.search(r"mxr_i8_m1_kernelILi(\d+)EfE", name)
    m4 = re.search(r"mxr_i8_rows_kernelILi(\d+)ELi(\d+)ELi(\d+)ELb(\d)ELi(\d+)EfE", name)
    if m:
        label = "i8 M=%s R=%s bar=%s fb=%s" % m.groups()
    elif m4:
        label = "i8 M=%s R=%s fb=%s pf=%s xp=%s" % m4.groups()
    elif m2:
        label = "i8 M=%s R=%s fb=%s pf=%s" % m2.groups()
    elif m3:
        label = "i8 m1-lds R=%s" % m3.group(1)
    elif (m := re.search(r"mxfp4_gemv_rows_kernelILi(\d+)EfEEv", name)):
        label = "prod rows M=%s" % m.group(1)
    elif "mxfp4_gemv1_lds_kernelIfE" in name:
        label = "prod M=1 lds"
    elif (m := re.search(r"mxfp4_gemv_kernelILi(\d+)EfEEv", name)):
        label = "prod gemv M=%s" % m.group(1)
    else:
        continue
    c = dict(valu=0, dot=0, fma=0, perm=0, dual=0, gl=0, ds=0, call=0)
    for l in f.split("\n"):
        l = l.split(";")[0].strip()
        if not l or l.startswith(".") or l.endswith(":"):
            continue
        op = l.split()[0]
        if op.startswith("v_"):
            c["valu"] += 1
            c["dot"] += op.startswith("v_dot4")
            c["fma"] += bool(re.match(r"v_(dual_)?fma", op)) or "fmac" in l
            c["perm"] += op.startswith("v_perm")
            c["dual"] += op.startswith("v_dual")
        elif op.startswith("global_load"):
            c["gl"] += 1
        elif op.startswith("ds_load") or op.startswith("ds_read"):
            c["ds"] += 1
        elif op.startswith("s_swappc") or op.startswith("s_call"):
            c["call"] += 1
    u = next((x for k, x in usage.items() if k == name), ("?", "?", "?", "?"))
    print("%-34s %6d %5d %5d %5d %5d %5d %5d %5d | %5s %5s %4s" % (label, c["valu"], c["dot"], c["fma"], c["perm"], c["dual"], c["gl"], c["ds"], c["call"], u[0], u[1], u[2]))
