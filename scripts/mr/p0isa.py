#!/usr/bin/env python3
"""Lane mr P0: instruction mix and resource usage of the MXFP4 rows-kernel variants in scripts/mr/mxp0.hip.
usage: p0isa.py <p0.s> <p0-res.log>      (static counts over the whole function: the loop body dominates, the warp-sum epilogue is in)
Columns: VALU instructions; fmas in dual-issue (VOPD) halves; single fmas; dual share = dual / (dual + single);
bf16-unpack-type ops (v_and/v_lshl/v_lshr); ds_bpermute; global loads; VGPR, spill, occupancy from -Rpass-analysis."""
import re
import sys

asm = open(sys.argv[1]).read()
res = open(sys.argv[2]).read()
FMA = ("v_dual_fmac_f32", "v_dual_fmamk_f32", "v_dual_fmaak_f32")
NAMES = {-1: "prod", 0: "0 template", 1: "1 f32 x (A)", 2: "2 fixed group", 3: "3 f32+fixed", 4: "4 ALU floor",
         5: "5 2 rows/barrier (B)", 6: "6 f32 + 2 rows (A+B)"}

usage = {}
for blk in re.split(r"\n(?=\S*:\d+:\d+: remark: Function Name)", res):
    m = re.search(r"Function Name: (\S+)", blk)
    if not m:
        continue
    g = lambda k: (re.search(k + r": (\d+)", blk) or [0, "?"])[1]
    usage[m.group(1)] = (g("VGPRs"), g("VGPRs Spill"), g("Occupancy \\[waves/SIMD\\]"))

rows = []
for f in re.split(r"\n(?=_Z\S+:)", asm):
    head = re.match(r"(_Z\S+):", f)
    if not head:
        continue
    name = head.group(1)
    m = re.search(r"p0_kernelILi(\d+)ELi(\d+)E", name)
    if m:
        v, M = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"mxfp4_gemv_rows_kernelILi(\d+)EfEEv", name)
        if not m:
            continue
        v, M = -1, int(m.group(1))
    valu = dual = single = unpack = bperm = gl = ff = fo = dual_lines = 0
    hist = {}
    for l in f.split("\n"):
        l = l.split(";")[0].strip()
        if not l or l.startswith(".") or l.endswith(":"):
            continue
        op = l.split()[0]
        hist[op] = hist.get(op, 0) + 1
        if op.startswith("v_dual_"):
            valu += 1
            nf = sum(1 for h in l.split("::") if h.split()[0] in FMA)
            dual += nf
            ff += nf == 2
            fo += nf == 1
            dual_lines += 1
        elif op.startswith("v_"):
            valu += 1
            if re.match(r"v_fma(c|mk|ak)?_f32", op):
                single += 1
            if op.startswith(("v_and_b32", "v_lshlrev_b32", "v_lshrrev_b32")):
                unpack += 1
        elif op.startswith("ds_bpermute"):
            bperm += 1
        elif op.startswith("global_load"):
            gl += 1
    u = next((x for k, x in usage.items() if k == name or name in k or k in name), ("?", "?", "?"))
    if v == -1 and M == 16:
        print("ops of prod M=16:", sorted(hist.items(), key=lambda kv: -kv[1])[:16])
    rows.append((M, v, valu, dual, single, unpack, bperm, gl, u, ff, fo, dual_lines))

print("%-22s %3s %6s %6s | %5s %5s %5s | %6s | %6s %6s %6s | %5s %5s %4s" %
      ("variant", "M", "VALU", "dualL", "fma2", "fma+o", "sglF", "fmas", "pair%", "unpack", "bperm", "VGPR", "spill", "occ"))
for M, v, valu, dual, single, unpack, bperm, gl, u, ff, fo, dl in sorted(rows):
    F = dual + single  # all fma ops; fma2 = dual lines with two fmas, fma+o = dual lines with one fma, sglF = single fma lines
    pair = 100.0 * 2 * ff / max(1, F)  # share of fmas that issue together with another fma
    print("%-22s %3d %6d %6d | %5d %5d %5d | %6d | %5.1f%% %6d %6d | %5s %5s %4s" %
          (NAMES[v], M, valu, dl, ff, fo, single, F, pair, unpack, bperm, u[0], u[1], u[2]))
