"""Variant sources of the shipped NVFP4 expert kernels (kernels/metal/inkling2_moe.metal) for the 1-row ceiling probe
(lane ink-dfuse, measurement only; nothing here is shipped). Each variant is the shipped file with common.h inlined and:
  SG  simdgroups per gate+up threadgroup (shipped 4)      LG  lanes per gate+up row (8)     NB  gate+up chunks ahead (1)
  LD  lanes per down row (4; rows per down threadgroup = 32 / LD)                         NBD down chunks ahead (1)
The shuffle reductions become loops over the lane count (bitwise the same at the shipped 8 / 4). LG / LD other than the
shipped change the chunk order and the reduction tree, so those variants are not bitwise with the shipped kernel.
usage: python3 -I moevar_gen.py KERNEL_DIR OUT_DIR"""
import sys, os
kd, od = sys.argv[1], sys.argv[2]
common = open(os.path.join(kd, "common.h")).read()
moe = open(os.path.join(kd, "inkling2_moe.metal")).read()
def sub(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:80])
    return s.replace(a, b)
base = sub(moe, '#include "common.h"', common)
# generic shuffle reductions
base = sub(base, """        a += simd_shuffle_xor(a, 1);
        b += simd_shuffle_xor(b, 1);
        a += simd_shuffle_xor(a, 2);
        b += simd_shuffle_xor(b, 2);
        a += simd_shuffle_xor(a, 4);
        b += simd_shuffle_xor(b, 4);""", """        for (ushort m = 1; m < INK_LG; m <<= 1) {
            a += simd_shuffle_xor(a, m);
            b += simd_shuffle_xor(b, m);
        }""")
base = sub(base, """    s += simd_shuffle_xor(s, 1);
    s += simd_shuffle_xor(s, 2);
    return s;""", """    for (ushort m = 1; m < INK_LD; m <<= 1) s += simd_shuffle_xor(s, m);
    return s;""")
# down chunks ahead apart from the gate+up ones
i0 = base.index("inline float ink_dn_row")
i1 = base.index("// down per pair.")
base = base[:i0] + base[i0:i1].replace("INK_NB", "INK_NBD") + base[i1:]
# rows per fused-down threadgroup = 32 / INK_LD
i0 = base.index("inline void ink_dn_f_impl")
i1 = base.index("// y rows -> out")
f = base[i0:i1]
f = sub(f, "const uint n = tg.y * 8u + lane / INK_LD", "const uint n = tg.y * INK_RD + lane / INK_LD")
f = sub(f, "ys[k * 8u + lane / INK_LD]", "ys[k * INK_RD + lane / INK_LD]")
f = sub(f, "if (tid < 8u && tg.y * 8u + tid < H)", "if (tid < INK_RD && tg.y * INK_RD + tid < H)")
f = sub(f, "yv[k] = ys[k * 8u + tid]", "yv[k] = ys[k * INK_RD + tid]")
f = sub(f, "(ulong)t * H + tg.y * 8u + tid", "(ulong)t * H + tg.y * INK_RD + tid")
base = base[:i0] + f + base[i1:]
consts = """constant constexpr uint INK_SG = 4;   // simdgroups per threadgroup of the expert kernels
constant constexpr uint INK_LG = 8;   // lanes per gate+up row
constant constexpr uint INK_LD = 4;   // lanes per down row
constant constexpr uint INK_NB = 1;   // chunks a lane loads ahead per step (3 measured slower: 83 -> 91 us at one NVFP4 row)"""
assert base.count(consts) == 1
DEF = dict(SG=4, LG=8, NB=1, LD=4, NBD=1, XR=4, THR=0)
VARS = {
    "ship": {}, "gu_nb2": dict(NB=2), "gu_lg16": dict(LG=16), "gu_lg16_nb2": dict(LG=16, NB=2), "gu_sg8": dict(SG=8),
    "gu_lg4": dict(LG=4), "dn_nb2": dict(NBD=2), "dn_ld8": dict(LD=8), "dn_ld8_nb2": dict(LD=8, NBD=2),
    "dn_ld16": dict(LD=16), "dn_thr256": dict(THR=256), "dn_ld8_thr256": dict(LD=8, THR=256),
    # N2: one pair per union leader (T = 1 calls: bitwise), down chunks ahead 3 / 4, combinations
    "gu_xr1": dict(XR=1), "gu_xr1_nb2": dict(XR=1, NB=2), "gu_xr1_lg16": dict(XR=1, LG=16), "dn_nb3": dict(NBD=3),
    "dn_nb4": dict(NBD=4), "xr1_dn_nb2": dict(XR=1, NBD=2), "xr1_dn_nb3": dict(XR=1, NBD=3),
    "xr1_dn_ld8_nb2": dict(XR=1, LD=8, NBD=2),
}
xr_line = "constant constexpr uint INK_XR = 4;  // pairs a union leader serves"
assert base.count(xr_line) == 1
for name, o in VARS.items():
    v = dict(DEF, **o)
    c = (f"constant constexpr uint INK_SG = {v['SG']}u;\nconstant constexpr uint INK_LG = {v['LG']}u;\n"
         f"constant constexpr uint INK_LD = {v['LD']}u;\nconstant constexpr uint INK_NB = {v['NB']}u;\n"
         f"constant constexpr uint INK_NBD = {v['NBD']}u;\nconstant constexpr uint INK_RD = 32u / INK_LD;")
    thr = f" THR {v['THR']}" if v["THR"] else ""
    src = base.replace(consts, c).replace(xr_line, f"constant constexpr uint INK_XR = {v['XR']}u;")
    hdr = f"// variant {name}: SG {v['SG']} LG {v['LG']} NB {v['NB']} LD {v['LD']} NBD {v['NBD']} XR {v['XR']}{thr}\n"
    open(os.path.join(od, f"{name}.metal"), "w").write(hdr + src)
    print(name, v)
