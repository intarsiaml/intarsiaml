#!/bin/bash
# build + admission-run planted mutants of the NAX kernels: mutants.sh NAME   (NAME: m1..m6)
cd /opt/code/intarsiaml/.worktrees/nxp-mut
git checkout -q -- kernels crates 2>/dev/null
N=$1
case $N in
 m1) sed -i '' 's/if (ct.is_valid_element(i)) ct\[i\] \*= scale;/if (ct.is_valid_element(i)) ct[i] *= 1.0f;/' kernels/metal/gemm_nax_fp8.metal ;;
 m2) python3 - <<'PY'
p='kernels/metal/fp8.metal'; s=open(p).read()
s=s.replace("    fp8_tile16(*(device const uint2*)wp, *(device const uint2*)(wp + 8), o);\n    wb[2 * (ulong)g]","    fp8_tile16(*(device const uint2*)wp, *(device const uint2*)(wp + 8), o);\n    if ((g % (p.c / 16)) == p.c / 16 - 1) for (int i = 0; i < 8; ++i) o[i] = 0;\n    wb[2 * (ulong)g]")
open(p,'w').write(s)
PY
 ;;
 m3) python3 - <<'PY'
p='kernels/metal/fp8.metal'; s=open(p).read()
s=s.replace("    fp8_tile16(*(device const uint2*)wp, *(device const uint2*)(wp + 8), o);\n    wb[2 * (ulong)g]","    fp8_tile16(*(device const uint2*)wp, *(device const uint2*)(wp + 8), o);\n    { uint t0 = o[0]; o[0] = o[1]; o[1] = t0; }\n    wb[2 * (ulong)g]")
open(p,'w').write(s)
PY
 ;;
 m4) python3 - <<'PY'
p='kernels/metal/gemm_nax_fp8.metal'; s=open(p).read()
s=s.replace("cb[i] = static_cast<bfloat>(ct[i] * scale);","cb[i] = as_type<bfloat>(ushort(as_type<uint>(ct[i] * scale) >> 16));")
open(p,'w').write(s)
PY
 ;;
 m5) sed -i '' 's/const float g = moe_rb(smem\[r \* MOE_LC + c\] \* g0), u = moe_rb(smem\[r \* MOE_LC + 32 + c\] \* g1);/const float g = moe_rb(smem[r * MOE_LC + c] * g0), u = moe_rb(smem[r * MOE_LC + 32 + c]);/' kernels/metal/moe_nax.metal ;;
 m6) sed -i '' 's/out\[(ulong)pairs\[s0 + r\] \* H + n0 + c\] = f2bf(smem\[r \* MOE_LC + c\] \* g2);/out[(ulong)pairs[s0 + r] * H + n0 + c] = f2bf(smem[r * MOE_LC + c]);/' kernels/metal/moe_nax.metal ;;
esac
git diff --stat | tail -2
CARGO_TARGET_DIR=$PWD/target cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E "^error" -A6
cp target/release/intarsia .nxp-out/mut/intarsia-$N
git checkout -q -- kernels crates
