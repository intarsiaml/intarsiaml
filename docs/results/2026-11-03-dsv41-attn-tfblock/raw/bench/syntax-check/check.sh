#!/bin/bash
# Offline parse / device-compile check of the variants on a machine WITHOUT the CUDA toolkit (no nvcc, no ptxas, no GPU): Homebrew LLVM clang
# (`brew install llvm`, has the NVPTX backend) against the minimal stand-in headers in this directory. What it shows: every -D combination of
# the two patches parses, type-checks and lowers to PTX for sm_120, the host launch code and both bench files parse, and a few structural counts of
# the PTX. What it does NOT show: that nvcc accepts the files, anything about registers, spills or occupancy (ptxas never runs), the SASS, or
# any result. clang's PTX is not nvcc's.  usage: check.sh   (from anywhere; writes under <repo>/.worktrees/syntax-check/)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); BENCH=$HERE/..; ROOT=$(cd "$HERE/../../../../../.." && pwd)
CL=${CLANG:-/opt/homebrew/opt/llvm/bin/clang}; INC=$(echo "$(dirname "$(dirname "$CL")")"/lib/clang/*/include)
T=$ROOT/.worktrees/syntax-check; rm -rf "$T"; mkdir -p "$T"; trap 'rm -rf "$T"' EXIT
K=$ROOT/kernels/cuda
cp "$K/dsv41_attn.cu" "$T/attn_var.cu"; patch -s "$T/attn_var.cu" < "$BENCH/attn-tfblock.patch"
cp "$K/dsv41_misc.cu" "$T/misc_var.cu"; patch -s "$T/misc_var.cu" < "$BENCH/rope-inplace.patch"
cat "$K/dsv41_attn.cu" "$BENCH/occ_probe.inc" > "$T/attn_base_probe.cu"; cat "$T/attn_var.cu" "$BENCH/occ_probe.inc" > "$T/attn_var_probe.cu"
sed 's#"../../../../../kernels/abi/ik_abi.h"#"abi/ik_abi.h"#' "$BENCH/attn_ab_bench.cu" > "$T/attn_ab_bench.cu"
sed 's#"../../../../../kernels/abi/ik_abi.h"#"abi/ik_abi.h"#' "$BENCH/rope_ip_bench.cu" > "$T/rope_ip_bench.cu"
COMMON=(-x cuda --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -std=c++17 -isystem "$INC" -I "$HERE" -I "$INC" -I "$K" -I "$ROOT/kernels" -Xclang -target-feature -Xclang +ptx87 -Wno-unknown-cuda-version -Wno-unused-function)
dev() { "$CL" "${COMMON[@]}" --cuda-device-only -O2 -S -o "$T/$1.ptx" "$2" "${@:3}"; }
host() { "$CL" "${COMMON[@]}" --cuda-host-only -fsyntax-only "$1" "${@:2}"; }
echo "clang: $("$CL" --version | head -1)"
n=0
for cfg in "base|$K/dsv41_attn.cu|" "e5|$T/attn_var.cu|-DATTN_SM4" "e2|$T/attn_var.cu|-DATTN_TFBLOCK" "e2e5|$T/attn_var.cu|-DATTN_TFBLOCK -DATTN_SM4"; do
  IFS='|' read -r name src fl <<<"$cfg"; dev "attn_$name" "$src" $fl; n=$((n+1))
done
for cfg in "base|$K/dsv41_misc.cu|" "ip|$T/misc_var.cu|-DROPE_INPLACE" "ipshfl|$T/misc_var.cu|-DROPE_INPLACE_SHFL"; do
  IFS='|' read -r name src fl <<<"$cfg"; dev "misc_$name" "$src" $fl; n=$((n+1))
done
echo "device-only compile to PTX (sm_120): $n configurations ok"
host "$T/attn_base_probe.cu"; host "$T/attn_var_probe.cu" -DATTN_TFBLOCK -DATTN_SM4; host "$T/attn_var_probe.cu"; host "$T/attn_ab_bench.cu"; host "$T/rope_ip_bench.cu"
echo "host-only parse: probe-appended attention files (base, variant with and without flags), attn_ab_bench.cu, rope_ip_bench.cu ok"
python3 -I - "$T" <<'PY'
import re, sys
T = sys.argv[1]
print(f"{'ptx':14s} {'kernel':26s} {'mma.sync':>8s} {'bar.sync':>8s} {'shfl':>5s} {'ldmatrix':>8s} {'static smem B':>13s}")
for n in ['attn_base', 'attn_e5', 'attn_e2', 'attn_e2e5']:
    s = open(f'{T}/{n}.ptx').read()
    for m in re.finditer(r'\.visible \.entry (\S+?)\(', s):
        nm = m.group(1)
        k = 'attn_sparse_mma_tfb' if 'attn_sparse_mma_tfb' in nm else ('attn_sparse_mma' if re.search(r'attn_sparse_mmaE', nm) else None)
        if not k: continue
        body = s[m.start():s.find('\n}\n', m.start())]
        smem = sum(int(c) for c in re.findall(r'\.shared\s+(?:\.align\s+\d+\s+)?\.\w+\s+\S+?\[(\d+)\];', body))
        print(f"{n:14s} {k:26s} {body.count('mma.sync'):8d} {body.count('bar.sync'):8d} {body.count('shfl.sync'):5d} {body.count('ldmatrix'):8d} {smem:13d}")
PY
