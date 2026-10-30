#!/usr/bin/env bash
# Lane hload, ON tatooine: dump gfx1201 ISA of kernels/hip files (the crate's flags, device only) for the load-wait audit.
#   isa.sh <tree> <tag> [name ...]   (default: fp8 moe moe_block moe_block_grouped moe_shared nvfp4)
# Output: <tree>/hload-out/isa-<tag>/<name>.s and the resource usage in <name>.log. CPU only (a --load cpu fleetq job).
# EXTRA="-DFOO" adds hipcc flags; INC=<dir> (relative to the tree) puts that directory before kernels/hip.
set -uo pipefail
tree=${1:?tree dir}; tag=${2:?tag}; shift 2
files=${*:-"fp8 moe moe_block moe_block_grouped moe_shared nvfp4"}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p "$tree/hload-out/isa-$tag"
rc=0
for f in $files; do
  src=kernels/hip/$f.hip; [[ -n "${INC:-}" && -f "$tree/$INC/$f.hip" ]] && src=$INC/$f.hip
  docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$tree":/work -w /work "$IMG" bash -c \
    "hipcc -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 --cuda-device-only -S -Rpass-analysis=kernel-resource-usage \
     ${INC:+-I $INC} -I kernels/hip -I kernels/abi ${EXTRA:-} $src -o hload-out/isa-$tag/$f.s" > "$tree/hload-out/isa-$tag/$f.log" 2>&1 < /dev/null || rc=$?
  echo "$f exit $rc $(wc -l < "$tree/hload-out/isa-$tag/$f.s" 2>/dev/null) lines"
done
exit $rc
