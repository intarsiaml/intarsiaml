#!/usr/bin/env bash
# Lane hfp8, ON tatooine: compile scripts/hfp8/bench_fp8.hip (includes kernels/hip/fp8.hip) in the HIP dev image, with the
# VGPR / scratch report of every instantiation in compile-<tag>.log.   bench.sh <tree> <tag> [extra hipcc flags]
# Compile is CPU only (a --load cpu fleetq job); running hfp8-out/bench-<tag> needs a GPU window.
set -uo pipefail
tree=${1:?tree dir}; tag=${2:?tag}; shift 2
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p "$tree/hfp8-out"
rc=0
docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$tree":/work -w /work "$IMG" bash -c \
  "hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-function -Rpass-analysis=kernel-resource-usage -I kernels/hip -I kernels/abi $* scripts/hfp8/bench_fp8.hip -o hfp8-out/bench-$tag" \
  > "$tree/hfp8-out/compile-$tag.log" 2>&1 < /dev/null || rc=$?
echo "compile $tag exit $rc"
ls -la "$tree/hfp8-out/bench-$tag"
exit $rc
