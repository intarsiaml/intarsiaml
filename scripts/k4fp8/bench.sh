#!/usr/bin/env bash
# Lane k4fp8 (from lane hfp8), ON tatooine: compile scripts/k4fp8/bench_fp8.hip (includes kernels/hip/fp8.hip) in the HIP dev image, with the
# VGPR / scratch report of every instantiation in compile-<tag>.log.   bench.sh <tree> <tag> [extra hipcc flags]
# INC=<dir> puts <dir> (an ablation copy of fp8.hip) before kernels/hip. Compile is CPU only (a --load cpu fleetq job); running k4fp8-out/bench-<tag> needs a GPU window.
set -uo pipefail
tree=${1:?tree dir}; tag=${2:?tag}; shift 2
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p "$tree/k4fp8-out"
rc=0
docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$tree":/work -w /work "$IMG" bash -c \
  "hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-function -Rpass-analysis=kernel-resource-usage ${INC:+-I $INC} -I kernels/hip -I kernels/abi $* scripts/k4fp8/bench_fp8.hip -o k4fp8-out/bench-$tag" \
  > "$tree/k4fp8-out/compile-$tag.log" 2>&1 < /dev/null || rc=$?
echo "compile $tag exit $rc"
ls -la "$tree/k4fp8-out/bench-$tag"
exit $rc
