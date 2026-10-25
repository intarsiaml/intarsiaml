#!/usr/bin/env bash
# Lane wg, ON tatooine: build and run the standalone fp8 WMMA bench in the HIP dev container (GPU: inside a fleetq slot).
#   bench.sh build          compile only (cpu)        bench.sh run [ENV=..]   run (needs the GPU slot)
set -euo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
REPO=${REPO:-$HOME/intarsiaml-wg}
XFLAGS=${XFLAGS:-}
mode=${1:-run}; shift || true
BUILD=${BUILD:-$HOME/wg-build}; mkdir -p "$BUILD"
docker run --rm -i --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e XFLAGS="${XFLAGS:-}" \
  ${HIP_VISIBLE_DEVICES:+-e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES} ${A16:+-e A16=1} ${BF16:+-e BF16=1} ${FOLDS:+-e FOLDS=1} ${ABL:+-e ABL=1} ${F8R:+-e F8R=$F8R} ${DEV:+-e DEV=$DEV} \
  -v "$REPO":/work -v "$BUILD":/b -w /b "$IMG" bash -c "
set -e
if [ '$mode' = build ]; then
  hipcc -c -O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 $XFLAGS -I /work/kernels -o fp8.o /work/kernels/hip/fp8.hip
  hipcc -c -O3 -std=c++17 --offload-arch=gfx1201 $XFLAGS -I /work/kernels -o bench.o /work/scripts/wg/bench_f8.cpp
  hipcc -c -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 $XFLAGS -I /work/kernels -o matmul.o /work/kernels/hip/matmul.hip
  hipcc -o bench bench.o fp8.o matmul.o --offload-arch=gfx1201
  echo built
else
  [ -x ./bench ] || { echo 'no bench binary: run build first'; exit 2; }
  ./bench
fi
"
