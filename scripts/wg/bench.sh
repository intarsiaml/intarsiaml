#!/usr/bin/env bash
# Lane wg, ON tatooine: build and run the standalone fp8 WMMA bench in the HIP dev container (GPU: inside a fleetq slot).
#   bench.sh build          compile only (cpu)        bench.sh run [ENV=..]   run (needs the GPU slot)
set -euo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
REPO=${REPO:-$HOME/intarsiaml-wg}
mode=${1:-run}; shift || true
docker run --rm -i --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
  ${HIP_VISIBLE_DEVICES:+-e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES} ${A16:+-e A16=1} ${BF16:+-e BF16=1} ${DEV:+-e DEV=$DEV} \
  -v "$REPO":/work -w /work "$IMG" bash -c "
set -e
mkdir -p /tmp/b && cd /tmp/b
[ -f fp8.o ] && [ ! /work/kernels/hip/fp8.hip -nt fp8.o ] && [ ! /work/kernels/hip/ik_f8_rows.h -nt fp8.o ] || hipcc -c -O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 -I /work/kernels -o fp8.o /work/kernels/hip/fp8.hip
hipcc -c -O3 -std=c++17 --offload-arch=gfx1201 -I /work/kernels -o bench.o /work/scripts/wg/bench_f8.cpp
hipcc -o bench bench.o fp8.o --offload-arch=gfx1201
[ '$mode' = run ] && ./bench || echo built
"
