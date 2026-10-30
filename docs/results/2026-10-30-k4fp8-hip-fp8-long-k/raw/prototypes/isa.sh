#!/usr/bin/env bash
# Run on the GPU box (CPU only): device assembly of the bench (tree kernels) for reading the rows kernel's inner loop.
set -uo pipefail
T=$HOME/intarsiaml-tp2-k4fp8; cd "$T"
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$T":/work -w /work "$IMG" bash -c \
  "hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 --offload-device-only -S -Wno-unused-function ${INC:+-I $INC} -I kernels/hip -I kernels/abi $* scripts/k4fp8/bench_fp8.hip -o k4fp8-out/isa-${TAG:-main}.s" < /dev/null
echo "exit $?"; ls -la k4fp8-out/isa-${TAG:-main}.s
