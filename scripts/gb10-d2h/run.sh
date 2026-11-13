#!/usr/bin/env bash
# Build and run the gb10-d2h repro on a GB10 box. Usage: run.sh <repo-root> <out-dir> [d2h_repro args...]
# Builds with the engine's nvcc flags (build.rs): -O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=sm_121, MAX_JOBS-safe (two objects).
set -euo pipefail
repo=$1; out=$2; shift 2
mkdir -p "$out"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
flags="-O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=${INTARSIA_CUDA_ARCH:-sm_121} -Xcompiler -fPIC,-Wall"
if [ ! -x "$out/d2h_repro" ] || [ "$repo/scripts/gb10-d2h/d2h_repro.cu" -nt "$out/d2h_repro" ]; then
  $NVCC $flags -I "$repo/kernels/cuda" -o "$out/d2h_repro" "$repo/scripts/gb10-d2h/d2h_repro.cu" "$repo/kernels/cuda/ik_runtime.cu" -lcuda -lpthread
fi
exec "$out/d2h_repro" "$@"
