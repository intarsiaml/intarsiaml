#!/usr/bin/env bash
# Lane l8, ON tatooine inside a fleetq job (--load cpu): build scripts/l8/<name>.hip into .l8bin/<name>
# (+ resource-usage log + ISA). usage: build.sh <name> [extra hipcc flags...]
# Needs ~/intarsiaml-l8-src synced (scripts/l8/sync.sh).
set -eo pipefail
cd ~/intarsiaml-l8-src
name=${1:?}; shift || true
mkdir -p .l8bin
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
hc() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/intarsiaml-l8-src:/work -w /work $IMG hipcc -O3 -std=c++17 -Wno-unused-function -Wno-unused-variable --offload-arch=gfx1201 -I kernels/hip -I kernels "$@"; }
hc -S --cuda-device-only -Rpass-analysis=kernel-resource-usage "$@" scripts/l8/$name.hip -o .l8bin/$name.s 2> .l8bin/$name-res.log || { grep -E 'error' -A4 .l8bin/$name-res.log | head -40; exit 1; }
hc "$@" scripts/l8/$name.hip kernels/hip/ik_runtime.hip -o .l8bin/$name 2>&1 | grep -E "error" || true
ls -la .l8bin/$name .l8bin/$name.s
