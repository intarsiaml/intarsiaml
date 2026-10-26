#!/usr/bin/env bash
# Lane mxr, ON tatooine inside a fleetq job (--load cpu): build the Phase 0 harness, its ISA and resource usage.
# usage: build.sh [harness, default mxr1]. Output in ~/intarsiaml-mxr-dev/.mxrbin: <h>, <h>.s, <h>-res.log, <h>-isa.txt
set -eo pipefail
cd ~/intarsiaml-mxr-dev
H=${1:-mxr1}
mkdir -p .mxrbin
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
hc() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/intarsiaml-mxr-dev:/work -w /work $IMG hipcc -O3 -std=c++17 -Wno-unused-function --offload-arch=gfx1201 -I kernels/hip "$@"; }
hc -S --cuda-device-only -Rpass-analysis=kernel-resource-usage scripts/mxr/$H.hip -o .mxrbin/$H.s 2> .mxrbin/$H-res.log || { grep -B2 -A6 error .mxrbin/$H-res.log | head -60; exit 1; }
hc -fPIC scripts/mxr/$H.hip -o .mxrbin/$H 2>&1 | grep -E "error" -A3 || true
ls -la .mxrbin/$H .mxrbin/$H.s
sha256sum .mxrbin/$H
python3 scripts/mxr/isa.py .mxrbin/$H.s .mxrbin/$H-res.log | tee .mxrbin/$H-isa.txt
