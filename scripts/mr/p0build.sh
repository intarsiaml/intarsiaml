#!/usr/bin/env bash
# Lane mr, ON tatooine inside a fleetq job (--load cpu): build scripts/mr/mxp0.hip (harness + resource usage) and its ISA, print the instruction mix.
# Needs ~/mr-hip synced (scripts/mr/sync.sh dev). Output: .mrbin/mxp0, .mrbin/p0.s, .mrbin/p0-res.log, .mrbin/p0-isa.txt
set -eo pipefail
cd ~/mr-hip
mkdir -p .mrbin
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
hc() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/mr-hip:/work -w /work $IMG hipcc -O3 -std=c++17 -Wno-unused-function --offload-arch=gfx1201 -I kernels/hip "$@"; }
hc -S --cuda-device-only -Rpass-analysis=kernel-resource-usage scripts/mr/mxp0.hip -o .mrbin/p0.s 2> .mrbin/p0-res.log
hc -fPIC scripts/mr/mxp0.hip -o .mrbin/mxp0 2>&1 | grep -v "warning\|note:\|^ *[0-9]* |\|\^" || true
ls -la .mrbin/mxp0 .mrbin/p0.s
python3 scripts/mr/p0isa.py .mrbin/p0.s .mrbin/p0-res.log | tee .mrbin/p0-isa.txt
