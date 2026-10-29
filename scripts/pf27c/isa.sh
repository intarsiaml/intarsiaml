#!/usr/bin/env bash
# Lane PF27, ON tatooine inside a fleetq job (--load cpu, no GPU): device assembly and per-kernel resource usage (VGPRs, SGPRs, scratch, LDS,
# occupancy) of a HIP kernel source for gfx1201 into pf27c-out/isa/<file>.{s,res.txt} of the lane's repo copy.
#   ssh tatooine 'bash ~/intarsiaml-pf27/scripts/pf27/isa.sh mxfp4_a8_gemm'
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-pf27c}"
cd "$INTARSIA_REPO"
mkdir -p pf27c-out/isa
for f in ${@:-mxfp4_a8_gemm}; do
  deploy/tatooine/dev.sh bash -c "cd /work && /opt/rocm/bin/hipcc -S -O3 -std=c++17 --offload-arch=gfx1201 --cuda-device-only \
    -Rpass-analysis=kernel-resource-usage -I kernels/hip -o pf27c-out/isa/$f.s kernels/hip/$f.hip" > "pf27c-out/isa/$f.res.txt" 2>&1
  echo "=== $f: $(grep -c 'remark' pf27c-out/isa/$f.res.txt) remarks, $(grep -ac 'error' pf27c-out/isa/$f.res.txt) error lines"
done
