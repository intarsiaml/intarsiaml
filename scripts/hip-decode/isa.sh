#!/usr/bin/env bash
# CPU-slot ISA / resource dump of HIP kernel sources (no GPU): device assembly and per-kernel resource usage
# (VGPRs, SGPRs, scratch = spills, LDS, occupancy) for gfx1201, into bench-out/isa/<file>.{s,res.txt}.
#   FLEETQ_JOB=<id> ssh tatooine 'bash ~/intarsiaml-r6/scripts/hip-decode/isa.sh moe fp8 gdn'
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-r6}"
cd "$INTARSIA_REPO"
mkdir -p bench-out/isa
for f in ${@:-moe fp8 gdn attention_flash}; do
  deploy/tatooine/dev.sh bash -c "cd /work && /opt/rocm/bin/hipcc -S -O3 -std=c++17 --offload-arch=gfx1201 --cuda-device-only \
    -Rpass-analysis=kernel-resource-usage -I kernels/hip -o bench-out/isa/$f.s kernels/hip/$f.hip" > "bench-out/isa/$f.res.txt" 2>&1
  echo "=== $f: $(grep -c 'remark' bench-out/isa/$f.res.txt) remarks"
done
