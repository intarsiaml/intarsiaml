#!/usr/bin/env bash
# Slot 1 (vLLM UP, small buffers only): build base (origin/main export) and dev on spark1, diff the SASS of the kernels
# this branch must not change, run the single-tile OMMA layout check, grep the vLLM MoE backend from its log.
#   fleetq run --host spark1 --mem 8 --label "nvfp4-mma dev1" -- scripts/cuda-moe-nvfp4/dev1.sh <base-export-dir>
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
BASE_SRC=${1:?base export dir}
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SRC_DIR="$BASE_SRC" INTARSIA_STAMP=nvfp4-base "$REPO/scripts/cuda-decode/build.sh" nv-base
"$REPO/scripts/cuda-decode/build.sh" nv-dev --tests
echo "== single-tile OMMA layout check"
rsh "cd $R/nv-dev && nvcc -O2 --std=c++17 -gencode arch=compute_121a,code=sm_121a kernels/cuda_checks/omma_tile.cu -o /tmp/omma_tile_$FLEETQ_JOB && /tmp/omma_tile_$FLEETQ_JOB; rm -f /tmp/omma_tile_$FLEETQ_JOB" || true
echo "== SASS hashes per object (cargo build output)"
for t in nv-base nv-dev; do
  rsh "cd $R/$t && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && for o in \$d/*.o; do printf '%s ' \$(basename \$o); cuobjdump -sass \$o 2>/dev/null | grep -v '^\s*//' | sed -E 's/_GLOBAL__N__[0-9a-f]+_[A-Za-z0-9_]*cu_[0-9a-f]+/NS/g' | md5sum | cut -c1-12; done" > "/tmp/sass_${t}_$FLEETQ_JOB.txt"
done
diff "/tmp/sass_nv-base_$FLEETQ_JOB.txt" "/tmp/sass_nv-dev_$FLEETQ_JOB.txt" && echo "SASS identical for every object" || echo "SASS differs (lines above; moe_group.cuh refactor / new files expected)"
echo "== vLLM MoE backend (read-only log grep)"
rsh "docker logs missioncntrl-guidance 2>&1 | grep -iE 'moe|nvfp4|fp4' | grep -iE 'backend|kernel|marlin|cutlass|flashinfer|using' | head -12" || true
echo "== quantmoe + admission tests (dev)"
"$REPO/scripts/cuda-decode/gpu-test.sh" nv-dev --test quantmoe --test admission -- --test-threads=1 2>&1 | tail -25
