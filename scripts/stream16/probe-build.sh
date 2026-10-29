#!/usr/bin/env bash
# Lane stream16: copy probe.cu to spark1 and compile it (M = 16, 12, 8) against the synced tree's kernels/cuda headers (fleetq --load cpu):
#   fleetq run --host spark1 --load cpu --expect 6 --label "stream16: probe build" -- scripts/stream16/probe-build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; L='~/intarsiaml-stream16'
ssh -n -o BatchMode=yes spark1 "mkdir -p $L/probe"
scp -q "$H/probe.cu" spark1:intarsiaml-stream16/probe/probe.cu
for m in 16 12 8; do
  ssh -n -o BatchMode=yes spark1 "export PATH=/usr/local/cuda/bin:\$PATH; cd $L/probe && nvcc -O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=sm_121 -DPM=$m -I $L/src/kernels/cuda -Xptxas -v probe.cu -o probe$m 2>&1 | grep -E 'error|spill' | grep -E ' [1-9][0-9]* bytes spill|error' | head -20; ls -la probe$m" &
done
wait
