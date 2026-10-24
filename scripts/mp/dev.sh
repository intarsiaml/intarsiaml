#!/usr/bin/env bash
# Lane mp: sync kernels/ to a Spark, build the standalone MoE bench per variant, run it. Run inside a fleetq slot:
#   fleetq run --host spark2 --quiet gpu,cpu --expect 10 --label "mp bench" -- scripts/mp/dev.sh spark2 "4096,2048" 10 0 [name:-DFLAG=1 ...]
# Usage: dev.sh <host> <T list> <reps> <skew> [variant ...]    variant = name[:nvcc flags]; no variant = "base" (no flags)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=$1; TS=$2; REPS=$3; SKEW=$4; shift 4
[ $# -eq 0 ] && set -- base
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ssh -n -o BatchMode=yes "$HOST" "mkdir -p ~/mp/src ~/mp/bin"
rsync -a --delete "$SRC/kernels/" "$HOST:mp/src/kernels/"
for v in "$@"; do
  name=${v%%:*}; flags=""; [[ "$v" == *:* ]] && flags=${v#*:}
  echo "== $name flags='$flags' $(date +%T)"
  ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; cd ~/mp/src && \
    extra=''; [ -f kernels/cuda/moe_w4a16_v3.cu ] && extra='kernels/cuda/moe_w4a16_v3.cu -DHAVE_V3'; \
    nvcc -O3 -std=c++17 -arch=sm_121 -lineinfo $flags -I kernels/cuda -I kernels kernels/cuda_checks/moe_w4a16_bench.cu kernels/cuda/moe_w4a16.cu \$extra -o ~/mp/bin/bench_$name 2>&1 | grep -E 'error|warning: .*(spill|stack)' | head -20; \
    ONLY=${ONLY:-} ~/mp/bin/bench_$name $TS $REPS $SKEW"
done
echo "== done $(date +%T)"
