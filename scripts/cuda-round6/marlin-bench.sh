#!/usr/bin/env bash
# vLLM NVFP4 Marlin MoE microbench inside the running vLLM container on spark1 (does not touch the server).
# Run on the Mac inside a fleetq slot (small dev run: --mem 4; the clean timing run takes an exclusive slot, caller's call):
#   fleetq run --host spark1 --mem 4 --label "cuda-round6: vLLM Marlin MoE microbench" -- scripts/cuda-round6/marlin-bench.sh --tokens 1024,2048,4096,8192 --reps 10
# Args are passed to marlin_moe_bench.py (--tokens, --reps, --warmup, --profile, --no-check ...).
set -eu
: "${FLEETQ_JOB:?run inside a fleetq slot}"
HERE="$(cd "$(dirname "$0")" && pwd)"
C=${CONTAINER:-missioncntrl-guidance}
# shellcheck disable=SC2029
ssh -n -o BatchMode=yes spark1 "FLEETQ_JOB=$FLEETQ_JOB; nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu --format=csv; nvidia-smi -q -d PERFORMANCE | grep -iE 'active|idle|slowdown' | head -12; nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
ssh -o BatchMode=yes spark1 "FLEETQ_JOB=$FLEETQ_JOB; docker exec -i $C sh -c 'cat > /tmp/marlin_moe_bench.py'" < "$HERE/marlin_moe_bench.py"
ARGS=$(printf '%q ' "$@")
# shellcheck disable=SC2029
ssh -n -o BatchMode=yes spark1 "FLEETQ_JOB=$FLEETQ_JOB; docker exec $C timeout 900 python3 /tmp/marlin_moe_bench.py $ARGS; rc=\$?; docker exec $C rm -f /tmp/marlin_moe_bench.py; exit \$rc"
