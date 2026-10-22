#!/usr/bin/env bash
# Lane S1: one vLLM-down window on spark2 (role retro, container missioncntrl-guidance), 00:00-07:00 EDT only.
#   fleetq run --host spark2 --exclusive --takes-down retro --mem 100 --expect 30 --label "..." -- scripts/s1/window.sh <max-min> <label> <remote script>
# Wraps scripts/cuda-round7/window.sh (stop vLLM, wait for memory, run the script on the box under `timeout`, restore vLLM in
# the EXIT trap and wait for /v1/models, log the downtime) with spark2's names. Refuses to start outside 00:00-07:00 EDT.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --takes-down retro}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
h=$(TZ=America/New_York date +%H)
if (( 10#$h >= 7 )); then echo "refusing: $(TZ=America/New_York date +%T) EDT is outside the 00:00-07:00 window"; exit 2; fi
export HOST=spark2 VLLM_CONTAINER=missioncntrl-guidance VLLM_HOST=${VLLM_HOST:-spark-fa00}
export DOWNTIME_LOG=${DOWNTIME_LOG:-$REPO/docs/results/2026-10-22-s1-spark-prefill/raw/downtime.log}
mkdir -p "$(dirname "$DOWNTIME_LOG")"
exec "$REPO/scripts/cuda-round7/window.sh" "$@"
