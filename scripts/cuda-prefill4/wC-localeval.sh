#!/usr/bin/env bash
# Window C driver (runs here, on endor): localeval speed of intarsia arms on spark1 while vLLM is down, all inside ONE
# scripts/cuda-decode/window.sh lease (localeval runs with FLEETQ_JOB set to it, so it does not queue behind it).
#   ARMS="base dev" scripts/cuda-prefill4/wC-localeval.sh <max-min> <out-log>   (arm names; their envs are in wBC.sh)
# For each arm the spark1 side (wC-serve.sh) starts the server and prints READY <arm>; this script then runs
#   uv run localeval speed --url http://spark-4e77:8090/v1 --model intarsia-35b --thinking off --sizes $SIZES --gen 128 --reps 3
# and creates the arm's done marker on spark1. Env: SIZES (1024,4096,16384), DOWNTIME_LOG (passed to window.sh), REMOTE
# (the spark1 window script, default wBC.sh: the wB-ab.sh speed A/B, then wC-serve.sh; it must print READY <arm> for each
# arm named in ARMS), LABEL (the fleetq label).
set -uo pipefail
MAXMIN=${1:?max minutes}; LOG=${2:?output log}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ARMS:?arm list}"
SIZES=${SIZES:-1024,4096,16384}
: > "$LOG"
"$REPO/scripts/cuda-decode/window.sh" "$MAXMIN" "${LABEL:-prefill-4 W-C localeval}" \
  "${REMOTE:-/home/nvidia/intarsia-decfuse/p4-dev/scripts/cuda-prefill4/wBC.sh}" >> "$LOG" 2>&1 &
wpid=$!
job=""; done_arms=" "
while kill -0 $wpid 2>/dev/null; do
  [[ -z "$job" ]] && job=$(grep -Eo 'lease [0-9]+ held' "$LOG" | grep -Eo '[0-9]+' | head -1)
  for arm in $(grep -Eo '^READY [^ ]+' "$LOG" | awk '{print $2}'); do
    [[ "$done_arms" == *" $arm "* ]] && continue
    echo "$(date +%T) localeval arm $arm (job $job)" >> "$LOG"
    (cd /opt/code/deepeval && FLEETQ_JOB=$job uv run localeval speed --url http://spark-4e77:8090/v1 --model intarsia-35b \
      --thinking off --sizes "$SIZES" --gen 128 --reps 3 --label "intarsia-prefill4-$arm") >> "$LOG" 2>&1
    echo "$(date +%T) localeval arm $arm exit $?" >> "$LOG"
    FLEETQ_JOB=$job ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$job; touch ~/intarsia-decfuse/out/p4C/$arm.done"
    done_arms="$done_arms$arm "
  done
  sleep 3
done
wait $wpid; rc=$?
echo "$(date +%T) window exit $rc" >> "$LOG"
exit $rc
