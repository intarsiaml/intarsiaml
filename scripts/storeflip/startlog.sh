#!/usr/bin/env bash
# storeflip (narrowed PR): startup-log check on spark2, one short visit per arm, vLLM `retro` stays up. No requests are sent.
#   FLEETQ_OWNER=claude-storeflip fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 12 --label "storeflip startup-log check" -- bash scripts/storeflip/startlog.sh
# arms: unset (nothing set) | mac1 (INTARSIA_PREFIX_STORE_MACRO=1) | zero (PREFAULT=0)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-storeflip/raw; LOGF=$OUT/startlog.txt; : > "$LOGF"
HOST=spark2; HN=spark-fa00; PORT=8191
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-storeflip/bin/intarsia'; W='~/intarsiaml-storeflip/out'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
for arm in unset mac1 zero; do
  case $arm in unset) ENVS="";; mac1) ENVS="INTARSIA_PREFIX_STORE_MACRO=1";; zero) ENVS="INTARSIA_PREFIX_STORE_PREFAULT=0";; esac
  rsh "ss -ltn | grep -q ':$PORT '" && { echo "port $PORT busy" | tee -a "$LOGF"; exit 1; }
  SERVE="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on"
  echo "== arm $arm (env: ${ENVS:-none}): intarsia serve $SERVE" | tee -a "$LOGF"
  SPID=$(rsh "env $ENVS nohup $BIN serve $SERVE > $W/startlog-$arm.log 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 120); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  echo "up=$up; MemAvailable MB $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo")" | tee -a "$LOGF"
  rsh "grep -E 'macro|workspace|prefix store' $W/startlog-$arm.log | cut -c1-230" | tee -a "$LOGF"
  rsh "kill $SPID; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; true"
  rsh "ss -ltn | grep -q ':$PORT '" && echo "WARNING: port $PORT still bound" | tee -a "$LOGF"
done
echo "servers left running: $(rsh "pgrep -fc '[i]ntarsiaml-storeflip/bin/intarsia serve'")" | tee -a "$LOGF"
