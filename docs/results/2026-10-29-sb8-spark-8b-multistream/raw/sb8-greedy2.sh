#!/usr/bin/env bash
set -uo pipefail
: "${FLEETQ_JOB:?}"
H=spark2; HN=spark-fa00; PORT=8193
D=$(dirname $0)
M=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
BIN=/home/nvidia/intarsiaml-sb8/bin/intarsia
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
arm() { # name  "ENV=.."  "serve args"  ns reps
  local name=$1 envs=$2 args=$3 ns=$4 reps=$5
  echo "== $name env[$envs] args[$args]"
  SPID=$(rsh "nohup env $envs $BIN serve --backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 36 --profile-every 0 $args > ~/intarsiaml-sb8/out/greedy-$name.log 2>&1 < /dev/null & echo \$!")
  local up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  [[ $up == 1 ]] || { echo "server did not come up"; stop_server; return; }
  python3 $D/sb8-greedy.py http://$HN:$PORT/v1 $ns $reps
  stop_server
}
R16="--recipe /home/nvidia/intarsiaml-sb8/qwen3-mb16.toml --max-batch 16"
KOFF="INTARSIA_CUDA_FP8_ROWS_Q=0 INTARSIA_CUDA_BF16_ROWS_SPLIT=0 INTARSIA_CUDA_NVFP4_ROWS_SPLIT=0 INTARSIA_CUDA_NVFP4_ROWS_FULL=0"
arm mb16-nobg "INTARSIA_BATCH_GRAPHS=0" "$R16" 5,8,16 2
arm mb16-nographs "X=1" "$R16 --no-graphs" 5,8,16 2
echo "== done"
