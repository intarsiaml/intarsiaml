#!/usr/bin/env bash
# sb8: Qwen3-8B BF16 multi-stream on spark2 (GB10), localeval speed, 1k prompt, 256 gen, streams 1/4/8/16.
set -uo pipefail
: "${FLEETQ_JOB:?}"
H=spark2; HN=spark-fa00; PORT=8191
OUT=/opt/code/intarsiaml/.worktrees/sb8/docs/results/2026-10-29-sb8-spark-8b-multistream/raw; mkdir -p $OUT
M=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
BIN=/home/nvidia/intarsiaml-sb8/bin/intarsia
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
run_arm() { # name round extra-args
  local name=$1 r=$2; shift 2; local extra="$*"
  echo "== $name r$r $(date +%T)"
  SPID=$(rsh "nohup $BIN serve --backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 36 --profile-every 0 $extra > ~/intarsiaml-sb8/out/serve-$name-r$r.log 2>&1 < /dev/null & echo \$!")
  local up=0
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then echo "server did not come up"; rsh "tail -8 ~/intarsiaml-sb8/out/serve-$name-r$r.log"; stop_server; return; fi
  rsh "nvidia-smi --query-gpu=memory.used --format=csv,noheader; awk '/MemAvailable/ {print int(\$2/1024)\" MB avail\"}' /proc/meminfo" | tee -a $OUT/mem.txt
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "http://$HN:$PORT/v1" --model intarsia --thinking off --sizes 1024 --gen 256 --reps 3 --concurrency 1,4,8,16 --label "sb8-$name-r$r" --no-unim --recipe "intarsia serve --backend cuda --max-seq 8192 --mem-budget 36 $extra (Qwen3-8B BF16, spark2)" ) > $OUT/$name-r$r.out 2>&1
  grep -aE 'run_id|Run id|results/' $OUT/$name-r$r.out | head -2 | tee -a $OUT/index.txt
  grep -aE 'streams: aggregate' $OUT/$name-r$r.out
  rsh "grep -iE 'max.batch|batch' ~/intarsiaml-sb8/out/serve-$name-r$r.log | head -3" 
  stop_server
}
ssh -n -o BatchMode=yes $H "sed 's/^max_batch = 8/max_batch = 16/' ~/intarsiaml-sb8/src/recipes/qwen3.toml > ~/intarsiaml-sb8/qwen3-mb16.toml; grep -n '^max_batch' ~/intarsiaml-sb8/qwen3-mb16.toml"
R16="--recipe /home/nvidia/intarsiaml-sb8/qwen3-mb16.toml --max-batch 16"
run_arm mb16 1 $R16
run_arm mb8 1 --max-batch 8
run_arm default 1
run_arm mb16 2 $R16
run_arm mb16 3 $R16
echo "== done $(date +%T)"
