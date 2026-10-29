#!/usr/bin/env bash
# Lane step16 (from msd) (from scripts/spf/le-window.sh): same-window localeval concurrency A/B on spark1 (GB10). Mac side, run INSIDE a fleetq slot on spark1
# (FLEETQ_JOB set), spark1-vllm (role retro) up and idle:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "..." -- scripts/step16/le-window.sh <outname> <rounds> <arm> ...
# arm = name:bin[:ENV=V,ENV=V[:extra_serve_args, spaces as underscores]]    bin = ~/intarsiaml-step16/bin/<bin>; the special arm "vllm" is the box's vLLM
# Per round every arm runs once (even rounds reversed): a fresh `intarsia serve` per intarsia arm, then localeval speed
# (size 1024, gen $GEN, concurrency $CONC, 3 reps + 1 warmup for the single-stream rows, thinking off). Raw: $OUT/<label>.out, run ids in $OUT/index.txt.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=${HOST:-spark1}; HN=${HN:-spark-4e77}; PORT=${PORT:-8190}
OUTN=$1; ROUNDS=$2; shift 2
OUT=/opt/code/intarsiaml/.worktrees/step16/docs/results/2026-10-29-step16-decode-step/raw/$OUTN; mkdir -p $OUT
M35=${MODEL:-/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5}
SN=${SN:-intarsia}; SIZES=${SIZES:-1024}; GEN=${GEN:-256}; CONC=${CONC:-1,4,8,16}; MS=${MS:-8192}; MB=${MB:-40}
SPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2" ; SPID=""; }; }
trap 'stop_server' EXIT INT TERM
le() { # le <label> <url> <model> <recipe>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "$2" --model "$3" --thinking off --sizes $SIZES --gen $GEN --reps 3 --concurrency $CONC \
      --label "$1" --no-unim --recipe "$4" ) > $OUT/$1.out 2>&1
  echo "$1: $(grep -aE 'run_id|Run id|results/' $OUT/$1.out | head -2 | tr '\n' ' ')" | tee -a $OUT/index.txt
  grep -aE 'streams: aggregate' $OUT/$1.out
}
run_arm() { # run_arm <round> <spec>
  local r=$1 spec=$2 name bin envs xargs
  IFS=: read -r name bin envs xargs <<< "$spec"; envs=${envs//,/ }; xargs=${xargs//_/ }
  echo "== round $r arm $name $(date +%T)"
  if [[ $name == vllm ]]; then
    le "step16-vllm-r$r" "http://$HN:8000/v1" guidance "spark1 production vLLM (role retro, served as guidance/capcom): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-chunked-prefill"
    return
  fi
  SPID=$(rsh "mkdir -p ~/intarsiaml-step16/out; nohup env $envs ~/intarsiaml-step16/bin/$bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $SN --max-seq $MS --mem-budget $MB --profile-every 0 $xargs > ~/intarsiaml-step16/out/serve-$name-r$r.log 2>&1 < /dev/null & echo \$!")
  local up=0 i
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then echo "server $name did not come up"; rsh "tail -5 ~/intarsiaml-step16/out/serve-$name-r$r.log"; stop_server; return; fi
  echo "$name r$r: MemAvailable $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo") MB with the server up" | tee -a $OUT/mem.txt
  le "step16-$name-r$r" "http://$HN:$PORT/v1" $SN "intarsia serve --backend cuda --max-seq $MS --mem-budget $MB bin=$bin env='$envs' $xargs"
  stop_server
}
ARMS=("$@")
for r in $(seq 1 $ROUNDS); do
  if (( r % 2 == 0 )); then for ((k=${#ARMS[@]}-1; k>=0; k--)); do run_arm $r "${ARMS[k]}"; done
  else for a in "${ARMS[@]}"; do run_arm $r "$a"; done; fi
done
echo "== done $(date +%T)"
