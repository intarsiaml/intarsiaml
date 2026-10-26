#!/usr/bin/env bash
# Lane sk (from scripts/mp/le-window.sh): same-window localeval prefill A/B. Mac side, run INSIDE a fleetq slot on the Spark (FLEETQ_JOB set), vLLM up and idle:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "..." -- scripts/sk/le-window.sh <outname> <rounds> <arm> ...
# arm = name:tree[:chunk[:ENV=V,ENV=V]]   (the special arm "vllm" benchmarks the box's vLLM `guidance` role)
# Per round every arm runs once (round 2 reversed): a fresh `intarsia serve` per intarsia arm, then localeval speed
# (sizes 1024,4096,16384, gen 128, 3 reps + 1 warmup, thinking off). Results: /opt/code/deepeval/results/<run_id>, copied names in $OUT/index.txt.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=${HOST:-spark2}; HN=${HN:-spark-fa00}; PORT=${PORT:-8190}
OUTN=$1; ROUNDS=$2; shift 2
OUT=/opt/code/intarsiaml/.worktrees/sk/docs/results/2026-10-25-sk-spark-prefill/raw/$OUTN; mkdir -p $OUT
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
SPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2" ; SPID=""; }; }
trap 'stop_server' EXIT INT TERM
le() { # le <label> <url> <model> <recipe>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "$2" --model "$3" --thinking off --sizes 1024,4096,16384 --gen 128 --reps 3 \
      --label "$1" --no-unim --recipe "$4" ) > $OUT/$1.out 2>&1
  echo "$1: $(grep -aE 'run_id|Run id|results/' $OUT/$1.out | head -2 | tr '\n' ' ')" | tee -a $OUT/index.txt
  grep -aE '^\|' $OUT/$1.out | head -12
}
run_arm() { # run_arm <round> <spec>
  local r=$1 spec=$2 name tree chunk envs
  IFS=: read -r name tree chunk envs <<< "$spec"; envs=${envs//,/ }
  echo "== round $r arm $name $(date +%T)"
  if [[ $name == vllm ]]; then
    le "sk-vllm-r$r" "http://$HN:8000/v1" guidance "vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --enable-prefix-caching (box launcher missioncntrl-guidance-run.sh)"
    return
  fi
  local sargs=""; [[ -n ${chunk:-} ]] && sargs="--prefill-chunk $chunk"
  SPID=$(rsh "mkdir -p ~/intarsiaml-sk-$tree/out; nohup env $envs ~/intarsiaml-sk-$tree/target/acc/release/intarsia serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia-35b --max-seq 20480 --mem-budget 45 --profile-every 0 $sargs > ~/intarsiaml-sk-$tree/out/sk-serve-$name-r$r.log 2>&1 < /dev/null & echo \$!")
  local up=0 i
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then echo "server $name did not come up"; rsh "tail -5 ~/intarsiaml-sk-$tree/out/sk-serve-$name-r$r.log"; stop_server; return; fi
  rsh "grep -aE 'prefill chunk|fp8|plan' ~/intarsiaml-sk-$tree/out/sk-serve-$name-r$r.log | head -4" | cut -c1-200
  le "sk-$name-r$r" "http://$HN:$PORT/v1" intarsia-35b "intarsia serve --backend cuda --max-seq 20480 --mem-budget 45 tree=$tree env='$envs' $sargs"
  stop_server
}
ARMS=("$@")
for r in $(seq 1 $ROUNDS); do
  if (( r % 2 == 0 )); then for ((k=${#ARMS[@]}-1; k>=0; k--)); do run_arm $r "${ARMS[k]}"; done
  else for a in "${ARMS[@]}"; do run_arm $r "$a"; done; fi
done
echo "== done $(date +%T)"
