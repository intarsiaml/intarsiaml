#!/usr/bin/env bash
# Lane cgv (from scripts/s27d/le-window.sh): same-window localeval prefill+decode A/B. Mac side, run INSIDE a fleetq slot on the Spark (FLEETQ_JOB set), vLLM up and idle:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "..." -- scripts/cgv/le-window.sh <outname> <rounds> <arm> ...
# arm = name:tree[:chunk[:ENV=V,ENV=V[:extra_serve_args, spaces as underscores]]]   (the special arm "vllm" benchmarks the box's vLLM `guidance` role)
# Per round every arm runs once (round 2 reversed): a fresh `intarsia serve` per intarsia arm, then localeval speed
# (sizes 1024,4096,16384, gen 128, 3 reps + 1 warmup, thinking off). Results: /opt/code/deepeval/results/<run_id>, copied names in $OUT/index.txt.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=${HOST:-spark1}; HN=${HN:-spark-4e77}; PORT=${PORT:-8190}
OUTN=$1; ROUNDS=$2; shift 2
OUT=/opt/code/intarsiaml/.worktrees/cgv/docs/results/2026-10-27-cgv-cuda-mxfp4-gemv/raw/$OUTN; mkdir -p $OUT
M35=${MODEL:-/home/nvidia/intarsiaml-cgv/models/Qwen3.8-27B-MXFP4}
SN=${SN:-intarsia-27b}; SIZES=${SIZES:-1024,4096,16384}; MS=${MS:-20480}; MB=${MB:-45}
SPID=""
vtok() { curl -s --max-time 5 "http://$HN:8000/metrics" | grep -E "^vllm:generation_tokens_total" | head -1 | awk '{print $2}'; }
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2" ; SPID=""; }; }
trap 'stop_server' EXIT INT TERM
le() { # le <label> <url> <model> <recipe>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "$2" --model "$3" --thinking off --sizes $SIZES --gen ${GEN:-128} --reps ${REPS:-3} \
      --label "$1" --no-unim --recipe "$4" ) > $OUT/$1.out 2>&1
  echo "$1: $(grep -aE 'run_id|Run id|results/' $OUT/$1.out | head -2 | tr '\n' ' ')" | tee -a $OUT/index.txt
  grep -aE '^\|' $OUT/$1.out | head -12
}
run_arm() { # run_arm <round> <spec>
  local r=$1 spec=$2 name tree chunk envs xargs
  IFS=: read -r name tree chunk envs xargs <<< "$spec"; envs=${envs//,/ }; xargs=${xargs//_/ }
  echo "== round $r arm $name $(date +%T)"
  if [[ $name == vllm ]]; then
    le "cgv-vllm-r$r" "http://$HN:8000/v1" guidance "spark1 production vLLM (served as flight/guidance/capcom): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt_fp4 --max-model-len 32768 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.50 --enable-prefix-caching --enable-chunked-prefill"
    return
  fi
  local sargs=""; [[ -n ${chunk:-} ]] && sargs="--prefill-chunk $chunk"
  sargs="$sargs $xargs"
  SPID=$(rsh "mkdir -p ~/intarsiaml-cgv/trees/$tree/out; nohup env $envs ~/intarsiaml-cgv/trees/$tree/target/release/intarsia serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $SN --max-seq $MS --mem-budget $MB --profile-every 0 $sargs > ~/intarsiaml-cgv/trees/$tree/out/cgv-serve-$name-r$r.log 2>&1 < /dev/null & echo \$!")
  local up=0 i
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then echo "server $name did not come up"; rsh "tail -5 ~/intarsiaml-cgv/trees/$tree/out/cgv-serve-$name-r$r.log"; stop_server; return; fi
  echo "$name r$r: MemAvailable $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo") MB with the server up" | tee -a $OUT/mem.txt
  rsh "grep -aE 'prefill chunk|fp8|plan' ~/intarsiaml-cgv/trees/$tree/out/cgv-serve-$name-r$r.log | head -4" | cut -c1-200
  local v0 v1; v0=$(vtok)
  le "cgv-$name-r$r" "http://$HN:$PORT/v1" $SN "intarsia serve --backend cuda --max-seq $MS --mem-budget $MB tree=$tree env='$envs' $sargs"
  v1=$(vtok); echo "$name r$r vllm generation_tokens_total $v0 -> $v1 (delta $(python3 -c "print(int(float('${v1:-0}')-float('${v0:-0}')))" 2>/dev/null))" | tee -a $OUT/load.txt
  stop_server
}
ARMS=("$@")
for r in $(seq 1 $ROUNDS); do
  if (( r % 2 == 0 )); then for ((k=${#ARMS[@]}-1; k>=0; k--)); do run_arm $r "${ARMS[k]}"; done
  else for a in "${ARMS[@]}"; do run_arm $r "$a"; done; fi
done
echo "== done $(date +%T)"
