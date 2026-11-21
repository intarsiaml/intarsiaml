#!/usr/bin/env bash
# think2 window: one localeval thinking-on quality run against arm A (production vLLM on spark2, never touched, requests only)
# or arm B (intarsiaml shipping config, a fresh server in this window), under a fleetq slot.
#   FLEETQ_OWNER=claude-overnight-think2 fleetq run --host spark2 --quiet gpu,cpu --mem 2|50 --expect N --label ... -- bash scripts/think2/window.sh <A|B|smoke> <gsm8k|bbh> <label> [limit]
# smoke: B server up, a tiny run on B and on A (--no-unim), to check reasoning_content, finish_reason and scoring.
# Sampling is whatever localeval quality sends: temperature 0 (greedy), no top_p / seed, chat_template_kwargs.enable_thinking true.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
MODE=$1; SET=$2; LABEL=$3; LIMIT=${4:-}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-21-think2/raw; mkdir -p "$OUT/recipes"
HOST=spark2; HN=spark-fa00; PORT=8191; VPORT=8000; MODEL=guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-think2/bin/intarsia'; W='~/intarsiaml-think2/out'; MS=131072; MB=40
MAXTOK=${MAXTOK:-16384}
BBH_TASKS=dyck_languages,multistep_arithmetic_two,logical_deduction_seven_objects,geometric_shapes,tracking_shuffled_objects_seven_objects
UA=http://$HN:$VPORT/v1; UB=http://$HN:$PORT/v1
SERVE_ARGS="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $MODEL --max-seq $MS --mem-budget $MB --profile-every 0 --max-batch 2 --prefix-store on"
SPID=""; WPID=""; CPID=""; T0=$(date +%s); WLOG=$OUT/window-$LABEL.log
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
stop_all() {
  [[ -n "$CPID" ]] && { rsh "kill $CPID 2>/dev/null"; CPID=""; }
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'intarsiaml-think2/bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsiaml-think2/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap 'stop_all; log "window end"' EXIT; trap 'exit 130' INT TERM
recipe() { printf '%s\n' "$2" > "$OUT/recipes/$1.txt"; echo "$OUT/recipes/$1.txt"; }
RA=$(recipe A "production vLLM (missioncntrl-guidance, image missioncntrl/vllm-openai:65b7662d): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (env VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel; vllm 0.26.1rc1.dev602; no speculative config; spark2 retro role)")
RB=$(recipe B "intarsia serve $SERVE_ARGS")
keep() { local rid; rid=$(grep -aoE 'results/[0-9]{8}-[^ ]*' "$1" | head -1 | sed 's#results/##; s#/.*##'); echo "$2 run_id=$rid" | tee -a "$OUT/index.txt"
  [[ -n $rid && -d /opt/code/deepeval/results/$rid ]] && { mkdir -p "$OUT/$rid"; cp /opt/code/deepeval/results/$rid/summary.md /opt/code/deepeval/results/$rid/result.json "$OUT/$rid/" 2>/dev/null; }; }
le() { # le <label> <url> <recipe> <out> [more localeval quality args]
  local label=$1 url=$2 rec=$3 out=$4; shift 4
  ( cd /opt/code/deepeval && uv run localeval quality --url "$url" --model $MODEL --thinking on --structured prompt --wait-idle 600 --recipe "$rec" --max-tokens $MAXTOK "$@" --label "think2-$label" ) > "$out" 2>&1
  tail -6 "$out"; keep "$out" "$label"; }
setargs() { # setargs <set> <limit>: localeval args for the set
  case $1 in
    gsm8k) echo "--bench gsm8k --limit ${2:-200} --pin-gsm8k";;
    bbh) echo "--bench bbh --tasks $BBH_TASKS --limit ${2:-30}";;
  esac; }
start_b() {
  rsh "mkdir -p $W && rm -f $W/mem-$LABEL.txt $W/clk-$LABEL.tsv"
  scp -q -o BatchMode=yes "$REPO/scripts/think2/memwatch.sh" "$REPO/scripts/think2/clk.sh" "$HOST:intarsiaml-think2/out/" || { log "script copy failed"; exit 1; }
  if rsh "ss -ltn | grep -q ':$PORT '"; then log "port $PORT busy on $HOST, aborting"; exit 1; fi
  log "vLLM /v1/models: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 $UA/models); earlyoom $(rsh 'systemctl is-active earlyoom'); free: $(rsh 'free -g | sed -n 2p')"
  log "binary: $(rsh "$BIN --version | head -1")"
  CPID=$(rsh "nohup bash $W/clk.sh $W/clk-$LABEL.tsv > /dev/null 2>&1 < /dev/null & echo \$!")
  log "serve B: $BIN serve $SERVE_ARGS"
  SPID=$(rsh "nohup $BIN serve $SERVE_ARGS > $W/serve-$LABEL.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $W/mem-$LABEL.txt > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 200); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve B did not come up"; rsh "tail -20 $W/serve-$LABEL.log" | tee -a "$WLOG"; exit 1; }
  log "serve B up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |spec mtp|batched MTP|prefix store: |MTP|context|KV cache' $W/serve-$LABEL.log | tr '\n' '|'")"
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$LABEL.metrics.before" 2>/dev/null
}
finish_b() {
  sleep 3; curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$LABEL.metrics" 2>/dev/null
  rsh "cat $W/serve-$LABEL.log" > "$OUT/serve-$LABEL.log"; rsh "cat $W/clk-$LABEL.tsv" > "$OUT/clk-$LABEL.tsv"; rsh "cat $W/mem-$LABEL.txt" > "$OUT/mem-$LABEL.txt" 2>/dev/null
  log "min MemAvailable MB (time MB): $(sort -k2 -n "$OUT/mem-$LABEL.txt" | head -1); KILLED lines: $(grep -c KILLED "$OUT/mem-$LABEL.txt"); warn/error lines in serve log: $(grep -ciE 'warn|error|panic|fail' "$OUT/serve-$LABEL.log")"
  log "clock (SM MHz at util>90): $(awk '$4>90 {n++; s+=$2; if (min==""||$2<min) min=$2} END {printf "n=%d min=%d mean=%d", n, min, (n?s/n:0)}' "$OUT/clk-$LABEL.tsv")"
}
vcount() { curl -s --max-time 5 "http://$HN:$VPORT/metrics" | awk '/^vllm:request_success_total/ {s+=$2} /^vllm:generation_tokens_total/ {g=$2} END {printf "%d %d", s, g}'; }

log "window $LABEL mode $MODE set $SET limit ${LIMIT:-default} max_tokens $MAXTOK; fleetq job $FLEETQ_JOB; vLLM counters (success gen_tokens) $(vcount)"
case $MODE in
  A)
    rsh "ss -ltn | grep -q ':$PORT '" && { log "port $PORT busy on $HOST (a B server is up), aborting"; exit 1; }
    le "A-$LABEL" "$UA" "$RA" "$OUT/$LABEL.out" $(setargs $SET $LIMIT);;
  B)
    start_b
    le "B-$LABEL" "$UB" "$RB" "$OUT/$LABEL.out" $(setargs $SET $LIMIT)
    finish_b;;
  smoke)
    start_b
    for s in gsm8k bbh; do n=3; [[ $s == bbh ]] && n=1
      le "B-smoke-$s" "$UB" "$RB" "$OUT/smoke-B-$s.out" $(setargs $s $n) --no-unim
      le "A-smoke-$s" "$UA" "$RA" "$OUT/smoke-A-$s.out" $(setargs $s $n) --no-unim
    done
    finish_b;;
esac
log "vLLM counters after: $(vcount)"
