#!/usr/bin/env bash
# style7 window on spark2: first-token top-20 logprobs + GSM8K (localeval, pinned, thinking off, n=200) per config.
#   FLEETQ_OWNER=claude-style7 fleetq run --host spark2 --mem 50 --expect 60 --label "style7 window" -- bash scripts/style7/window.sh <wname> <cfg>...
# cfgs: vllm (production, request traffic only, one at a time) | store | nostore | nostore-c128 | nostore-fp8kv | store-fp8kv | nostore-c128-fp8kv | nostore-mb1 (max-batch 1)
# One server of ours at a time, beside vLLM; nothing of ours sends to both engines at once.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
WN=$1; shift; CFGS="$*"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-style7/raw; mkdir -p "$OUT/recipes"; SC=$REPO/scripts/style7
HOST=spark2; HN=spark-fa00; PORT=8191; VPORT=8000; MODEL=guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-style7/bin/intarsia'; W='~/intarsiaml-style7/out'; MS=131072; MB=40
TOK=$OUT/vllm-tokenize-pinned200-think-off.json
BASE="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $MODEL --max-seq $MS --mem-budget $MB --profile-every 0 --no-supervisor"
SPID=""; WPID=""; T0=$(date +%s); WLOG=$OUT/window-$WN.log
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
stop_b() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "kill -TERM $SPID 2>/dev/null; for i in \$(seq 1 40); do pgrep -f 'intarsiaml-style7/bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsiaml-style7/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap 'stop_b; log "window end"' EXIT; trap 'exit 130' INT TERM
start_b() { # start_b <cfg> <extra serve args>
  local cfg=$1; shift
  rsh "mkdir -p $W && rm -f $W/mem-$cfg.txt"
  scp -q -o BatchMode=yes "$SC/memwatch.sh" "$HOST:intarsiaml-style7/out/" || { log "script copy failed"; exit 1; }
  rsh "ss -ltn | grep -q ':$PORT '" && { log "port $PORT busy on $HOST, aborting"; exit 1; }
  log "serve $cfg: $BIN serve $BASE $*; free: $(rsh 'free -g | sed -n 2p'); earlyoom $(rsh 'systemctl is-active earlyoom')"
  SPID=$(rsh "nohup $BIN serve $BASE $* > $W/serve-$cfg.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $W/mem-$cfg.txt > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 200); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $cfg did not come up"; rsh "tail -20 $W/serve-$cfg.log" | tee -a "$WLOG"; exit 1; }
  log "serve $cfg up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |prefix store|kv|chunk' $W/serve-$cfg.log | head -8 | tr '\n' '|'")"
}
finish_b() { # finish_b <cfg>
  rsh "cat $W/serve-$1.log" > "$OUT/serve-$1.log"; rsh "cat $W/mem-$1.txt" > "$OUT/mem-$1.txt" 2>/dev/null
  log "min MemAvailable MB (time MB): $(sort -k2 -n "$OUT/mem-$1.txt" | head -1); KILLED: $(grep -c KILLED "$OUT/mem-$1.txt")"
  stop_b
}
ft() { # ft <label> <url> [--vllm]
  python3 "$SC/firsttok.py" --url "$2" --model $MODEL --tokens "$TOK" --out "$OUT/ft-$1.json" ${3:-} 2>&1 | tail -2 | tee -a "$WLOG"; }
gsm() { # gsm <label> <url> <recipe-file>
  ( cd /opt/code/deepeval && uv run localeval quality --url "$2" --model $MODEL --thinking off --structured prompt --wait-idle 600 --recipe "$3" --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k --label "style7-$1" ) > "$OUT/$1-gsm8k.out" 2>&1
  tail -3 "$OUT/$1-gsm8k.out" | tee -a "$WLOG"
  local rid; rid=$(grep -aoE 'results/[0-9]{8}-[^ ]*' "$OUT/$1-gsm8k.out" | head -1 | sed 's#results/##; s#/.*##'); echo "$1 run_id=$rid" | tee -a "$OUT/index.txt"
  [[ -n $rid && -d /opt/code/deepeval/results/$rid ]] && { mkdir -p "$OUT/$rid"; cp /opt/code/deepeval/results/$rid/summary.md /opt/code/deepeval/results/$rid/result.json "$OUT/$rid/" 2>/dev/null; }; }
recipe() { printf '%s\n' "$2" > "$OUT/recipes/$1.txt"; echo "$OUT/recipes/$1.txt"; }

log "window $WN cfgs: $CFGS; fleetq job $FLEETQ_JOB; vLLM models: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://$HN:$VPORT/v1/models)"
for cfg in $CFGS; do
  case $cfg in
    vllm)
      R=$(recipe vllm "production vLLM (retro, spark2), request traffic only, one request at a time")
      for p in 1 2 3; do ft "vllm-p$p" http://$HN:$VPORT/v1 --vllm; done
      gsm vllm http://$HN:$VPORT/v1 "$R" ;;
    store)         X="--max-batch 2 --prefix-store on" ;;
    nostore)       X="--max-batch 2" ;;
    nostore-c128)  X="--max-batch 2 --prefill-chunk 128" ;;
    nostore-fp8kv) X="--max-batch 2 --kv-dtype fp8_e4m3" ;;
    store-fp8kv)   X="--max-batch 2 --prefix-store on --kv-dtype fp8_e4m3" ;;
    nostore-c128-fp8kv) X="--max-batch 2 --prefill-chunk 128 --kv-dtype fp8_e4m3" ;;
    nostore-mb1)   X="--max-batch 1" ;;
    *) log "unknown cfg $cfg"; continue ;;
  esac
  [[ $cfg == vllm ]] && continue
  R=$(recipe "$cfg" "intarsia serve $BASE $X")
  start_b "$cfg" $X
  for p in 1 2; do ft "$cfg-p$p" http://$HN:$PORT/v1; done
  gsm "$cfg" http://$HN:$PORT/v1 "$R"
  finish_b "$cfg"
done
