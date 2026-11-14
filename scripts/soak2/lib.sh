#!/usr/bin/env bash
# soak2 lane helpers (sourced by soak.sh / acc.sh). spark1, production vLLM stays UP: nothing here stops or restarts any container.
# Everything runs inside a fleetq slot (FLEETQ_JOB set). Adapted from scripts/mtp-store/window.sh and scripts/soak/lib.sh.
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=${SOAK_OUT:-$REPO/docs/results/2026-11-14-soak2/raw}; mkdir -p "$OUT"
SC=$REPO/scripts/soak2; SOAKS=$REPO/scripts/soak
HOST=spark1; HN=spark-4e77; PORT=8191; VPORT=8000; MODEL=soak2
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsia-soak2/bin/intarsia'; MS=131072; MB=40; W='~/intarsia-soak2/out'
URL=http://$HN:$PORT/v1
T0=$(date +%s); SPID=""; WPID=""; SAMP=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$OUT/${WLOG:-lib}.log"; }
vcount() { curl -s --max-time 5 "http://$HN:$VPORT/metrics" | awk '/^vllm:request_success_total/ {s+=$2} /^vllm:generation_tokens_total/ {g=$2} END {printf "success=%d gen_tokens=%d", s, g}'; }
scrape() { curl -sf --max-time 10 "http://$HN:$PORT/metrics" > "$1" 2>/dev/null; }
# replay2 <arm> <order> <seed> <streams> <trace-file> [extra replay2 args]: scripts/soak2/replay2.py (per-conversation temperature)
replay2() {
  local arm=$1 order=$2 seed=$3 k=$4 trace=$5; shift 5
  echo "== $arm replay2 k=$k $order seed $seed $* $(date +%T)"
  ( cd /opt/code/deepeval && perl -e 'alarm 1500; exec @ARGV' uv run python "$SC/replay2.py" --url "$URL" --model $MODEL --arm "$arm" --order "$order" --seed "$seed" --streams "$k" \
      --plan "$OUT/replay-plan.json" --trace "$trace" --out "$OUT" "$@" ) > "$OUT/$arm-replay2-$order-seed$seed.out" 2>&1
  tail -12 "$OUT/$arm-replay2-$order-seed$seed.out"
}
stop_server() {
  [[ -n "$SAMP" ]] && { rsh "kill $SAMP 2>/dev/null"; SAMP=""; }
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  # the pattern names our own binary: other lanes' servers on this box are never touched
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'intarsia-soak2/bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsia-soak2/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap 'stop_server; log "window end"' EXIT; trap 'exit 130' INT TERM
prep_box() {
  rsh "mkdir -p $W" && scp -q -o BatchMode=yes "$SC/memwatch.sh" "$SC/sampler.sh" "$HOST:intarsia-soak2/out/" || { log "script copy failed"; exit 1; }
  if rsh "ss -ltn | grep -q ':$PORT '"; then log "port $PORT busy on $HOST, aborting"; exit 1; fi
  log "vLLM (prod, stays up) /v1/models: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://$HN:$VPORT/v1/models); vLLM counters $(vcount); earlyoom: $(rsh 'systemctl is-active earlyoom'); free: $(rsh 'free -g | sed -n 2p')"
  [[ $(rsh 'systemctl is-active earlyoom') == active ]] || log "WARNING: earlyoom NOT active: keep 15 GB free (memwatch kills at 10 GB)"
  rsh "$BIN --version | head -1" | tee -a "$OUT/${WLOG:-lib}.log"
}
start_serve() { # start_serve <name> <extra serve args...>   (env assignments: SERVE_ENV)
  local name=$1; shift
  log "serve arm $name: ${SERVE_ENV:+$SERVE_ENV }$BIN serve --backend cuda --model <HF snapshot 1355db6a> --port $PORT --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f $W/mem-$name.txt $W/sample-$name.tsv; mkdir -p $W"
  SPID=$(rsh "nohup env ${SERVE_ENV:-} $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $MODEL --max-seq $MS --mem-budget $MB --profile-every 0 $* > $W/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $W/mem-$name.txt > /dev/null 2>&1 < /dev/null & echo \$!")
  SAMP=$(rsh "nohup bash $W/sampler.sh $SPID $PORT $W/sample-$name.tsv ${SAMPLE_S:-60} > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 $W/serve-$name.log" | tee -a "$OUT/${WLOG:-lib}.log"; finish_serve "$name" 1; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |spec mtp|batched MTP|prefix store: |MTP' $W/serve-$name.log | tr '\n' '|'")"
  scrape "$OUT/serve-$name.metrics.before"
}
finish_serve() { # finish_serve <name> [failed]
  [[ -z "${2:-}" ]] && { sleep 3; scrape "$OUT/serve-$1.metrics"; }
  rsh "cat $W/serve-$1.log" > "$OUT/serve-$1.log"
  rsh "cat $W/sample-$1.tsv" > "$OUT/sample-$1.tsv" 2>/dev/null
  rsh "sort -k2 -n $W/mem-$1.txt | head -1; grep -c KILLED $W/mem-$1.txt" | tr '\n' ' ' | sed "s/^/min MemAvailable MB (time MB; KILLED count) ($1): /" | tee -a "$OUT/${WLOG:-lib}.log"; echo | tee -a "$OUT/${WLOG:-lib}.log" >/dev/null
  rsh "cat $W/mem-$1.txt" > "$OUT/mem-$1.txt" 2>/dev/null
  log "log lines with warn/error/panic/fail in serve-$1.log: $(grep -ciE 'warn|error|panic|fail' "$OUT/serve-$1.log")"
  stop_server
}
