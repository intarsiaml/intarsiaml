#!/usr/bin/env bash
# soak lane helpers (sourced by window_a.sh / window_b.sh / vllm_arms.sh), adapted from scripts/interleave/window.sh.
# Everything runs inside a fleetq slot (FLEETQ_JOB set), so localeval does not queue again.
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-13-cutover-soak/raw; mkdir -p "$OUT"
SC=$REPO/scripts/soak; IL=$REPO/scripts/interleave
HOST=spark2; HN=spark-fa00; PORT=8191; VPORT=8000; CT=missioncntrl-guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN=${BIN:-'~/intarsia-soak/bin/intarsia'}; MS=131072; MB=40
W='~/intarsia-soak/out'
URL=http://$HN:$PORT/v1; VURL=http://$HN:$VPORT/v1
T0=$(date +%s); SPID=""; WPID=""; SAMP=""; DOWN=0; DOWN_T=0; CAPPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$OUT/${WLOG:-lib}.log"; }
# replay2 <arm> <url> <order> <seed> <streams> <trace-file>: plan is shared (copied from the interleave raw dir), the trace is
# recorded by the first run with that file name (vLLM arm) and replayed by the others
replay2() {
  echo "== $1 replay2 k=$5 $3 seed $4 $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$IL/replay2.py" --url "$2" --model guidance --arm "$1" --order "$3" --seed "$4" --streams "$5" \
      --plan "$OUT/replay-plan.json" --trace "$6" --out "$OUT" ) > "$OUT/$1-replay2-$3-seed$4.out" 2>&1
  tail -12 "$OUT/$1-replay2-$3-seed$4.out"
}
# ---- intarsia serve on the box
stop_server() {
  [[ -n "$SAMP" ]] && { rsh "kill $SAMP 2>/dev/null"; SAMP=""; }
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'bin/intarsia serve'; sleep 3"; SPID=""; }
}
start_serve() { # start_serve <name> <extra serve args...>   (env assignments: SERVE_ENV)
  local name=$1; shift
  log "serve arm $name: ${SERVE_ENV:+$SERVE_ENV }$BIN serve --backend cuda --model <HF snapshot 1355db6a> --port $PORT --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f $W/mem-$name.txt $W/sample-$name.tsv; mkdir -p $W"
  SPID=$(rsh "nohup env ${SERVE_ENV:-} $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $* > $W/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $W/mem-$name.txt > /dev/null 2>&1 < /dev/null & echo \$!")
  SAMP=$(rsh "nohup bash $W/sampler.sh $SPID $PORT $W/sample-$name.tsv ${SAMPLE_S:-60} > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 $W/serve-$name.log" | tee -a "$OUT/${WLOG:-lib}.log"; finish_serve "$name" 1; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |prefix store: on' $W/serve-$name.log | tr '\n' ' '")"
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$name.metrics.before" 2>/dev/null
}
finish_serve() { # finish_serve <name> [failed]
  [[ -z "${2:-}" ]] && { sleep 3; curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null; }
  rsh "cat $W/serve-$1.log" > "$OUT/serve-$1.log"
  rsh "cat $W/sample-$1.tsv" > "$OUT/sample-$1.tsv" 2>/dev/null
  rsh "sort -k2 -n $W/mem-$1.txt | head -1; grep -c KILLED $W/mem-$1.txt" | tr '\n' ' ' | sed "s/^/min MemAvailable MB (time MB; KILLED count) ($1): /" | tee -a "$OUT/${WLOG:-lib}.log"; echo | tee -a "$OUT/${WLOG:-lib}.log" >/dev/null
  rsh "cat $W/mem-$1.txt" > "$OUT/mem-$1.txt" 2>/dev/null
  stop_server
}
# ---- vLLM take-down with guaranteed restore
restore() {
  trap - EXIT INT TERM; [[ -n "$CAPPID" ]] && kill $CAPPID 2>/dev/null
  stop_server
  if [[ $DOWN == 1 ]]; then
    log "restoring vLLM: docker start $CT"
    rsh "docker start $CT" 2>&1 | tee -a "$OUT/${WLOG:-lib}.log"
    local i; for i in $(seq 1 150); do
      [[ $(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://$HN:$VPORT/v1/models") == 200 ]] && { log "vLLM /v1/models 200 at $(date +%T) (vLLM down for $(( $(date +%s)-DOWN_T ))s total)"; DOWN=0; break; }
      sleep 4
    done
    [[ $DOWN == 1 ]] && log "WARNING: vLLM did not answer /v1/models within 10 min after docker start"
  fi
  log "window end"
}
take_vllm_down() { # take_vllm_down <cap-min>: cap counts from here; restart adds about 3.5 min (measured 3.2 to 3.4 in earlier windows)
  trap restore EXIT; trap 'restore; exit 130' INT TERM
  rsh "mkdir -p $W" && scp -q -o BatchMode=yes "$SC/memwatch.sh" "$SC/sampler.sh" "$HOST:intarsia-soak/out/" || { log "script copy failed; aborting before vLLM stop"; exit 1; }
  log "vLLM up: $(curl -s -o /dev/null -w '%{http_code}' $VURL/models); earlyoom: $(rsh 'systemctl is-active earlyoom')"
  rsh "docker inspect $CT --format '{{.State.StartedAt}} {{.Config.Image}}'" | tee -a "$OUT/${WLOG:-lib}.log"
  rsh "docker stop $CT" 2>&1 | tee -a "$OUT/${WLOG:-lib}.log"; DOWN=1; DOWN_T=$(date +%s); T0=$DOWN_T
  log "vLLM DOWN (docker stop done) at $(date +%T)"
  (sleep $(( $1*60 )); echo "[cap] $1 min down reached, restoring" >> "$OUT/${WLOG:-lib}.log"; kill -TERM $$) >/dev/null 2>&1 &
  CAPPID=$!
  rsh "free -g | sed -n 2p" | tee -a "$OUT/${WLOG:-lib}.log"
}
# ---- vLLM request counters (to detect foreign traffic inside a measurement)
vcount() { curl -s --max-time 5 "http://$HN:$VPORT/metrics" | awk '/^vllm:request_success_total/ {s+=$2} /^vllm:prompt_tokens_total/ {p=$2} /^vllm:generation_tokens_total/ {g=$2} END {printf "success=%d prompt_tokens=%d gen_tokens=%d\n", s, p, g}'; }
# ---- localeval GSM8K (pinned template, no nonce, temperature 0 sent by localeval)
RDIR=$REPO/.soak-recipes; mkdir -p "$RDIR"
recipe_file() { printf '%s\n' "$2" > "$RDIR/$1.txt"; echo "$RDIR/$1.txt"; }
keep() { # keep <out-log> <label>: copy the localeval run's summary/result next to the log and note the run id
  local rid; rid=$(grep -aoE 'results/[0-9]{8}-[^ ]*' "$1" | head -1 | sed 's#results/##; s#/.*##')
  echo "$2 run_id=$rid" | tee -a "$OUT/index.txt"
  [[ -n $rid && -d /opt/code/deepeval/results/$rid ]] && { mkdir -p "$OUT/$rid"; cp /opt/code/deepeval/results/$rid/summary.md /opt/code/deepeval/results/$rid/result.json "$OUT/$rid/" 2>/dev/null; }
}
# gsm8k <label> <url> <recipe-file> <thinking on|off> <limit> <max-tokens>
gsm8k() {
  echo "== $1 gsm8k thinking=$4 n=$5 max_tokens=$6 $(date +%T)"
  ( cd /opt/code/deepeval && uv run localeval quality --url "$2" --model guidance --thinking "$4" --structured prompt --wait-idle 600 --recipe "$3" \
      --bench gsm8k --limit "$5" --max-tokens "$6" --pin-gsm8k --label "soak-$1-gsm8k" ) > "$OUT/$1-gsm8k.out" 2>&1
  tail -3 "$OUT/$1-gsm8k.out"; keep "$OUT/$1-gsm8k.out" "$1 gsm8k"
}
