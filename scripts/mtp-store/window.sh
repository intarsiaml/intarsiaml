#!/usr/bin/env bash
# lane mtp-store: paired speed A/B on spark1, vLLM left up (nothing is taken down). One build (~/intarsia-mtp-store/bin/intarsia), one
# `intarsia serve --max-batch 2 --prefix-store on` at a time, a fresh server per run (an empty store each time):
#   A  INTARSIA_MTP_STORE=0   the shape before the lane: the store skips the recipe's MTP default, no request drafts
#   B  (unset)                the lane's default on CUDA: the MTP default applies (k 2, batched MTP N = 1), restored requests draft
#   C  INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1   as B, drafting while both streams decode (optional, ARMS)
#   D  no --prefix-store, the recipe's MTP default; E  no --prefix-store, --spec off (is the ITL p99 rise there without a store?)
#   F, G  as D, E on the origin/main binary (bin/intarsia-main, built with BINNAME=intarsia-main SHA=<main> build.sh)
# Each run replays the multi-turn trace of lane interleave (scripts/interleave/replay2.py: 8 conversations x 6 turns, temperature 0,
# thinking off, 160 tokens) at K streams. ARMS: space-separated <arm>:<K>[:tag] in run order (default: A:2 B:2 B:2:r A:2:r A:1 B:1).
#   fleetq run --host spark1 --quiet gpu --mem 55 --expect 45 --label "mtp-store: A/B" -- bash scripts/mtp-store/window.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-13-mtp-store/raw; mkdir -p "$OUT"
IL=$REPO/scripts/interleave; ILRAW=$REPO/docs/results/2026-11-13-interleave/raw
HOST=spark1; HN=spark-4e77; PORT=8191; VPORT=8000
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsia-mtp-store/bin/intarsia'; BIN_MAIN='~/intarsia-mtp-store/bin/intarsia-main'; W='~/intarsia-mtp-store/out'; MS=131072; MB=${MB:-40}
ARMS=${ARMS:-"A:2 B:2 B:2:r A:2:r A:1 B:1"}
URL=http://$HN:$PORT/v1
SPID=""; WPID=""; T0=$(date +%s)
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$OUT/window.log"; }
vcount() { curl -s --max-time 5 "http://$HN:$VPORT/metrics" | awk '/^vllm:request_success_total/ {s+=$2} /^vllm:generation_tokens_total/ {g=$2} END {printf "success=%d gen_tokens=%d", s, g}'; }
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'intarsia-mtp-store/bin/intarsia[-a-z]* serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsia-mtp-store/bin/intarsia[-a-z]* serve'; sleep 3"; SPID=""; }
}
trap 'stop_server; log "window end"' EXIT; trap 'exit 130' INT TERM
start_serve() { # start_serve <name> <env-assignments or -> <serve args...>
  local name=$1 envs=$2; shift 2; [[ $envs == - ]] && envs=""
  log "serve $name: ${envs:+$envs }$BIN serve --backend cuda --model <HF snapshot 1355db6a> --port $PORT --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f $W/mem-$name.txt; mkdir -p $W"
  SPID=$(rsh "nohup env $envs $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name mtpstore --max-seq $MS --mem-budget $MB --profile-every 0 $* > $W/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $W/mem-$name.txt > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 $W/serve-$name.log" | tee -a "$OUT/window.log"; rsh "cat $W/serve-$name.log" > "$OUT/serve-$name.log"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |spec mtp|batched MTP|prefix store: (on|--spec|a request)' $W/serve-$name.log | tr '\n' ' '")"
}
finish_serve() { # finish_serve <name>
  sleep 3
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat $W/serve-$1.log" > "$OUT/serve-$1.log"
  rsh "sort -k2 -n $W/mem-$1.txt | head -1; grep -c KILLED $W/mem-$1.txt" | tr '\n' ' ' | sed "s/^/min MemAvailable (time MB; KILLED count) ($1): /" | tee -a "$OUT/window.log"; echo >> "$OUT/window.log"
  stop_server
}
run_arm() { # run_arm <arm> <K> [tag]
  local arm=$1 k=$2 tag=${3:-} envs args="--max-batch 2 --prefix-store on" BIN=$BIN
  case $arm in
    A) envs="INTARSIA_MTP_STORE=0";;
    B) envs="-";;
    C) envs="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1";;
    D) envs="-"; args="--max-batch 2";;              # store off, the recipe's MTP default (main's shipped shape without a store)
    E) envs="-"; args="--max-batch 2 --spec off";;   # store off, no MTP
    F) envs="-"; args="--max-batch 2"; BIN=$BIN_MAIN;;             # as D on the origin/main binary (bin/intarsia-main)
    G) envs="-"; args="--max-batch 2 --spec off"; BIN=$BIN_MAIN;;  # as E on the origin/main binary
    *) log "unknown arm $arm"; return 1;;
  esac
  local name="$arm-k$k${tag:+-$tag}"
  start_serve "$name" "$envs" $args || return 1
  local v0; v0=$(vcount)
  echo "== $name replay2 k=$k $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$IL/replay2.py" --url "$URL" --model mtpstore --arm "$name" --order interleaved --seed 0 --streams "$k" \
      --plan "$OUT/replay-plan.json" --trace "$OUT/replay-trace-seed0.json" --out "$OUT" ) > "$OUT/$name-replay2.out" 2>&1
  tail -10 "$OUT/$name-replay2.out" | tee -a "$OUT/window.log"
  log "$name vLLM counters before: $v0, after: $(vcount) (a change is foreign GPU traffic inside the run)"
  finish_serve "$name"
}
cp -n "$ILRAW/replay-plan.json" "$ILRAW/replay-trace-seed0.json" "$OUT/"
rsh "mkdir -p $W" && scp -q -o BatchMode=yes "$REPO/scripts/soak/memwatch.sh" "$HOST:intarsia-mtp-store/out/" || { log "script copy failed"; exit 1; }
log "window start; arms: $ARMS; vLLM up: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); binary: $(rsh "$BIN --version | head -1")"
for a in $ARMS; do
  IFS=: read -r arm k tag <<< "$a"
  run_arm "$arm" "$k" "${tag:-}" || log "arm $a failed"
done
log "vLLM up at the end: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); $(rsh 'free -g | sed -n 2p')"
