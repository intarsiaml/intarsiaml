#!/usr/bin/env bash
# lane interleave: spark2 vLLM-down window, main vs this branch at 2 streams with the prefix store (template: scripts/ab35c/window3.sh
# on branch ab35c). usage: window.sh <cap-minutes> [k3]
# Runs INSIDE: FLEETQ_OWNER=claude-interleave fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 45 ...
# Stops spark2's vLLM, runs the arms one `intarsia serve` at a time, restores vLLM on ANY exit (trap) and waits for /v1/models 200.
# Hard cap on vLLM downtime = <cap-minutes>; no new arm starts after 80 % of it.
#   E-main  (main 10b235785, --max-batch 2 --prefix-store on): replay2 interleaved seed 0, K=2 (window 3's D-store setup)
#   E-il    (this branch, same flags; INTARSIA_PREFILL_PAUSE_ROWS / INTARSIA_SETTLE_SLICE_MB at their defaults): same replay
#   with "k3": both again at K=3 (E-main-k3, E-il-k3) while time allows
# Binaries: BINMAIN / BINIL (built by scripts/interleave/build.sh on spark2).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
CAPMIN=${1:-35}; K3=${2:-}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-13-interleave/raw; mkdir -p "$OUT"
S=$REPO/scripts/ab35b; SC=$REPO/scripts/interleave
HOST=spark2; HN=spark-fa00; PORT=8191; VPORT=8000; CT=missioncntrl-guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BINMAIN=${BINMAIN:-'~/intarsia-il-main/bin/intarsia'}; BINIL=${BINIL:-'~/intarsia-interleave/bin/intarsia'}; MS=131072; MB=40
W='~/intarsia-interleave/out'
CAPSEC=$((CAPMIN*60)); SOFT=$((CAPSEC*8/10))
SPID=""; WPID=""; DOWN=0; DOWN_T=0; T0=$(date +%s); CAPPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$OUT/window.log"; }
# replay2 <arm> <url> <order> <seed> [streams=2]: K-concurrent-stream replay, window 3's plan and trace (copied into $OUT)
replay2() {
  local k="${5:-2}"
  echo "== $1 replay2 k=$k $3 seed $4 $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$SC/replay2.py" --url "$2" --model guidance --arm "$1" --order "$3" --seed "$4" --streams "$k" \
      --plan "$OUT/replay-plan.json" --trace "$OUT/replay-trace-seed$4.json" --out "$OUT" ) > "$OUT/$1-replay2-$3-seed$4.out" 2>&1
  tail -14 "$OUT/$1-replay2-$3-seed$4.out"
}
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'bin/intarsia serve'; sleep 3"; SPID=""; }
}
restore() {
  trap - EXIT INT TERM; [[ -n "$CAPPID" ]] && kill $CAPPID 2>/dev/null
  stop_server
  if [[ $DOWN == 1 ]]; then
    log "restoring vLLM: docker start $CT"
    rsh "docker start $CT" 2>&1 | tee -a "$OUT/window.log"
    local i; for i in $(seq 1 150); do
      [[ $(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://$HN:$VPORT/v1/models") == 200 ]] && { log "vLLM /v1/models 200 at $(date +%T) (vLLM down for $(( $(date +%s)-DOWN_T ))s total)"; DOWN=0; break; }
      sleep 4
    done
    [[ $DOWN == 1 ]] && log "WARNING: vLLM did not answer /v1/models within 10 min after docker start"
  fi
  log "window end"
}
trap restore EXIT; trap 'restore; exit 130' INT TERM
start_serve() { # start_serve <name> <bin> <env-assignments or -> <serve args...>
  local name=$1 bin=$2 envs=$3; shift 3; [[ $envs == - ]] && envs=""
  log "serve arm $name: ${envs:+$envs }$bin serve --backend cuda --model <HF snapshot 1355db6a> --port $PORT --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f $W/mem-$name.txt; mkdir -p $W"
  SPID=$(rsh "nohup env $envs $bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $* > $W/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 $W/serve-$name.log" | tee -a "$OUT/window.log"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded ' $W/serve-$name.log | tr '\n' ' '")"
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$name.metrics.before" 2>/dev/null
}
finish_serve() { # finish_serve <name>
  sleep 3  # let an idle settle land before the scrape
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat $W/serve-$1.log" > "$OUT/serve-$1.log"
  rsh "sort -k2 -n $W/mem-$1.txt | head -1; grep -c KILLED $W/mem-$1.txt" | tr '\n' ' ' | sed "s/^/min MemAvailable MB (time MB; KILLED count) ($1): /" | tee -a "$OUT/window.log"; echo | tee -a "$OUT/window.log" >/dev/null
  stop_server
}
late() { (( $(date +%s)-T0 > SOFT )); }
arm() { # arm <name> <bin> <streams>
  late && { log "skipping $1 (past 80 % of the cap)"; return; }
  start_serve "$1" "$2" INTARSIA_PREFIX_STORE_DEBUG=1 --max-batch 2 --prefix-store on || return
  replay2 "$1" "$URL" interleaved 0 "$3"
  finish_serve "$1"
}
URL=http://$HN:$PORT/v1
log "window start (cap ${CAPMIN} min); vLLM up: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom')"
log "binaries: main $(rsh "$BINMAIN --version | head -1"); interleave $(rsh "$BINIL --version | head -1")"
rsh "mkdir -p $W" && scp -q -o BatchMode=yes "$S/memwatch.sh" "$HOST:intarsia-interleave/out/memwatch.sh" || { log "memwatch copy failed; aborting before vLLM stop"; exit 1; }
rsh "docker inspect $CT --format '{{.State.StartedAt}} {{.Config.Image}}'" | tee -a "$OUT/window.log"
rsh "docker stop $CT" 2>&1 | tee -a "$OUT/window.log"; DOWN=1; DOWN_T=$(date +%s)
log "vLLM DOWN (docker stop done) at $(date +%T)"
(sleep $CAPSEC; echo "[cap] ${CAPMIN} min down reached, restoring" >> "$OUT/window.log"; kill -TERM $$) >/dev/null 2>&1 &
CAPPID=$!
rsh "free -g | sed -n 2p; nvidia-smi --query-gpu=memory.used --format=csv,noheader" | tee -a "$OUT/window.log"

arm E-main "$BINMAIN" 2
arm E-il "$BINIL" 2
if [[ $K3 == k3 ]]; then
  arm E-il-k3 "$BINIL" 3
  arm E-main-k3 "$BINMAIN" 3
fi
