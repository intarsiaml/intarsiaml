#!/usr/bin/env bash
# ab35b window. usage: window.sh w1|w2 <cap-minutes>
# Runs INSIDE: FLEETQ_OWNER=claude-ab35b fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect N ...
# Stops spark2's vLLM, runs the arms of this window one `intarsia serve` at a time, restores vLLM on ANY exit (trap) and waits
# for /v1/models 200. Hard cap on vLLM downtime = <cap-minutes>; no new arm starts after 80 % of it.
#   w1: B-batch2 (--max-batch 2, shipped defaults: MTP k2): replay x2 orders, gsm8k n=200, sampled + MTP acceptance, tool smoke;
#       then B-batch2-specoff (--spec off): gsm8k n=200 (MTP-vs-off exact-reply equality)
#   w2: B-single (default --max-batch 1 single-sequence engine, no prefix store): replay x2 orders;
#       sampled decode + MTP acceptance and tool smoke on B-single (sampled requests speculate only in the single-sequence engine);
#       B-store (--prefix-store on; recipe: a configured store keeps plain decode): replay x2 orders
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
W=$1; CAPMIN=${2:-45}
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
HOST=spark2; PORT=8191; VPORT=8000; CT=missioncntrl-guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsia-ab35b/bin/intarsia'; MS=131072; MB=40
CAPSEC=$((CAPMIN*60)); SOFT=$((CAPSEC*8/10))
SPID=""; WPID=""; DOWN=0; DOWN_T=0; T0=$(date +%s); LEPID=""; CAPPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$OUT/window-$W.log"; }
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'intarsia-ab35b/bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsia-ab35b/bin/intarsia serve'; sleep 3"; SPID=""; }
}
restore() {
  trap - EXIT INT TERM; [[ -n "$CAPPID" ]] && kill $CAPPID 2>/dev/null
  [[ -n "$LEPID" ]] && kill $LEPID 2>/dev/null
  stop_server
  if [[ $DOWN == 1 ]]; then
    log "restoring vLLM: docker start $CT"
    rsh "docker start $CT" 2>&1 | tee -a "$OUT/window-$W.log"
    local i; for i in $(seq 1 150); do
      [[ $(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://$HN:$VPORT/v1/models") == 200 ]] && { log "vLLM /v1/models 200 at $(date +%T) (vLLM down for $(( $(date +%s)-DOWN_T ))s total)"; DOWN=0; break; }
      sleep 4
    done
    [[ $DOWN == 1 ]] && log "WARNING: vLLM did not answer /v1/models within 10 min after docker start"
  fi
  log "window end"
}
trap restore EXIT; trap 'restore; exit 130' INT TERM
start_serve() { # start_serve <name> <extra args...>
  local name=$1; shift
  log "serve arm $name: serve --backend cuda --model <HF snapshot 1355db6a> --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $*"
  rsh "rm -f ~/intarsia-ab35b/out/mem-$name.txt"
  SPID=$(rsh "nohup $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq $MS --mem-budget $MB --profile-every 0 $* > ~/intarsia-ab35b/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash ~/intarsia-ab35b/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 ~/intarsia-ab35b/out/serve-$name.log" | tee -a "$OUT/window-$W.log"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded ' ~/intarsia-ab35b/out/serve-$name.log")"
}
finish_serve() { # finish_serve <name>
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat ~/intarsia-ab35b/out/serve-$1.log" > "$OUT/serve-$1.log"
  rsh "sort -k2 -n ~/intarsia-ab35b/out/mem-$1.txt | head -1; grep -c KILLED ~/intarsia-ab35b/out/mem-$1.txt" | tr '\n' ' ' | sed "s/^/min MemAvailable MB (time MB; KILLED count) ($1): /" | tee -a "$OUT/window-$W.log"; echo | tee -a "$OUT/window-$W.log" >/dev/null
  stop_server
}
URL=http://$HN:$PORT/v1
log "window $W start (cap ${CAPMIN} min); vLLM up: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom')"
rsh "docker inspect $CT --format '{{.State.StartedAt}} {{.Config.Image}}'" | tee -a "$OUT/window-$W.log"
rsh "docker stop $CT" 2>&1 | tee -a "$OUT/window-$W.log"; DOWN=1; DOWN_T=$(date +%s)
log "vLLM DOWN (docker stop done) at $(date +%T)"
(sleep $CAPSEC; echo "[cap] ${CAPMIN} min down reached, restoring" >> "$OUT/window-$W.log"; kill -TERM $$) >/dev/null 2>&1 &
CAPPID=$!
rsh "free -g | sed -n 2p; nvidia-smi --query-gpu=memory.used --format=csv,noheader" | tee -a "$OUT/window-$W.log"
late() { (( $(date +%s)-T0 > SOFT )); }
if [[ $W == w1 ]]; then
  RB=$(recipe_file B-default "intarsia serve --backend cuda (HF snapshot nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a) --max-seq 131072 --mem-budget 40 --max-batch 2 --profile-every 0; shipped CUDA defaults: MTP k2 via batched_mtp, bf16 KV, prefill chunk 2048, macro 8192, tool parser hermes")
  RO=$(recipe_file B-specoff "intarsia serve --backend cuda (HF snapshot nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a) --max-seq 131072 --mem-budget 40 --max-batch 2 --profile-every 0 --spec off")
  if start_serve B-default --max-batch 2; then
    replay B-default "$URL" interleaved 0
    replay B-default "$URL" sequential 1
    gsm8k B-default "$URL" "$RB"
    sampled B-default "$URL" "http://$HN:$PORT/metrics"
    tools B-default "$URL"
    finish_serve B-default
  fi
  if ! late && start_serve B-specoff --max-batch 2 --spec off; then
    gsm8k B-specoff "$URL" "$RO"
    finish_serve B-specoff
  else log "skipping spec-off arm (time or start failure)"; fi
else
  if start_serve B-single; then
    replay B-single "$URL" interleaved 0
    replay B-single "$URL" sequential 1
    sampled B-single "$URL" "http://$HN:$PORT/metrics"
    tools B-single "$URL"
    finish_serve B-single
  fi
  if ! late && start_serve B-store --prefix-store on; then
    replay B-store "$URL" interleaved 0
    replay B-store "$URL" sequential 1
    finish_serve B-store
  else log "skipping store arm (time or start failure)"; fi
fi
