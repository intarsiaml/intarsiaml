#!/usr/bin/env bash
# herd gate (b): ABBA over `intarsia serve --max-batch 8 --prefix-store on` on spark2, herd off (A) and on (B), vLLM stays up.
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 58 --expect 70 --label "herd ABBA" -- bash scripts/herd/window.sh [A B B A]
# One server per visit (INTARSIA_PREFIX_HERD is read at startup), a fresh store per visit. Per visit: herd8 seeds 1 2 3 (8 at once, one
# new 16448-token prefix + 200 own tokens each), distinct (8 unrelated 8192-token prompts), solo seeds 1 2 3 (the herd8 prompts, one at a time).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$REPO/scripts/herd
OUT=$REPO/docs/results/2026-11-17-herd/raw; mkdir -p "$OUT"
HOST=spark2; HN=spark-fa00; PORT=8193
L='~/intarsiaml-herd'; BIN="$L/bin/intarsia"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(A B B A)
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
SPID=""; WPID=""
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'intarsiaml-herd/bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsiaml-herd/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap 'stop_server' EXIT; trap 'stop_server; exit 130' INT TERM
scp -q -o BatchMode=yes "$S/memwatch.sh" "$HOST:intarsiaml-herd/memwatch.sh"
log "window start: arms ${ARMS[*]}; earlyoom $(rsh 'systemctl is-active earlyoom'); vLLM /v1/models $(curl -s -o /dev/null -w '%{http_code}' http://$HN:8000/v1/models); $(rsh "free -g | sed -n 2p")"
v=0
for arm in "${ARMS[@]}"; do
  v=$((v+1)); name="$arm$v"
  ENVV=""; [[ $arm == B ]] && ENVV="INTARSIA_PREFIX_HERD=1"
  log "visit $name: serve (herd $([[ $arm == B ]] && echo ON || echo off)) $(rsh "free -g | sed -n 2p")"
  rsh "rm -f $L/out/mem-$name.txt"
  SPID=$(rsh "env $ENVV nohup $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq 24576 --max-batch 8 --prefix-store on --mem-budget 40 --profile-every 0 > $L/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $L/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 120); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 3; done
  if [[ $up != 1 ]]; then log "serve $name did not come up"; rsh "tail -20 $L/out/serve-$name.log" | tee -a "$OUT/window.log"; stop_server; continue; fi
  log "serve $name up; $(rsh "free -g | sed -n 2p")"
  C="python3 $S/herd_bench.py --url http://$HN:$PORT --arm $arm --visit $name --out $OUT/results.jsonl"
  # warm the kernels / graphs (a short unrelated prompt), result not recorded
  $C --mode distinct --seed 900 --n 2 --distinct-len 512 --max-tokens 8 --out /dev/null > /dev/null 2>&1
  for seed in 1 2 3; do $C --mode herd8 --seed $seed 2>&1 | tee -a "$OUT/window.log"; done
  $C --mode distinct --seed 11 2>&1 | tee -a "$OUT/window.log"
  for seed in 1 2 3; do $C --mode solo --seed $seed 2>&1 | tee -a "$OUT/window.log"; done
  curl -sf "http://$HN:$PORT/v1/intarsia/prefix-store" > "$OUT/prefix-store-$name.json"
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/metrics-$name.txt"
  rsh "cat $L/out/serve-$name.log" > "$OUT/serve-$name.log"
  rsh "sort -k2 -n $L/out/mem-$name.txt | head -1; grep -c KILLED $L/out/mem-$name.txt" | tr '\n' ' ' | sed "s/^/min MemAvailable MB (time MB; KILLED count) ($name): /" | tee -a "$OUT/window.log"; echo | tee -a "$OUT/window.log" > /dev/null
  stop_server
done
log "window end; $(rsh "free -g | sed -n 2p")"
