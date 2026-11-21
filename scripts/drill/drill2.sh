#!/usr/bin/env bash
# drill2: SIGTERM past the supervisor's 30 s grace (35B shipping config) on spark1, beside production vLLM (never touched).
#   FLEETQ_OWNER=claude-overnight-drill2 fleetq run --host spark1 --mem 52 --expect 40 --label "drill2: ..." -- bash scripts/drill/drill2.sh
# Stages: R1 (verify baseline, then long generations, SIGTERM the supervisor at SIGAT s), R2, R3 (long generations only), V (restart, 20 greedy verify).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-21-drill2/raw; mkdir -p "$OUT"
SD=$REPO/scripts/drill; DE=/opt/code/deepeval
HOST=spark1; HN=spark-4e77; PORT=8191; MODEL=guidance; UB=http://$HN:$PORT/v1; RD=intarsiaml-drill/out
SIGAT=${SIGAT:-5}; LONGTOK=${LONGTOK:-12000}; STREAMS=${STREAMS:-2}
T0=$(date +%s); LOG=$OUT/drill.log
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$LOG"; }
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
ctl() { local lf=$1; shift; rsh "bash $RD/ctl.sh $*" 2>&1 | tee -a "$OUT/$lf"; }
epoch() { python3 -c 'import time;print(f"{time.time():.3f}")'; }
offset() { local m0 s m1; m0=$(epoch); s=$(rsh 'date +%s.%N'); m1=$(epoch); python3 -c "print('$1 offset_spark_minus_mac_s', round($s-($m0+$m1)/2,3), 'rtt_s', round($m1-$m0,3))" | tee -a "$OUT/clock-offset.txt"; }
cleanup() { ctl ctl-cleanup.log cleanup >/dev/null 2>&1; log "window end"; }
trap cleanup EXIT; trap 'exit 130' INT TERM
vrun() { ( cd $DE && uv run python "$SD/verify.py" --url $UB --model $MODEL --out "$OUT/verify-$1.json" ) > "$OUT/verify-$1.out" 2>&1; log "verify $1: $(tail -1 "$OUT/verify-$1.out")"; }

offset start; log "drill2 start, fleetq job $FLEETQ_JOB; binary: $(rsh '~/intarsiaml-drill/bin/intarsia --version | head -1')"
scp -q -o BatchMode=yes "$SD/ctl.sh" "$SD/ttft.py" "$HOST:$RD/" || { log "script copy failed"; exit 1; }
ctl ctl-snap.log snap before-all >/dev/null
for tag in R1 R2 R3 V; do
  log "=== stage $tag"
  ctl "ctl-$tag.log" start "$tag" | tee -a "$LOG" | grep -E '^RESULT|FAIL'; [[ ${PIPESTATUS[0]} == 0 ]] || { log "start $tag failed"; exit 1; }
  rsh "cat $RD/serve-$tag.log" > "$OUT/serve-$tag.log" 2>/dev/null
  if [[ $tag == R1 ]]; then vrun baseline; fi
  if [[ $tag == V ]]; then vrun restart; curl -s --max-time 10 "http://$HN:$PORT/metrics" > "$OUT/metrics-V.txt"
    python3 "$SD/cmp.py" "$OUT/verify-baseline.json" "$OUT/verify-restart.json" 2>&1 | tee "$OUT/cmp.txt" | tee -a "$LOG"
    ctl "ctl-$tag.log" term "$tag" sup | tee -a "$LOG" | grep -E '^RESULT'; break; fi
  ( cd $DE && uv run python "$SD/longgen.py" --base $UB --model $MODEL --out "$OUT/long-$tag.jsonl" --max-tokens "$LONGTOK" --streams "$STREAMS" ) > "$OUT/long-$tag.out" 2>&1 &
  LP=$!; sleep "$SIGAT"
  curl -s --max-time 5 "http://$HN:$PORT/metrics" > "$OUT/metrics-$tag-before-kill.txt"; offset "before-term-$tag"
  rsh "free -m | sed -n 2p; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader" > "$OUT/box-before-term-$tag.txt" 2>&1
  log "TERM at mac epoch $(epoch)"
  ctl "ctl-$tag.log" term "$tag" sup | tee -a "$LOG" | grep -E '^(RESULT|SIGNAL)'
  rsh "cat $RD/serve-$tag.log" > "$OUT/serve-$tag.log" 2>/dev/null; rsh "cat $RD/mem-$tag.txt" > "$OUT/mem-$tag.txt" 2>/dev/null
  wait "$LP"; log "long clients done: $(cat "$OUT/long-$tag.jsonl" | wc -l) lines"
done
log "all stages done"
