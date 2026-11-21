#!/usr/bin/env bash
# drill: cutover restart drill, intarsiaml 35B (shipping config) on spark2, beside the production vLLM (never touched).
#   FLEETQ_OWNER=claude-overnight-drill fleetq run --host spark2 --mem 52 --expect 120 --label "drill: ..." -- bash scripts/drill/drill.sh
# One chain of servers S0..S8; each is started (timed on spark2 by ctl.sh), verified (20 greedy requests cold and warm, byte-compared with S0),
# put under 2-stream mixed load (scripts/soak2/replay2.py) plus an SDK consumer (consumer.py), then killed in the way its row says; the next
# server's start is part of the same outage window. STAGES: tag | prefix-store spec | ctl kill command
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-21-drill/raw; mkdir -p "$OUT"
SD=$REPO/scripts/drill; DE=/opt/code/deepeval
HOST=spark2; HN=spark-fa00; PORT=8191; MODEL=guidance; UB=http://$HN:$PORT/v1
RD=intarsiaml-drill/out; DISKPS="host_mb=8192,disk_mb=20000,dir=/home/nvidia/intarsiaml-drill/storedisk"
LOADWARM=${LOADWARM:-40}; PRE=${PRE:-8}
STAGES=("S0|on|term sup" "S1|on|term group" "S2|on|kill9 both" "S3|on|kill9 both" "S4|on|kill9 both" "S5|on|kill9 child" "S6|on|term sup" "S7|$DISKPS|kill9 both" "S8|$DISKPS|kill9 sup" "S9|on|term sup")
[[ -n ${ONLY:-} ]] && IFS=';' read -ra STAGES <<< "$ONLY"   # e.g. ONLY='S0|on|term sup;S1|on|kill9 both' (smoke); OUTSUB=smoke keeps the raw files apart
OUT=$OUT${OUTSUB:+/$OUTSUB}; mkdir -p "$OUT"
T0=$(date +%s); LOG=$OUT/drill.log
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$LOG"; }
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
ctl() { local lf=$1 ps=$2; shift 2; rsh "PS='$ps' bash $RD/ctl.sh $*" 2>&1 | tee -a "$OUT/$lf"; }
epoch() { python3 -c 'import time;print(f"{time.time():.3f}")'; }
offset() { local m0 s m1; m0=$(epoch); s=$(rsh 'date +%s.%N'); m1=$(epoch); python3 -c "print('$1 offset_spark2_minus_mac_s', round($s-($m0+$m1)/2,3), 'rtt_s', round($m1-$m0,3))" | tee -a "$OUT/clock-offset.txt"; }
LPID=""; CPID=""
cleanup() { [[ -n $CPID ]] && kill "$CPID" 2>/dev/null; [[ -n $LPID ]] && kill "$LPID" 2>/dev/null; ctl ctl-cleanup.log on cleanup >/dev/null 2>&1; log "window end"; }
trap cleanup EXIT; trap 'exit 130' INT TERM

vrun() { # vrun tag pass
  ( cd $DE && uv run python "$SD/verify.py" --url $UB --model $MODEL --out "$OUT/verify-$1-p$2.json" ) > "$OUT/verify-$1-p$2.out" 2>&1; log "verify $1 pass $2: $(tail -1 "$OUT/verify-$1-p$2.out")"
  curl -s --max-time 10 "http://$HN:$PORT/metrics" > "$OUT/metrics-$1-after-p$2.txt"; curl -s --max-time 10 "http://$HN:$PORT/v1/intarsia/prefix-store" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"stats":d.get("stats"),"n_entries":len(d.get("entries") or [])}))' > "$OUT/store-$1-after-p$2.json" 2>/dev/null
}
idle_wait() { # wait until the server has nothing running or queued (a cancelled consumer request must not share the first verify batch)
  local t0=$(date +%s) n=1 i; for i in $(seq 1 120); do n=$(curl -s --max-time 5 "http://$HN:$PORT/metrics" | awk '/^intarsia_requests_(running|waiting) / {n+=$2} END {print n+0}'); [[ $n == 0 ]] && break; sleep 1; done
  log "idle wait before verify: running+waiting=$n after $(( $(date +%s)-t0 )) s"; }
verify() { # verify tag: two passes (cold store, then warm), compared with S0's
  local t=$1; idle_wait; curl -s --max-time 10 "http://$HN:$PORT/metrics" > "$OUT/metrics-$t-up.txt"
  vrun "$t" 1; vrun "$t" 2
  { echo "== $t: pass1 vs pass2 (same server)"; python3 "$SD/cmp.py" "$OUT/verify-$t-p1.json" "$OUT/verify-$t-p2.json"
    if [[ $t != S0 ]]; then echo "== $t vs S0: pass1 vs S0 pass1"; python3 "$SD/cmp.py" "$OUT/verify-S0-p1.json" "$OUT/verify-$t-p1.json"; echo "== $t vs S0: pass2 vs S0 pass2"; python3 "$SD/cmp.py" "$OUT/verify-S0-p2.json" "$OUT/verify-$t-p2.json"; fi; } 2>&1 | tee -a "$OUT/cmp.txt" | tee -a "$LOG" >/dev/null
}
finish_client() { # after the next server answered: let the consumer see recovery for 10 s, stop it, wait for the load
  sleep 10; [[ -n $CPID ]] && kill "$CPID" 2>/dev/null; CPID=""; pkill -f 'drill/consumer.py --base' 2>/dev/null; [[ -n $LPID ]] && { wait "$LPID" 2>/dev/null; }; LPID=""
}

offset start; log "drill start, fleetq job $FLEETQ_JOB; binary: $(rsh '~/intarsiaml-drill/bin/intarsia --version | head -1')"
scp -q -o BatchMode=yes "$SD/ctl.sh" "$SD/ttft.py" "$HOST:$RD/" || { log "script copy failed"; exit 1; }
rsh "rm -rf ~/intarsiaml-drill/storedisk; mkdir -p ~/intarsiaml-drill/storedisk"
cp "$REPO/docs/results/2026-11-20-ab35e-cutover-scoreboard/raw/replay-plan.json" "$OUT/replay-plan.json"
ctl ctl-snap.log on snap before-all | tee -a "$LOG" >/dev/null
k=0; skipup=0; prevtag=""
for st in "${STAGES[@]}"; do
  IFS='|' read -r tag ps kill <<< "$st"; k=$((k+1)); log "=== stage $tag (prefix-store '$ps'), then: $kill"
  if [[ $skipup == 0 ]]; then
    ctl "ctl-$tag.log" "$ps" start "$tag" | tee -a "$LOG" | grep -E '^RESULT|FAIL' ; [[ ${PIPESTATUS[0]} == 0 ]] || { log "start $tag failed"; exit 1; }
    [[ -n $LPID$CPID ]] && finish_client
  else skipup=0; finish_client; rsh "ln -sf serve-$prevtag.log $RD/serve-$tag.log; ln -sf mem-$prevtag.txt $RD/mem-$tag.txt"; fi
  rsh "cat $RD/serve-$tag.log" > "$OUT/serve-$tag.log" 2>/dev/null
  [[ -z ${SKIPVERIFY:-} ]] && verify "$tag"
  # steady 2-stream mixed load, then the SDK consumer for the last PRE seconds, then the kill
  seed=$((100+k))
  ( cd $DE && uv run python "$REPO/scripts/soak2/replay2.py" --url $UB --model $MODEL --arm "L$tag" --order interleaved --seed "$seed" --streams 2 \
      --plan "$OUT/replay-plan.json" --trace "$OUT/load-trace-$tag.json" --out "$OUT/load" --temp-mode mixed ) > "$OUT/load-$tag.out" 2>&1 &
  LPID=$!
  sleep "$LOADWARM"
  ( cd $DE && exec uv run python "$SD/consumer.py" --base $UB --model $MODEL --out "$OUT/consumer-$tag.jsonl" --long-tokens "${LONGTOK:-300}" ) > "$OUT/consumer-$tag.out" 2>&1 &
  CPID=$!
  sleep "$PRE"
  fleetq status 2>/dev/null | sed -n '/^spark2/,/^$/p' > "$OUT/fleetq-spark2-before-kill-$tag.txt"; offset "before-kill-$tag"; curl -s --max-time 5 "http://$HN:$PORT/metrics" > "$OUT/metrics-$tag-before-kill.txt"; log "KILL ($kill) at mac epoch $(epoch)"
  set -- $kill
  ctl "ctl-$tag.log" "$ps" "$1" "$tag" "$2" | tee -a "$LOG" | grep -E '^(RESULT|SIGNAL|SNAP)' >/dev/null
  rsh "cat $RD/serve-$tag.log" > "$OUT/serve-$tag.log" 2>/dev/null; rsh "cat $RD/mem-$tag.txt" > "$OUT/mem-$tag.txt" 2>/dev/null
  [[ $kill == "kill9 child" ]] && skipup=1
  prevtag=$tag
done
finish_client
log "all stages done"
