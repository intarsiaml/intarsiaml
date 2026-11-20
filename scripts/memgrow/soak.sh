#!/usr/bin/env bash
# memgrow: long soak of the shipping config on spark1 with a 30 s memory sampler; vLLM stays up and is sampled too.
#   FLEETQ_OWNER=claude-overnight-memgrow fleetq run --host spark1 --mem 52 --expect 150 --label "memgrow soak" -- bash scripts/memgrow/soak.sh <tag> [extra serve args]
# Env: SOAKMIN (load minutes, default 105), IDLEMIN (idle minutes after the load, default 10), SEED0 (first replay seed, default 201),
#      SERVE_EXTRA (extra `intarsia serve` args, e.g. "--prefix-store off"), INTARSIA_ENV (env assignments for the server, e.g. "INTARSIA_BATCHED_MTP=0").
# Phases (marks in $OUT/marks-<tag>.tsv): base (keepalive sampler only, no server) -> start -> settle (server idle) -> load -> idle -> stop -> after.
# Binary: ~/intarsiaml-memgrow/bin/intarsia (scripts/memgrow/build.sh). Remove ~/intarsiaml-memgrow when the lane is done.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TAG=${1:?tag}; shift; EXTRA="${SERVE_EXTRA:-} $*"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-memgrow/raw; mkdir -p "$OUT" "$OUT/replay-$TAG"
HOST=spark1; HN=spark-4e77; PORT=8191
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-memgrow/bin/intarsia'; W='~/intarsiaml-memgrow/out'
SERVE_ARGS="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on $EXTRA"
SOAKMIN=${SOAKMIN:-105}; IDLEMIN=${IDLEMIN:-10}; seed=${SEED0:-201}
UB=http://$HN:$PORT/v1
KA=""; SPID=""; T0=$(date +%s); WLOG=$OUT/window-$TAG.log; MARKS=$OUT/marks-$TAG.tsv; CSV=$W/mem-$TAG.csv
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
mark() { printf '%s\t%s\n' "$(date +%s)" "$1" >> "$MARKS"; log "mark $1"; }
avail() { rsh "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"; }
stop_server() { # the supervisor is $SPID; the engine is its child: TERM both by pid, KILL after 40 s, then log what is left
  [[ -n "$SPID" ]] || return 0
  rsh "ch=\$(pgrep -P $SPID); kill -TERM $SPID \$ch 2>/dev/null; for i in \$(seq 1 40); do n=0; for p in $SPID \$ch; do kill -0 \$p 2>/dev/null && n=1; done; [ \$n = 0 ] && exit 0; sleep 1; done; kill -9 $SPID \$ch 2>/dev/null; sleep 3"
  SPID=""; log "after stop, processes of ours left: [$(rsh "pgrep -f 'intarsiaml-memgrow/bin/intarsia serve' | tr '\n' ' '")]"
}
stop_all() {
  stop_server
  [[ -n "$KA" ]] && { rsh "kill $KA 2>/dev/null"; KA=""; }
  [[ -n "${WD:-}" ]] && { rsh "kill $WD 2>/dev/null"; WD=""; }
}
trap 'stop_all; log "window end"' EXIT; trap 'exit 130' INT TERM
: > "$MARKS"
scp -q -o BatchMode=yes "$REPO/scripts/memgrow/sampler.sh" "$HOST:intarsiaml-memgrow/out/" || { log "sampler copy failed"; exit 1; }
rsh "ss -ltn | grep -q ':$PORT '" && { log "port $PORT busy, abort"; exit 1; }
log "tag $TAG; fleetq job $FLEETQ_JOB; earlyoom $(rsh 'systemctl is-active earlyoom'); avail $(avail) GB; binary $(rsh "$BIN --version | head -1")"
log "serve args: $SERVE_ARGS; env: ${INTARSIA_ENV:-none}"
# keepalive + watchdog (kills the server below 15 GB available) + sampler, all tied to the keepalive
KA=$(rsh "nohup sleep 20000 > /dev/null 2>&1 < /dev/null & echo \$!")
rsh "rm -f $W/ipid-$TAG $CSV"
cat > "$OUT/.wd-$TAG.sh" <<'EOF'
#!/usr/bin/env bash
# usage: wd.sh <keepalive-pid> <ipidfile> <logfile>: poll MemAvailable every 20 s; below 15 GB kill -9 the process in <ipidfile>
ka=$1; pf=$2; lf=$3
while kill -0 "$ka" 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); echo "$(date +%T) $a" >> "$lf"
  if [ "$a" -lt 15 ]; then p=$(cat "$pf" 2>/dev/null); if [ -n "$p" ]; then ch=$(pgrep -P "$p"); kill -9 "$p" $ch 2>/dev/null; fi; echo "KILLED server sup=$p children=$ch avail=$a GB" >> "$lf"; fi
  sleep 20
done
EOF
scp -q -o BatchMode=yes "$OUT/.wd-$TAG.sh" "$HOST:intarsiaml-memgrow/out/wd.sh"; rm -f "$OUT/.wd-$TAG.sh"
WD=$(rsh "nohup bash $W/wd.sh $KA $W/ipid-$TAG $W/wd-$TAG.log > /dev/null 2>&1 < /dev/null & echo \$!")
SAMP=$(rsh "IPIDFILE=$W/ipid-$TAG nohup bash $W/sampler.sh $KA $PORT $CSV 30 > /dev/null 2>&1 < /dev/null & echo \$!")
mark base; sleep 90

mark start
SPID=$(rsh "env ${INTARSIA_ENV:-} nohup $BIN serve $SERVE_ARGS > $W/serve-$TAG.log 2>&1 < /dev/null & echo \$!")
rsh "echo $SPID > $W/ipid-$TAG"
log "server pid $SPID"
up=0; for i in $(seq 1 200); do curl -sf "http://$HN:$PORT/health" > /dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
[[ $up == 1 ]] || { log "server did not come up"; rsh "tail -20 $W/serve-$TAG.log" | tee -a "$WLOG"; exit 1; }
log "server up, avail $(avail) GB"
mark settle; sleep 120

mark load; st=$(date +%s); end=$((st + SOAKMIN*60)); fails=0; nrep=0
log "load: no new replay after $(date -r $end +%T)"
while (( $(date +%s) < end )); do
  ( cd /opt/code/deepeval && perl -e 'alarm 1500; exec @ARGV' uv run python "$REPO/scripts/soak2/replay2.py" --url "$UB" --model guidance --arm "S$seed" --order interleaved --seed "$seed" --streams 2 \
      --plan "$OUT/replay-plan.json" --trace "$OUT/replay-$TAG/trace-seed$seed.json" --out "$OUT/replay-$TAG" --temp-mode mixed ) > "$OUT/replay-$TAG/S$seed.out" 2>&1
  tail -1 "$OUT/replay-$TAG/S$seed.out" | cut -c1-160
  if [[ ! -s $OUT/replay-$TAG/S$seed-interleaved-seed$seed-k2.summary.json ]]; then fails=$((fails+1)); log "replay S$seed produced no summary ($fails)"; (( fails >= 2 )) && { log "ABORT load: two failed replays"; break; }; else fails=0; nrep=$((nrep+1)); fi
  seed=$((seed+1)); rsh "kill -0 $SPID" 2>/dev/null || { log "SERVER GONE"; break; }
done
log "load done: $nrep replays in $(( ($(date +%s)-st)/60 )) min"
mark idle; sleep $((IDLEMIN*60))
mark stop
stop_server
rsh "rm -f $W/ipid-$TAG"; sleep 150; mark after
log "server stopped, avail $(avail) GB"
rsh "cat $CSV" > "$OUT/mem-$TAG.csv"; rsh "cat $W/serve-$TAG.log" > "$OUT/serve-$TAG.log"; rsh "cat $W/wd-$TAG.log" > "$OUT/wd-$TAG.log" 2>/dev/null
log "warn/error lines in serve log: $(grep -ciE 'warn|error|panic|fail' "$OUT/serve-$TAG.log"); watchdog kills: $(grep -c KILLED "$OUT/wd-$TAG.log" 2>/dev/null); min avail GB (time GB): $(sort -k2 -n "$OUT/wd-$TAG.log" | head -1)"
# keep only the summaries of the replays
stop_all
rm -f "$OUT/replay-$TAG"/*.jsonl "$OUT/replay-$TAG"/trace-seed*.json
