#!/usr/bin/env bash
# memmix: one mixed-traffic memory run of the shipping config (Qwen3.6-35B-A3B NVFP4, CUDA, --max-batch 2 --prefix-store on) on a Spark, with memgrow's 30 s sampler,
# a watchdog (kills the server below 15 GB MemAvailable) and malloc_trim(0) every TRIMEVERY minutes inside the engine (libmemstat.c: RssAnon before / after in the serve log, line "[memtrim]").
#   FLEETQ_OWNER=claude-overnight-memmix fleetq run --host spark1 --mem 52 --expect 190 --label "memmix soak" -- env HOST=spark1 bash scripts/memmix/run.sh <tag>
# Env: HOST (spark1|spark2), PROFILE (mixed|tiny), WEIGHTS (mixed.py --weights), SEED0 (default 7), SOAKMIN (load minutes, default 150), IDLEMIN (default 5),
#      TRIMEVERY (minutes between trims, 0 = none, default 15), MEMSTAT_PERIOD (malloc_info period s, default 300), SERVER_ENV (extra env for the server, e.g. "MALLOC_ARENA_MAX=2"), SERVE_EXTRA.
# Phases (marks in raw/marks-<tag>.tsv): base -> start -> settle -> load (+ trimN marks) -> idle -> stop -> after.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TAG=${1:?tag}; shift; EXTRA="${SERVE_EXTRA:-} $*"
HOST=${HOST:-spark1}; case $HOST in spark1) HN=spark-4e77;; spark2) HN=spark-fa00;; *) echo bad host; exit 1;; esac; PORT=8191
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-memmix/raw; mkdir -p "$OUT"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-memmix/bin/intarsia'; W='~/intarsiaml-memmix/out'
SERVE_ARGS="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on $EXTRA"
SOAKMIN=${SOAKMIN:-150}; IDLEMIN=${IDLEMIN:-5}; TRIMEVERY=${TRIMEVERY:-15}; PROFILE=${PROFILE:-mixed}; SEED=${SEED0:-7}
KA=""; SPID=""; WD=""; LOADPID=""; T0=$(date +%s); WLOG=$OUT/window-$TAG.log; MARKS=$OUT/marks-$TAG.tsv; CSV=$W/mem-$TAG.csv
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
mark() { printf '%s\t%s\n' "$(date +%s)" "$1" >> "$MARKS"; log "mark $1"; }
avail() { rsh "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"; }
stop_server() { # TERM the supervisor and its child by pid, KILL after 40 s
  [[ -n "$SPID" ]] || return 0
  rsh "ch=\$(pgrep -P $SPID); kill -TERM $SPID \$ch 2>/dev/null; for i in \$(seq 1 40); do n=0; for p in $SPID \$ch; do kill -0 \$p 2>/dev/null && n=1; done; [ \$n = 0 ] && exit 0; sleep 1; done; kill -9 $SPID \$ch 2>/dev/null; sleep 3"
  SPID=""; log "after stop, processes of ours left: [$(rsh "pgrep -f 'intarsiaml-memmix/bin/intarsia serve' | tr '\n' ' '")]"
}
stop_all() {
  [[ -n "$LOADPID" ]] && kill "$LOADPID" 2>/dev/null
  stop_server
  [[ -n "$KA" ]] && { rsh "kill $KA 2>/dev/null"; KA=""; }
  [[ -n "$WD" ]] && { rsh "kill $WD 2>/dev/null"; WD=""; }
}
trap 'stop_all; log "window end"' EXIT; trap 'exit 130' INT TERM
: > "$MARKS"
scp -q -o BatchMode=yes "$REPO/scripts/memmix/sampler.sh" "$REPO/scripts/memmix/libmemstat.c" "$HOST:intarsiaml-memmix/out/" || { log "copy failed"; exit 1; }
rsh "cd $W && gcc -O1 -shared -fPIC -o libmemstat.so libmemstat.c -lpthread" || { log "libmemstat build failed"; exit 1; }
rsh "ss -ltn | grep -q ':$PORT '" && { log "port $PORT busy, abort"; exit 1; }
SERVER_ENV="LD_PRELOAD=/home/nvidia/intarsiaml-memmix/out/libmemstat.so MEMSTAT_PERIOD=${MEMSTAT_PERIOD:-300} MEMSTAT_TRIGGER=/tmp/memstat-trigger-$TAG ${SERVER_ENV:-}"
log "tag $TAG host $HOST; fleetq job $FLEETQ_JOB; earlyoom $(rsh 'systemctl is-active earlyoom'); avail $(avail) GB; binary $(rsh "$BIN --version | head -1")"
log "serve args: $SERVE_ARGS; env: $SERVER_ENV; profile $PROFILE weights '${WEIGHTS:-}' seed $SEED load ${SOAKMIN} min trim every ${TRIMEVERY} min"
# keepalive (the sampler and watchdog live as long as it does) + watchdog + sampler
KA=$(rsh "nohup sleep 30000 > /dev/null 2>&1 < /dev/null & echo \$!")
rsh "rm -f $W/ipid-$TAG $CSV"
cat > "$OUT/.wd-$TAG.sh" <<'EOS'
#!/usr/bin/env bash
# usage: wd.sh <keepalive-pid> <ipidfile> <logfile>: poll MemAvailable every 20 s; below 15 GB kill -9 the process in <ipidfile> and its child
ka=$1; pf=$2; lf=$3
while kill -0 "$ka" 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); echo "$(date +%T) $a" >> "$lf"
  if [ "$a" -lt 15 ]; then p=$(cat "$pf" 2>/dev/null); if [ -n "$p" ]; then ch=$(pgrep -P "$p"); kill -9 "$p" $ch 2>/dev/null; fi; echo "KILLED server sup=$p children=$ch avail=$a GB" >> "$lf"; fi
  sleep 20
done
EOS
scp -q -o BatchMode=yes "$OUT/.wd-$TAG.sh" "$HOST:intarsiaml-memmix/out/wd.sh"; rm -f "$OUT/.wd-$TAG.sh"
WD=$(rsh "nohup bash $W/wd.sh $KA $W/ipid-$TAG $W/wd-$TAG.log > /dev/null 2>&1 < /dev/null & echo \$!")
SAMP=$(rsh "IPIDFILE=$W/ipid-$TAG nohup bash $W/sampler.sh $KA $PORT $CSV 30 > /dev/null 2>&1 < /dev/null & echo \$!")
mark base; sleep 90

mark start
SPID=$(rsh "env $SERVER_ENV nohup $BIN serve $SERVE_ARGS > $W/serve-$TAG.log 2>&1 < /dev/null & echo \$!")
rsh "echo $SPID > $W/ipid-$TAG"
log "server pid $SPID"
up=0; for i in $(seq 1 200); do curl -sf "http://$HN:$PORT/health" > /dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
[[ $up == 1 ]] || { log "server did not come up"; rsh "tail -20 $W/serve-$TAG.log" | tee -a "$WLOG"; exit 1; }
log "server up, avail $(avail) GB"
mark settle; sleep 120

mark load; st=$(date +%s)
mkdir -p "$OUT/mixed-$TAG"
WARG=(); [[ -n "${WEIGHTS:-}" ]] && WARG=(--weights "$WEIGHTS")
( cd /opt/code/deepeval && uv run python "$REPO/scripts/memmix/mixed.py" --url "http://$HN:$PORT" --model guidance --minutes "$SOAKMIN" --streams 2 --seed "$SEED" --profile "$PROFILE" ${WARG[@]+"${WARG[@]}"} --out "$OUT/mixed-$TAG" ) > "$OUT/mixed-$TAG.out" 2>&1 &
LOADPID=$!
log "$PROFILE load for $SOAKMIN min (pid $LOADPID)"
n=0; next=$(( st + TRIMEVERY*60 ))
while kill -0 "$LOADPID" 2>/dev/null; do
  sleep 5
  if [[ $TRIMEVERY -gt 0 && $(date +%s) -ge $next && $(( $(date +%s) - st )) -lt $(( SOAKMIN*60 - 60 )) ]]; then
    n=$((n+1)); next=$(( next + TRIMEVERY*60 ))
    ENG=$(rsh "pgrep -P $SPID | head -1"); rsh "touch /tmp/memstat-trigger-$TAG.$ENG"; mark trim$n
  fi
done
wait "$LOADPID"; LOADPID=""
tail -1 "$OUT/mixed-$TAG.out" | cut -c1-700 | tee -a "$WLOG"
log "load done after $(( ($(date +%s)-st)/60 )) min; $n trims"
# one final trim at the end of the load (before / after in the log), then idle
if [[ $TRIMEVERY -gt 0 ]]; then ENG=$(rsh "pgrep -P $SPID | head -1"); rsh "touch /tmp/memstat-trigger-$TAG.$ENG"; mark trimend; sleep 40; fi
mark idle; sleep $((IDLEMIN*60))
mark stop
stop_server
rsh "rm -f $W/ipid-$TAG"; sleep 120; mark after
log "server stopped, avail $(avail) GB"
rsh "cat $CSV" > "$OUT/mem-$TAG.csv"; rsh "cat $W/wd-$TAG.log" > "$OUT/wd-$TAG.log" 2>/dev/null
rsh "cat $W/serve-$TAG.log" > "$OUT/serve-$TAG.log"
grep -E '^\[memtrim\]' "$OUT/serve-$TAG.log" > "$OUT/memtrim-$TAG.txt" 2>/dev/null
log "warn/error lines in serve log: $(grep -ciE 'warn|error|panic|fail' "$OUT/serve-$TAG.log"); watchdog kills: $(grep -c KILLED "$OUT/wd-$TAG.log" 2>/dev/null); min avail GB (time GB): $(sort -k2 -n "$OUT/wd-$TAG.log" | head -1)"
stop_all
