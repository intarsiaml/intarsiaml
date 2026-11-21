#!/usr/bin/env bash
# memlong: long (hours) mixed-traffic memory soak of the shipping config on a Spark, with NO malloc_trim anywhere. One fleetq job, no babysitting.
#   fleetq run --host spark1 --expect 690 --mem 52 --label "memlong 11h untrimmed mixed soak" -- bash scripts/memlong/run.sh 39600 memlong
# usage: run.sh [DURATION-s=39600] [TAG=memlong]      Env: HOST (spark1), BIN (box path of a prebuilt binary, default ~/intarsiaml-memlong/bin/intarsia),
#   SEED0 (7), SEGMIN (minutes per mixed.py segment, each with a fresh seed; default 60), SAMPLE (s, 60), SERVE_EXTRA, MIN_AVAIL (GB, 15)
# Server env: only LD_PRELOAD=libmemstat.so (malloc_info every MEMSTAT_PERIOD=60 s, trigger file never set, so no trim) + MEMSTAT_PERIOD.
# Box side: work dir ~/intarsiaml-memlong-run-<TAG> (removed at the end, also on SIGTERM); the binary dir ~/intarsiaml-memlong stays.
# Dead-man on the box: the watchdog kills the server at MemAvailable < MIN_AVAIL or at DURATION + 40 min even if this script is gone.
# Results: docs/results/2026-11-20-memlong/raw/ (mem-<tag>.csv, marks-<tag>.tsv, serve-<tag>.log, wd-<tag>.log, window-<tag>.log, stop-reason-<tag>.txt, mixed-<tag>/).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
DURATION=${1:-39600}; TAG=${2:-memlong}
HOST=${HOST:-spark1}; case $HOST in spark1) HN=spark-4e77;; spark2) HN=spark-fa00;; *) echo bad host; exit 1;; esac; PORT=8191
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
MM=$REPO/scripts/memmix
OUT=$REPO/docs/results/2026-11-20-memlong/raw; mkdir -p "$OUT"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN=${BIN:-/home/nvidia/intarsiaml-memlong/bin/intarsia}; W=/home/nvidia/intarsiaml-memlong-run-$TAG
SEED=${SEED0:-7}; SEGMIN=${SEGMIN:-60}; SAMPLE=${SAMPLE:-60}; MIN_AVAIL=${MIN_AVAIL:-15}
SERVE_ARGS="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on ${SERVE_EXTRA:-}"
SERVER_ENV="LD_PRELOAD=$W/libmemstat.so MEMSTAT_PERIOD=60"
KA=""; SPID=""; LOADPID=""; REASON=""; DONE=0; T0=$(date +%s); WLOG=$OUT/window-$TAG.log; MARKS=$OUT/marks-$TAG.tsv
rsh() { ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
mark() { printf '%s\t%s\n' "$(date +%s)" "$1" >> "$MARKS"; log "mark $1"; }
avail() { rsh "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"; }
stop_server() { # TERM the supervisor and its child by pid, KILL after 40 s
  [[ -n "$SPID" ]] || return 0
  rsh "ch=\$(pgrep -P $SPID); kill -TERM $SPID \$ch 2>/dev/null; for i in \$(seq 1 40); do n=0; for p in $SPID \$ch; do kill -0 \$p 2>/dev/null && n=1; done; [ \$n = 0 ] && exit 0; sleep 1; done; kill -9 $SPID \$ch 2>/dev/null; sleep 3"
  SPID=""; log "after stop, serve processes of ours left: [$(rsh "pgrep -f '[${BIN:0:1}]${BIN:1} serve' | tr '\n' ' '")]"
}
collect() { # copy the box files back (cheap, repeatable)
  rsh "cat $W/mem-$TAG.csv" > "$OUT/mem-$TAG.csv.new" 2>/dev/null && mv "$OUT/mem-$TAG.csv.new" "$OUT/mem-$TAG.csv"
  rsh "cat $W/wd-$TAG.log" > "$OUT/wd-$TAG.log" 2>/dev/null
  rsh "cat $W/serve-$TAG.log" > "$OUT/serve-$TAG.log" 2>/dev/null
}
finish() { # idempotent: stop everything, copy back, remove the box work dir, print the summary
  [[ $DONE == 1 ]] && return; DONE=1; trap '' TERM INT
  [[ -z "$REASON" ]] && REASON="exit"
  echo "$REASON" > "$OUT/stop-reason-$TAG.txt"
  [[ -n "$LOADPID" ]] && { kill "$LOADPID" 2>/dev/null; pkill -P "$LOADPID" 2>/dev/null; pkill -f "[m]ixed.py --url http://$HN:$PORT" 2>/dev/null; wait "$LOADPID" 2>/dev/null; LOADPID=""; }
  mark stop; stop_server
  collect
  [[ -n "$KA" ]] && { rsh "kill $KA 2>/dev/null"; KA=""; }
  sleep 3; rsh "pkill -f '[i]ntarsiaml-memlong-run-$TAG/' 2>/dev/null; rm -rf $W"; log "box dir removed: $(rsh "ls -d $W 2>&1 | head -1")"
  rsh "rm -f /tmp/memstat-trigger-$TAG*" 2>/dev/null
  python3 "$REPO/scripts/memlong/analyze.py" "$OUT" "$TAG" --oneline "$REASON" | tee -a "$WLOG"
}
trap 'finish' EXIT; trap 'REASON="sigterm"; exit 143' TERM; trap 'REASON="sigint"; exit 130' INT
nap() { sleep "$1" & wait $!; }   # interruptible by TERM
: > "$MARKS"
rsh "ss -ltn | grep -q ':$PORT '" && { REASON="port busy"; log "port $PORT busy, abort"; exit 1; }
rsh "test -x $BIN" || { REASON="no binary"; log "no binary at $BIN"; exit 1; }
rsh "rm -rf $W; mkdir -p $W" && scp -q -o BatchMode=yes "$MM/sampler.sh" "$MM/libmemstat.c" "$HOST:$W/" || { REASON="copy failed"; log "copy failed"; exit 1; }
rsh "cd $W && gcc -O1 -shared -fPIC -o libmemstat.so libmemstat.c -lpthread" || { REASON="libmemstat build failed"; log "libmemstat build failed"; exit 1; }
log "tag $TAG host $HOST; fleetq job $FLEETQ_JOB; earlyoom $(rsh 'systemctl is-active earlyoom'); avail $(avail) GB; binary $(rsh "$BIN --version | head -1")"
log "serve args: $SERVE_ARGS; env: $SERVER_ENV; no trim; duration ${DURATION}s; seeds from $SEED in ${SEGMIN}-minute segments; sampler ${SAMPLE}s"
DEADLINE=$(( $(date +%s) + DURATION + 3000 ))
KA=$(rsh "nohup sleep $(( DURATION + 4000 )) > /dev/null 2>&1 < /dev/null & echo \$!")
cat > "$OUT/.wd-$TAG.sh" <<'EOS'
#!/usr/bin/env bash
# usage: wd.sh <keepalive-pid> <ipidfile> <logfile> <deadline-epoch> <min-avail-GB>: every 20 s log MemAvailable; kill -9 the server (pid in <ipidfile>, and its child) below the limit or after the deadline
ka=$1; pf=$2; lf=$3; dl=$4; lim=$5
while kill -0 "$ka" 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); echo "$(date +%T) $a" >> "$lf"
  if [ "$a" -lt "$lim" ] || [ "$(date +%s)" -gt "$dl" ]; then p=$(cat "$pf" 2>/dev/null); if [ -n "$p" ]; then ch=$(pgrep -P "$p"); kill -9 "$p" $ch 2>/dev/null; echo "KILLED server sup=$p children=$ch avail=$a GB" >> "$lf"; fi; fi
  sleep 20
done
EOS
scp -q -o BatchMode=yes "$OUT/.wd-$TAG.sh" "$HOST:$W/wd.sh"; rm -f "$OUT/.wd-$TAG.sh"
rsh "WD=\$(nohup bash $W/wd.sh $KA $W/ipid $W/wd-$TAG.log $DEADLINE $MIN_AVAIL > /dev/null 2>&1 < /dev/null & echo \$!); IPIDFILE=$W/ipid nohup bash $W/sampler.sh $KA $PORT $W/mem-$TAG.csv $SAMPLE > /dev/null 2>&1 < /dev/null &"
mark base; nap 60

mark start
SPID=$(rsh "env $SERVER_ENV nohup $BIN serve $SERVE_ARGS > $W/serve-$TAG.log 2>&1 < /dev/null & echo \$!")
rsh "echo $SPID > $W/ipid"; log "server pid $SPID"
up=0; for i in $(seq 1 200); do curl -sf "http://$HN:$PORT/health" > /dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; nap 2; done
[[ $up == 1 ]] || { REASON="server did not come up"; log "$REASON"; rsh "tail -20 $W/serve-$TAG.log" | tee -a "$WLOG"; exit 1; }
log "server up, avail $(avail) GB"
mark settle; nap 120

mark load; st=$(date +%s); end=$(( st + DURATION )); seg=0; fails=0; ticks=0
mkdir -p "$OUT/mixed-$TAG"
while [[ -z "$REASON" ]]; do
  now=$(date +%s)
  if [[ -z "$LOADPID" ]]; then
    if (( now >= end )); then REASON="completed ${DURATION}s"; break; fi
    left=$(( (end - now + 59) / 60 )); m=$(( left < SEGMIN ? left : SEGMIN ))
    ( cd /opt/code/deepeval && exec uv run python "$MM/mixed.py" --url "http://$HN:$PORT" --model guidance --minutes "$m" --streams 2 --seed "$SEED" --profile mixed --out "$OUT/mixed-$TAG" ) > "$OUT/mixed-$TAG/seg-$SEED.out" 2>&1 &
    LOADPID=$!; log "segment $seg: seed $SEED, $m min (pid $LOADPID)"; seg=$((seg+1)); SEED=$((SEED+1))
  elif ! kill -0 "$LOADPID" 2>/dev/null; then
    wait "$LOADPID"; LOADPID=""; log "segment done: $(tail -1 "$OUT/mixed-$TAG/seg-$((SEED-1)).out" | cut -c1-300)"; continue
  fi
  nap 60
  chk=$(rsh "a=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo); kill -0 $SPID 2>/dev/null && s=1 || s=0; echo \$a \$s")
  if [[ -z "$chk" ]]; then fails=$((fails+1)); log "ssh check failed ($fails)"; (( fails >= 10 )) && REASON="ssh to $HOST lost for 10 min"; continue; fi
  fails=0; read -r a s <<< "$chk"
  (( s == 0 )) && { REASON="server died"; break; }
  (( a < MIN_AVAIL )) && { REASON="MemAvailable ${a} GB < ${MIN_AVAIL} GB"; break; }
  ticks=$((ticks+1)); (( ticks % 30 == 0 )) && collect
done
log "stop: $REASON"
exit 0
