#!/usr/bin/env bash
# Lane mt2 (docs/results/2026-11-07-mtp-moe-tp.md): 35B `--spec mtp` under tp=2 across the Sparks (rank 0 = spark1 `serve --tp-remote`,
# rank 1 = spark2 `intarsia tp-worker`, NCCL over CX-7 port f1), against tp=1 on spark1. vLLM stays UP on both boxes. Derived from
# scripts/tp2r/xg.sh (window mode removed) and scripts/spark35num/{idgate,speed}.sh. Runs HERE (the Mac), the one command of ONE fleetq job
# holding both Sparks:
#   STEPS=ids fleetq run --host spark1,spark2 --mem spark1=45,spark2=30 --load gpu --expect 40 --label "..." -- scripts/mt2/xg.sh
#   STEPS=speed fleetq run --host spark1,spark2 --mem spark1=45,spark2=30 --quiet gpu,cpu --expect 60 --label "..." -- scripts/mt2/xg.sh
# Steps (STEPS picks): ids    256 greedy ids x 6 prompts (scripts/fp8kv/greedy.py): dev tp2 off, dev tp2 mtp k2, base tp2 off; compares
#                     speed  localeval speed 1k/4k/16k gen 128 x 3 reps, arms ARMS (default "tp1m tp2off tp2m"), ROUNDS (default 2, even reversed)
# A memory watcher on each box kills this lane's processes if MemAvailable drops below 10 GB. Every process started here is killed on exit.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
OUT=${OUT:-$(date +%H%M)}
L=$REPO/docs/results/2026-11-07-mtp-moe-tp/raw/$OUT; mkdir -p "$L"
P='$HOME/intarsia-mt2'
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=${WPORT:-29561}; PORT=${PORT:-8297}; HN=spark-4e77
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8} NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576}"
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
XO="$P/out/$OUT"
kill_lane() { for h in spark1 spark2; do r $h "touch $P/stop-worker; pkill -TERM -f '[i]ntarsia-mt2/bin/'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-mt2/bin/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-mt2/bin/'; true"; done; }
cleanup() {
  local rc=$?
  log "cleanup (exit $rc)"
  kill_lane
  for h in spark1 spark2; do r $h "pkill -f '[i]ntarsia-mt2/memwatch' ; true"; done
  rsync -a "spark1:intarsia-mt2/out/$OUT/" "$L/spark1/" 2>/dev/null; rsync -a "spark2:intarsia-mt2/out/$OUT/" "$L/spark2/" 2>/dev/null
  trap - EXIT INT TERM; exit $rc
}
trap cleanup EXIT INT TERM
for h in spark1 spark2; do
  r $h "mkdir -p $XO; rm -f $P/stop-worker; echo $h: \$(strings $P/bin/dev-intarsia | grep -m1 -o 'head=[0-9a-f]* code-tree=[0-9a-f]* dirty=[a-z]*') base: \$(strings $P/bin/base-intarsia | grep -m1 -o 'head=[0-9a-f]* code-tree=[0-9a-f]* dirty=[a-z]*'); free -g | sed -n 2p" | tee -a "$L/steps.log"
  # memory watcher: below 10 GB available, kill this lane's processes at once (a GB10 swaps instead of OOM-killing)
  scp -q "$S/memwatch.sh" $h:intarsia-mt2/memwatch.sh && r $h "nohup bash $P/memwatch.sh $XO > /dev/null 2>&1 < /dev/null &"
done
sleep 1; for h in spark1 spark2; do r $h "pgrep -f '[i]ntarsia-mt2/memwatch' >/dev/null && echo $h memwatch up || echo $h MEMWATCH MISSING" | tee -a "$L/steps.log"; done

# rank 1: a loop of tp-workers (one per rank-0 connection) until stop-worker
worker_start() { # <tag> <bin> <flags...>
  local tag=$1 bin=$2; shift 2
  r spark2 "rm -f $P/stop-worker; nohup bash -c 'i=0; while [ ! -f $P/stop-worker ]; do i=\$((i+1)); env $NCCLENV ${WENV:-} $P/bin/$bin-intarsia tp-worker --listen $WIP:$WPORT $* >> $XO/worker-$tag.log 2>&1; echo \"worker exit \$? (\$i)\" >> $XO/worker-$tag.log; sleep 1; done' > /dev/null 2>&1 < /dev/null &"
  sleep 1
}
worker_stop() { r spark2 "touch $P/stop-worker; pkill -TERM -f '[t]p-worker --listen $WIP:$WPORT'; for i in \$(seq 1 30); do pgrep -f '[t]p-worker --listen $WIP:$WPORT' >/dev/null || break; sleep 1; done; pkill -KILL -f '[t]p-worker --listen $WIP:$WPORT'; true"; }
SPID=""
server_stop() { [[ -n $SPID ]] && r spark1 "kill $SPID 2>/dev/null; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null"; SPID=""; worker_stop; sleep 2; }
# server_start <tag> <bin> <tp: 1|2> <serve flags...> ; tp2 starts the worker with the same model flags
server_start() {
  local tag=$1 bin=$2 tp=$3; shift 3
  local mf="-m $M35 --backend cuda --max-seq 20480 --profile-every 0"
  if [[ $tp == 2 ]]; then
    worker_start $tag $bin --serving $mf --tp 2 --mem-budget 24 "$@"
    SPID=$(r spark1 "nohup env $NCCLENV $P/bin/$bin-intarsia serve $mf --tp 2 --tp-remote $WIP:$WPORT --mem-budget 24 --port $PORT --host 0.0.0.0 --served-model-name intarsia $* > $XO/serve-$tag.log 2>&1 < /dev/null & echo \$!")
  else
    SPID=$(r spark1 "nohup $P/bin/$bin-intarsia serve $mf --mem-budget 40 --port $PORT --host 0.0.0.0 --served-model-name intarsia $* > $XO/serve-$tag.log 2>&1 < /dev/null & echo \$!")
  fi
  local i
  for i in $(seq 1 150); do
    curl -sf -m 3 "http://$HN:$PORT/health" >/dev/null && {
      log "up $tag ($bin tp$tp $*): $(r spark1 "grep -aiE 'spec|mtp|drafter|verify|fusion' $XO/serve-$tag.log | head -6 | cut -c1-220 | tr '\n' '|'")"
      for h in spark1 spark2; do log "  $h $(r $h 'free -g | sed -n 2p')"; done
      return 0; }
    r spark1 "kill -0 $SPID 2>/dev/null" || break
    sleep 2
  done
  log "FAILED to start $tag"; r spark1 "tail -n 12 $XO/serve-$tag.log | cut -c1-300" | tee -a "$L/steps.log"
  [[ $tp == 2 ]] && r spark2 "tail -n 12 $XO/worker-$tag.log | cut -c1-300" | tee -a "$L/steps.log"
  server_stop; return 1
}
G=$REPO/scripts/fp8kv/greedy.py
if want ids; then
  for a in "tp2off dev --spec off" "tp2m dev --spec mtp --spec-k 2" "basetp2off base --spec off"; do
    set -- $a; tag=$1 bin=$2; shift 2
    if server_start ids-$tag $bin 2 "$@"; then
      python3 "$G" run --url "http://$HN:$PORT/v1" --model intarsia --gen 256 --out "$L/ids-$tag.json" > "$L/ids-$tag.log" 2>&1; log "ids $tag: $(tail -1 "$L/ids-$tag.log")"
      r spark1 "grep -aE 'spec|accept|draft' $XO/serve-ids-$tag.log | tail -3 | cut -c1-200" | tee -a "$L/steps.log"
      server_stop
    fi
  done
  cmp2() { if python3 "$G" compare "$L/ids-$1.json" "$L/ids-$2.json" > "$L/cmp-$1-vs-$2.txt" 2>&1; then log "PASS ids $1 == $2"; else log "FAIL ids $1 != $2"; fi; cat "$L/cmp-$1-vs-$2.txt" >> "$L/steps.log"; }
  cmp2 tp2off tp2m; cmp2 basetp2off tp2off
fi
if want speed; then
  le() { # le <label> <sizes> <gen>
    ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "http://$HN:$PORT/v1" --model intarsia --thinking off --sizes "$2" --gen "$3" --reps ${REPS:-3} \
        --label "$1" --no-unim --recipe "mt2 $1" ) > "$L/$1.out" 2>&1
    log "$1: $(grep -aE 'decode|prefill|tok/s' "$L/$1.out" | head -8 | tr '\n' '|' | cut -c1-600)"
  }
  arm() { # arm <round> <arm>
    local rd=$1 a=$2
    case $a in
      tp1m) server_start sp-$a-r$rd dev 1 --spec mtp --spec-k 2 ;;
      tp1off) server_start sp-$a-r$rd dev 1 --spec off ;;
      tp2off) server_start sp-$a-r$rd dev 2 --spec off ;;
      tp2m) server_start sp-$a-r$rd dev 2 --spec mtp --spec-k 2 ;;
      *) log "unknown arm $a"; return 1 ;;
    esac || return 0
    le "mt2-$a-r$rd" ${SIZES:-1024,4096,16384} ${GEN:-128}
    r spark1 "grep -aE 'spec|accept' $XO/serve-sp-$a-r$rd.log | tail -2 | cut -c1-200" | tee -a "$L/steps.log"
    server_stop
  }
  ARMS_FWD=${ARMS:-tp1m tp2off tp2m}
  for rd in $(seq 1 ${ROUNDS:-2}); do
    order=$ARMS_FWD; (( rd % 2 == 0 )) && order=$(echo $ARMS_FWD | tr ' ' '\n' | tail -r | tr '\n' ' ')
    for a in $order; do arm $rd $a; done
  done
fi
log "== done"
