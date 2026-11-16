#!/bin/bash
# Lane mhkv: lane mstore's restore window (scripts/mstore/window.sh), main against the fix, MTP on, fresh server per run.
# usage (inside a fleetq slot): scripts/mhkv/window.sh 35b|27b [ctx=4k] [cycles=1]
#   bins (pinned under target/mhkv-run/): M = main 10cb85061, F = this branch (`*.rev` beside each says what they are).
#   setup per bin (the store key holds the executable, so each bin writes its own): a fresh server with an empty store dir
#   D0-<bin> runs P (the shared prefix), waits for the disk writer, stops.
#   per bin visit, two fresh servers: B0 = store on, empty dir (cold with the store: Q); B = store on with a copy of
#   D0-<bin> (after a restart: Q restores P from disk / page cache; then U; then Q again, all but its last tokens from host).
#   visits ABBA per cycle: M F F M (inner order B0 B, then B B0 on the second visit); 90 s idle before every server.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mhkv; RUN=${MHKV_RUN:-$W/target/mhkv-run}; PR=$RUN/prompts; RAW=${MHKV_RAW:-$W/docs/results/2026-11-16-mhkv/raw}
case $1 in
  35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4 ;;
  27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4 ;;
  *) echo "model?"; exit 2 ;;
esac
CTX=${2:-4k}; CYC=${3:-1}; TAG=$1-$CTX; COOL=${MHKV_COOL:-90}
for b in main fix; do [ -x $RUN/intarsia-$b ] || { echo "no pinned binary $RUN/intarsia-$b" >&2; exit 3; }; done
[ -f $PR/Q-$CTX.json ] || python3 $W/scripts/mhkv/prompts.py $PR > /dev/null
R=$RAW/win-$TAG; mkdir -p $R
S=$RUN/store-$TAG; rm -rf $S; mkdir -p $S
PORT=18641; URL=http://127.0.0.1:$PORT
STORE="host_mb=8192,disk_mb=65536"
export INTARSIA_MTP_STORE=1
( while sleep 10; do echo "$(date +%T) $(memory_pressure -Q 2>/dev/null | tail -1) $(sysctl -n vm.swapusage)"; done ) > $R/mem.log 2>&1 &
MW=$!
trap 'kill $MW 2>/dev/null' EXIT

start() { # bin name -- extra serve args
  local b=$1 n=$2; shift 2
  echo "$(date +%T) cooldown ${COOL} s before $n"; sleep $COOL
  $RUN/intarsia-$b serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $R/$n.server.out 2> $R/$n.server.err < /dev/null &
  SP=$!
  local t0=$SECONDS
  for i in {1..300}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "SERVER DIED $n"; tail -5 $R/$n.server.err; return 1; }; sleep 1; done
  echo "## $n ($b) up in $((SECONDS - t0)) s: $*"
}
stop() {
  kill -INT $SP; for i in {1..60}; do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
}
req() { # tag file
  python3 $W/scripts/mhkv/req.py $URL $PR/$2 $1 256 | tee -a $R/requests.jsonl
}
wait_disk() { # until the disk writer has nothing pending
  local p=""
  for i in {1..240}; do
    p=$(curl -sf $URL/v1/intarsia/prefix-store | python3 -c 'import json,sys; d=json.load(sys.stdin); s=d.get("stats",d); print(s.get("disk_pending_bytes",0), (s.get("disk") or {}).get("bytes",0))' 2>/dev/null)
    [ "${p%% *}" = 0 ] && [ "${p##* }" != 0 ] && { echo "disk settled: pending/bytes $p after $((i/4)) s"; return; }
    sleep 0.25
  done
  echo "disk NOT settled: $p"
}

echo "## $TAG $(date +%T) main $(cat $RUN/intarsia-main.rev) fix $(cat $RUN/intarsia-fix.rev) model $M cycles $CYC"
for b in main fix; do
  start $b setup-$b --prefix-store "$STORE,dir=$S/D0-$b" || exit 1
  req setup-$b-P P-$CTX.json
  wait_disk
  curl -sf $URL/v1/intarsia/prefix-store > $R/setup-$b.store.json
  stop
  echo "D0-$b on disk: $(du -sk $S/D0-$b | cut -f1) KiB, $(find $S/D0-$b -name '*.bin' | wc -l | tr -d ' ') objects, $(find $S/D0-$b -name '*.entry' | wc -l | tr -d ' ') entries"
done

run_B0() { # bin visit
  rm -rf $S/E; start $1 B0-$1-$2 --prefix-store "$STORE,dir=$S/E" || return
  req B0-$1-$2-Q Q-$CTX.json; stop
}
run_B() { # bin visit
  rm -rf $S/D; cp -R $S/D0-$1 $S/D; start $1 B-$1-$2 --prefix-store "$STORE,dir=$S/D" || return
  req B-$1-$2-Q-disk Q-$CTX.json; req B-$1-$2-U U.json; req B-$1-$2-Q-repeat Q-$CTX.json
  curl -sf $URL/v1/intarsia/prefix-store > $R/B-$1-$2.store.json; stop
}
v=0
for c in $(seq 1 $CYC); do
  for b in main fix fix main; do
    v=$((v + 1))
    if (( v % 2 )); then run_B0 $b $v; run_B $b $v; else run_B $b $v; run_B0 $b $v; fi
  done
done
echo "## done $TAG $(date +%T)"
