#!/bin/zsh
# Lane mstore: TTFT and decode, cold vs restored from the prefix store, MTP on, fresh server per run.
# usage (inside a fleetq slot): scripts/mstore/window.sh 35b|27b 4k|32k [reps]
#   setup: a fresh server with an empty store dir D0 runs P (the shared prefix), waits for the disk writer, stops.
#   runs, order A B0 B B B0 A (x reps/2): A = store off (today's default, MTP on), B0 = store on with an empty dir (cold with the
#   store), B = store on with a copy of D0 (after a restart: Q restores P from disk / page cache; then U; then Q again, which
#   restores all but its last tokens from the host tier).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
# the CLI with the Metal backend (a no-op once built; a plain `-p intarsia-cli` build has no Metal backend)
# pinned: the first job builds it at the branch head of that moment, every later job measures that same binary
PIN=/opt/code/intarsiaml/.worktrees/mstore/target/mstore-run/intarsia-measure
if [ ! -x $PIN ]; then (cd /opt/code/intarsiaml/.worktrees/mstore && cargo build --release --locked -q -p intarsia-cli --features metal && cp target/release/intarsia $PIN.tmp && mv $PIN.tmp $PIN && git rev-parse --short HEAD > $PIN.rev && git status --short crates > $PIN.dirty) || { echo "build failed" >&2; exit 3; }; fi
W=/opt/code/intarsiaml/.worktrees/mstore; BIN=/opt/code/intarsiaml/.worktrees/mstore/target/mstore-run/intarsia-measure; PR=$W/target/mstore-run/prompts
case $1 in
  35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4 ;;
  27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4 ;;
  *) echo "model?"; exit 2 ;;
esac
CTX=$2; REPS=${3:-1}; TAG=$1-$CTX
R=$W/docs/results/2026-11-15-mstore-metal/raw/win-$TAG; mkdir -p $R
S=$W/target/mstore-run/store-$TAG; rm -rf $S; mkdir -p $S
PORT=18631; URL=http://127.0.0.1:$PORT
STORE="host_mb=8192,disk_mb=65536"
export INTARSIA_MTP_STORE=1
( while sleep 10; do echo "$(date +%T) $(memory_pressure -Q 2>/dev/null | tail -1) $(sysctl -n vm.swapusage)"; done ) > $R/mem.log 2>&1 &
MW=$!
trap 'kill $MW 2>/dev/null' EXIT

start() { # name -- extra serve args
  local n=$1; shift
  $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $R/$n.server.out 2> $R/$n.server.err < /dev/null &
  SP=$!
  local t0=$SECONDS
  for i in {1..300}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "SERVER DIED $n"; tail -5 $R/$n.server.err; return 1; }; sleep 1; done
  echo "## $n up in $((SECONDS - t0)) s: $*"
}
stop() {
  kill -INT $SP; for i in {1..60}; do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
}
req() { # tag file
  python3 $W/scripts/mstore/req.py $URL $PR/$2 $1 256 | tee -a $R/requests.jsonl
}
wait_disk() { # until the disk writer has nothing pending
  for i in {1..120}; do
    local p=$(curl -sf $URL/v1/intarsia/prefix-store | python3 -c 'import json,sys; d=json.load(sys.stdin); s=d.get("stats",d); print(s.get("disk_pending_bytes",0), (s.get("disk") or {}).get("bytes",0))' 2>/dev/null)
    [ "${p%% *}" = 0 ] && [ "${p##* }" != 0 ] && { echo "disk settled: pending/bytes $p after $((i/4)) s"; return; }
    sleep 0.25
  done
  echo "disk NOT settled: $p"
}

echo "## $TAG $(date +%T) bin $(cat $PIN.rev) model $M"
# setup: P into D0
start setup --prefix-store "$STORE,dir=$S/D0" || exit 1
req setup-P P-$CTX.json
wait_disk
curl -sf $URL/v1/intarsia/prefix-store > $R/setup.store.json
stop
echo "D0 on disk: $(du -sk $S/D0 | cut -f1) KiB, $(find $S/D0 -name '*.bin' | wc -l | tr -d ' ') objects, $(find $S/D0 -name '*.entry' | wc -l | tr -d ' ') entries"

# shutdown check: stop right after the reply (no wait for the disk writer); does a restart still restore P?
rm -rf $S/F; start F1 --prefix-store "$STORE,dir=$S/F" || exit 1
req F1-P P-$CTX.json; stop
echo "F after an immediate stop: $(du -sk $S/F | cut -f1) KiB, $(find $S/F -name '*.bin' | wc -l | tr -d ' ') objects, $(find $S/F -name '*.entry' | wc -l | tr -d ' ') entries"
start F2 --prefix-store "$STORE,dir=$S/F" || exit 1
req F2-P P-$CTX.json; stop

run_A()  { start A$1 || return; req A$1-Q Q-$CTX.json; stop; }
run_B0() { rm -rf $S/E; start B0$1 --prefix-store "$STORE,dir=$S/E" || return; req B0$1-Q Q-$CTX.json; stop; }
run_B()  { rm -rf $S/D; cp -R $S/D0 $S/D; start B$1 --prefix-store "$STORE,dir=$S/D" || return
           req B$1-Q-disk Q-$CTX.json; req B$1-U U.json; req B$1-Q-repeat Q-$CTX.json
           curl -sf $URL/v1/intarsia/prefix-store > $R/B$1.store.json; stop; }
for r in $(seq 1 $REPS); do
  if (( r % 2 )); then run_A $r; run_B0 $r; run_B $r; else run_B $r; run_B0 $r; run_A $r; fi
done
echo "## done $TAG $(date +%T)"
