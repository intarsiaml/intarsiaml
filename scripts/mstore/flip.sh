#!/bin/zsh
# Lane mstore: what the INTARSIA_MTP_STORE flip changes for a store user on Metal. Both arms: store on, a fresh server per run
# restoring the same D0 (P stored), then Q (new turn, from disk), U, Q again (from host). M = INTARSIA_MTP_STORE=1 (MTP drafts
# after the restore); P = INTARSIA_MTP_STORE=0 (today's Metal default: a store skips the MTP default, plain decode). Order M P P M.
# usage (inside a fleetq slot): scripts/mstore/flip.sh 35b|27b 4k|32k
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mstore; BIN=$W/target/mstore-run/intarsia-measure; PR=$W/target/mstore-run/prompts
case $1 in
  35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4 ;;
  27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4 ;;
  *) echo "model?"; exit 2 ;;
esac
CTX=$2; TAG=$1-$CTX
R=$W/docs/results/2026-11-15-mstore-metal/raw/flip-$TAG; mkdir -p $R
S=$W/target/mstore-run/store-$TAG
[ -d $S/D0 ] || { echo "no $S/D0 (run window.sh $1 $2 first)"; exit 2; }
PORT=18632; URL=http://127.0.0.1:$PORT
STORE="host_mb=8192,disk_mb=65536"
start() { local n=$1; shift
  env "$@" $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT --prefix-store "$STORE,dir=$S/D" > $R/$n.server.out 2> $R/$n.server.err < /dev/null &
  SP=$!
  for i in {1..300}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "SERVER DIED $n"; tail -5 $R/$n.server.err; return 1; }; sleep 1; done
  echo "## $n up: $*"; grep -a "spec\|prefix store" $R/$n.server.err | head -3 | cut -c1-160; }
stop() { kill -INT $SP; for i in {1..60}; do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null; }
req() { python3 $W/scripts/mstore/req.py $URL $PR/$2 $1 256 | tee -a $R/requests.jsonl; }
run() { local n=$1 v=$2; rm -rf $S/D; cp -R $S/D0 $S/D; start $n INTARSIA_MTP_STORE=$v || return
  req $n-Q-disk Q-$CTX.json; req $n-U U.json; req $n-Q-repeat Q-$CTX.json; stop; }
echo "## flip $TAG $(date +%T) bin $(cat $W/target/mstore-run/intarsia-measure.rev)"
run M1 1; run P1 0; run P2 0; run M2 1
echo "## done $(date +%T)"
