#!/bin/zsh
# M2 closeout: concurrent-identity gate (tools/cbidentity.py, 203 cases, B=8, stream mixed, interleaving on: busy chunk 256),
# reuse off and on, one exclusive endor slot. References: solo servers (--max-batch 1) at --prefill-chunk 2048 and 256; each
# batched request is compared with the reference whose chunk equals the prefill_chunk the batched server reports for it.
# The batched servers run with INTARSIA_CB_TRACE=1 (stderr) so tools/cbfifo.py can check FIFO on the same traffic.
#   usage: slot_identity.sh OUTDIR        env: BIN (required), PORT
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=${1:?outdir}; mkdir -p $O; O=${O:A}
BIN=${BIN:?}; PORT=${PORT:-8142}
W=/opt/code/intarsiaml-relcheck; CASES=$W/tools/prompts/cb_identity_cases.jsonl
M=$HOME/models/Qwen3-8B; URL=http://127.0.0.1:$PORT
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
CB=(uv run $W/tools/cbidentity.py)
SPID=""
start_server() {  # name, then serve args; env set by the caller
  local name=$1; shift
  $BIN serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@" > $O/server-$name.log 2>&1 &
  SPID=$!
  for i in {1..180}; do
    curl -sf $URL/health >/dev/null 2>&1 && { echo "server $name pid $SPID ready"; curl -s $URL/v1/intarsia/plan > $O/plan-$name.json; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "server $name died"; tail -5 $O/server-$name.log; return 1; }
    sleep 1
  done
  echo "server $name not ready"; return 1
}
stop_server() { if [ -n "$SPID" ] && kill -0 $SPID 2>/dev/null; then kill $SPID 2>/dev/null; for i in {1..20}; do kill -0 $SPID 2>/dev/null || break; sleep 0.5; done; kill -9 $SPID 2>/dev/null; fi; SPID=""; }
trap stop_server EXIT
trap "exit 143" INT TERM
echo "slot $FLEETQ_JOB $(date); binary $BIN $(shasum -a 256 $BIN | cut -c1-16); git $(git -C $W rev-parse --short HEAD); cases $(wc -l < $CASES) lines"
ref() { # chunk
  start_server ref$1 --max-batch 1 --prefill-chunk $1 || return 1
  "${CB[@]}" run --mode solo --url $URL --cases $CASES --arrival-scale 0.25 --token-ids on --out $O/ref$1.jsonl > $O/ref$1.log 2>&1; echo "ref $1: exit $? $(date +%T)"
  stop_server
}
conc() { # reuse(on|off)
  local RU=$1 tag=reuse$1
  if [ $RU = off ]; then export INTARSIA_CB_NO_REUSE=1; else unset INTARSIA_CB_NO_REUSE; fi
  INTARSIA_CB_TRACE=1 start_server batched-$tag --max-batch 8 || { unset INTARSIA_CB_NO_REUSE; return 1; }
  unset INTARSIA_CB_NO_REUSE
  "${CB[@]}" run --mode concurrent --url $URL --cases $CASES --arrival-scale 0.25 --token-ids on --b 8 --stream mixed --out $O/conc-$tag-mixed.jsonl > $O/conc-$tag-mixed.log 2>&1; echo "run $tag: exit $? $(date +%T)"
  curl -s $URL/metrics > $O/metrics-$tag.txt
  stop_server
}
ref 2048; ref 256
conc off; conc on
for RU in off on; do
  tag=reuse$RU
  "${CB[@]}" compare --cases $CASES --ref $O/ref2048.jsonl --ref-alt $O/ref256.jsonl --test $O/conc-$tag-mixed.jsonl --report $O/report-$tag-mixed.json > $O/compare-$tag-mixed.log 2>&1; echo "COMPARE $tag: exit $?"
  python3 $W/tools/cbfifo.py check $O/server-batched-$tag.log --allow-open --json $O/fifo-$tag.json > $O/fifo-$tag.txt 2>&1; echo "FIFO $tag: exit $?"; cat $O/fifo-$tag.txt
done
for f in $O/compare-*.log; do echo "--- $f"; tail -12 $f; done
pgrep -fl "intarsia.* serve" || echo "no intarsia serve left"
echo "done $(date)"
