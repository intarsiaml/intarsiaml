#!/bin/zsh
# Concurrent-identity gate for --busy-chunk 320 (and 256 as control), B=8, reuse off and on, one exclusive endor slot.
# References: solo servers (--max-batch 1) at --prefill-chunk 2048, 320, 256 (every case alone after a flush); each batched
# request is compared with the reference whose chunk equals the prefill_chunk the batched server reports for that request.
#   usage: idgate_slot.sh OUTDIR        env: BIN, STREAMS ("mixed"), PORT
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
source ${0:A:h}/guard.sh
O=${1:?outdir}; mkdir -p $O; O=${O:A}
BIN=${BIN:?}; PORT=${PORT:-8132}; STREAMS=(${=STREAMS:-mixed})
W=/opt/code/intarsiaml-chunk320; CASES=$W/tools/prompts/cb_identity_cases.jsonl
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
trap "stop_server; [ -n \"\${WDPID:-}\" ] && kill \$WDPID 2>/dev/null" EXIT
trap "exit 143" INT TERM
start_watchdog
echo "slot $FLEETQ_JOB $(date); binary $BIN $(shasum -a 256 $BIN | cut -c1-16); git $(git -C $W rev-parse --short HEAD); cases $(wc -l < $CASES) lines"
ref() { # chunk
  past_soft && { echo "SKIP ref $1 (past soft deadline)"; return 1; }
  start_server ref$1 --max-batch 1 --prefill-chunk $1 || return 1
  "${CB[@]}" run --mode solo --url $URL --cases $CASES --arrival-scale 0.25 --token-ids on --out $O/ref$1.jsonl > $O/ref$1.log 2>&1; echo "ref $1: exit $? $(date +%T)"
  stop_server
}
conc() { # busy-chunk reuse(on|off)
  past_soft && { echo "SKIP conc $1 $2 (past soft deadline)"; return 1; }
  local C=$1 RU=$2 tag=c$1-reuse$2
  if [ $RU = off ]; then export INTARSIA_CB_NO_REUSE=1; else unset INTARSIA_CB_NO_REUSE; fi
  start_server batched-$tag --max-batch 8 --busy-chunk $C || { unset INTARSIA_CB_NO_REUSE; return 1; }
  unset INTARSIA_CB_NO_REUSE
  for s in $STREAMS; do
    "${CB[@]}" run --mode concurrent --url $URL --cases $CASES --arrival-scale 0.25 --token-ids on --b 8 --stream $s --out $O/conc-$tag-$s.jsonl > $O/conc-$tag-$s.log 2>&1; echo "run $tag stream=$s: exit $? $(date +%T)"
  done
  curl -s $URL/metrics > $O/metrics-$tag.txt
  stop_server
}
cmp_all() { # busy-chunk
  for RU in off on; do for s in $STREAMS; do
    local tag=c$1-reuse$RU
    [ -s $O/conc-$tag-$s.jsonl ] || { echo "no run $tag $s"; continue; }
    "${CB[@]}" compare --cases $CASES --ref $O/ref2048.jsonl --ref-alt $O/ref320.jsonl $O/ref256.jsonl --test $O/conc-$tag-$s.jsonl --report $O/report-$tag-$s.json > $O/compare-$tag-$s.log 2>&1; echo "COMPARE $tag stream=$s: exit $?"
  done; done
}
ref 2048; ref 320
conc 320 off; conc 320 on
ref 256
conc 256 off; conc 256 on
cmp_all 320; cmp_all 256
for f in $O/compare-*.log; do echo "--- $f"; tail -12 $f; done
pgrep -fl "intarsia.* serve" || echo "no intarsia serve left"
echo "done $(date)"
