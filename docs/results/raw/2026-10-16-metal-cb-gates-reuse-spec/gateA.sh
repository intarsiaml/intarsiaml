#!/bin/zsh
# Gate A: concurrent identity with prefix reuse ON, incl. reuse of decode-written KV. One exclusive endor slot.
#   usage: gateA.sh <outdir>      env: CHAINS (130), CHUNK (256: both sides --prefill-chunk 256), DEFAULTRUN (1: also a default-chunk run)
# 1. solo reference: --max-batch 1 --prefill-chunk $CHUNK (prefix cache on); tools/cbreuse.py gen builds the chains
#    (P -> F -> F2, F's prompt = P's prompt + P's output ids) and records the solo results incl. cached_tokens
# 2. batched server: --max-batch 8 --prefill-chunk $CHUNK, reuse ON (no INTARSIA_CB_NO_REUSE), INTARSIA_CB_TRACE=1,
#    8 concurrent clients, one chain each, same prompts
# 3. optional: batched server with its default chunking (2048 alone / 256 while others decode); only requests whose
#    whole chain ran at the reference chunk are comparable
# 4. compare
set -u
setopt shwordsplit
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; mkdir -p $O
HERE=${0:A:h}
source $HERE/serve_lib.sh
cd /opt/code/intarsiaml-cb-gates
trap stop_server EXIT
CR=(uv run tools/cbreuse.py)
CHAINS=${CHAINS:-130}
CHUNK=${CHUNK:-256}
echo "fleetq job $FLEETQ_JOB; binary $BIN ($(shasum -a 256 $BIN | cut -c1-16)); git $(git rev-parse --short HEAD)" | tee $O/env.txt
start_server ref$CHUNK 8101 --prefill-chunk $CHUNK --max-batch 1 || exit 3
curl -s http://127.0.0.1:8101/v1/intarsia/plan > $O/plan-ref.json
"${CR[@]}" gen --url http://127.0.0.1:8101 --chains $CHAINS --out-dir $O/chains > $O/gen.log 2>&1; echo "gen: exit $?"
curl -s http://127.0.0.1:8101/metrics > $O/metrics-ref.txt
stop_server
# batched, same chunk, reuse on
INTARSIA_CB_TRACE=1 start_server batched$CHUNK 8101 --max-batch 8 --prefill-chunk $CHUNK || exit 3
curl -s http://127.0.0.1:8101/v1/intarsia/plan > $O/plan-batched$CHUNK.json
"${CR[@]}" run --url http://127.0.0.1:8101 --chains $O/chains/chains.jsonl --b 8 --out $O/conc-c$CHUNK-b8.jsonl > $O/run-c$CHUNK-b8.log 2>&1; echo "run c$CHUNK b8: exit $?"
curl -s http://127.0.0.1:8101/metrics > $O/metrics-batched$CHUNK.txt
curl -s http://127.0.0.1:8101/health > $O/health-batched$CHUNK.json
stop_server
"${CR[@]}" compare --ref $O/chains/ref.jsonl --chains $O/chains/chains.jsonl --test $O/conc-c$CHUNK-b8.jsonl --report $O/report-c$CHUNK-b8.json > $O/compare-c$CHUNK-b8.log 2>&1; echo "compare c$CHUNK: exit $?"
if [ "${DEFAULTRUN:-1}" = 1 ]; then
  INTARSIA_CB_TRACE=1 start_server batcheddef 8101 --max-batch 8 || exit 3
  curl -s http://127.0.0.1:8101/v1/intarsia/plan > $O/plan-batcheddef.json
  "${CR[@]}" run --url http://127.0.0.1:8101 --chains $O/chains/chains.jsonl --b 8 --out $O/conc-def-b8.jsonl > $O/run-def-b8.log 2>&1; echo "run def b8: exit $?"
  curl -s http://127.0.0.1:8101/metrics > $O/metrics-batcheddef.txt
  stop_server
  "${CR[@]}" compare --ref $O/chains/ref.jsonl --chains $O/chains/chains.jsonl --test $O/conc-def-b8.jsonl --report $O/report-def-b8.json > $O/compare-def-b8.log 2>&1; echo "compare def: exit $?"
fi
pgrep -fl "intarsia.* serve" || echo "no intarsia serve left"
