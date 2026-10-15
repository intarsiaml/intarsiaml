#!/bin/zsh
# cb-m1 step 2 gate 3 (concurrent identity), one exclusive endor slot: usage gate3.sh <outdir> <rev> [args]
# 1. solo references: servers with --max-batch 1 and --prefill-chunk 2048 and 256 (cold prefill, harness flush)
# 2. batched server (--max-batch 8, INTARSIA_CB_NO_REUSE=1), client B in 2,4,8 x stream on,off
# 3. compare each request with the reference whose chunk equals the chunk the batched server reports for it
set -u
setopt shwordsplit
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; REV=$2; mkdir -p $O
HERE=${0:A:h}
source $HERE/serve_lib.sh
cd /opt/code/intarsiaml-cb-wire
trap stop_server EXIT
CB=(uv run tools/cbidentity.py)
SCALE=${SCALE:-0.25}
for C in 2048 256; do
  if [ -n "${REFDIR:-}" ]; then cp $REFDIR/ref$C.jsonl $O/ref$C.jsonl; continue; fi   # reuse references of an earlier run
  start_server ref$C 8101 --prefill-chunk $C --max-batch 1 || exit 3
  "${CB[@]}" run --mode solo --url http://127.0.0.1:8101 --token-ids on --out $O/ref$C.jsonl > $O/run-ref$C.log 2>&1; echo "ref $C: exit $?"
  stop_server
done
MB=${MB:-8}
INTARSIA_CB_NO_REUSE=${NOREUSE:-1} INTARSIA_CB_TRACE=${TRACE:-0} start_server batched 8101 --max-batch $MB || exit 3
for B in ${BS:-2 4 8}; do
  for SM in on off; do
    "${CB[@]}" run --mode concurrent --url http://127.0.0.1:8101 --token-ids on --b $B --stream $SM --arrival-scale $SCALE --out $O/conc-mb$MB-b$B-$SM.jsonl > $O/run-conc-mb$MB-b$B-$SM.log 2>&1; echo "conc mb$MB B=$B $SM: exit $?"
  done
done
curl -s http://127.0.0.1:8101/metrics > $O/metrics-end-mb$MB.txt
curl -s http://127.0.0.1:8101/health > $O/health-end-mb$MB.json
stop_server
"${CB[@]}" compare --ref $O/ref2048.jsonl --ref-alt $O/ref256.jsonl --test $O/conc-mb$MB-*.jsonl --report $O/report-mb$MB.json > $O/compare-mb$MB.log 2>&1; echo "compare: exit $?"
