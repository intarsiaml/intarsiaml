#!/bin/zsh
# Gate B: speculative decoding (--spec ngram) with batching (--max-batch 8) on Metal. One exclusive endor slot.
#   usage: gateB.sh <outdir>     env: IDS (comma list of case ids; default all 48), BS ("2 3 4 6 8"), SCALE_LOW (0.1), SCALE_HIGH (1.0)
# 1. solo references: --max-batch 1 --spec ngram at --prefill-chunk 2048 and 256 (cold, harness flush)
# 2. batched server --max-batch 8 --spec ngram, INTARSIA_CB_NO_REUSE=1 (reuse is gate A), INTARSIA_CB_TRACE=1
#    INTARSIA_SPEC_TRACE=1; client B in $BS, streams mixed; /metrics snapshots (spec counters) and epoch-ms windows per run
# 3. compare each run with the reference of the chunk the server reported (--ref-alt), as in gate 3 of M1 step 2
set -u
setopt shwordsplit
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; mkdir -p $O
HERE=${0:A:h}
source $HERE/serve_lib.sh
cd /opt/code/intarsiaml-cb-gates
trap stop_server EXIT
CB=(uv run tools/cbidentity.py)
CASES=tools/prompts/cb_spec_cases.jsonl
IDS=${IDS:-}
SEL=()
[ -n "$IDS" ] && SEL=(--ids $IDS)
nowms() { python3 -c 'import time;print(int(time.time()*1000))'; }
echo "fleetq job $FLEETQ_JOB; binary $BIN ($(shasum -a 256 $BIN | cut -c1-16)); git $(git rev-parse --short HEAD); cases sha $(shasum -a 256 $CASES | cut -c1-16)" | tee $O/env.txt
for C in 2048 256; do
  start_server ref$C 8101 --prefill-chunk $C --max-batch 1 --spec ngram || exit 3
  curl -s http://127.0.0.1:8101/v1/intarsia/plan > $O/plan-ref$C.json
  curl -s http://127.0.0.1:8101/metrics > $O/metrics-ref$C-before.txt
  "${CB[@]}" run --mode solo --url http://127.0.0.1:8101 --cases $CASES $SEL --token-ids on --out $O/ref$C.jsonl > $O/run-ref$C.log 2>&1; echo "ref $C: exit $?"
  curl -s http://127.0.0.1:8101/metrics > $O/metrics-ref$C-after.txt
  stop_server
done
INTARSIA_CB_NO_REUSE=1 INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1 start_server batched 8101 --max-batch 8 --spec ngram || exit 3
curl -s http://127.0.0.1:8101/v1/intarsia/plan > $O/plan-batched.json
: > $O/windows.txt
for B in ${BS:-2 3 4 6 8}; do
  if [ $B -le 3 ]; then SC=${SCALE_LOW:-0.1}; else SC=${SCALE_HIGH:-1.0}; fi
  curl -s http://127.0.0.1:8101/metrics > $O/metrics-b$B-before.txt
  T0=$(nowms)
  "${CB[@]}" run --mode concurrent --url http://127.0.0.1:8101 --cases $CASES $SEL --token-ids on --b $B --stream mixed --arrival-scale $SC --out $O/conc-b$B.jsonl > $O/run-conc-b$B.log 2>&1; echo "conc B=$B: exit $?"
  T1=$(nowms)
  curl -s http://127.0.0.1:8101/metrics > $O/metrics-b$B-after.txt
  echo "b$B $T0 $T1" >> $O/windows.txt
done
curl -s http://127.0.0.1:8101/health > $O/health-end.json
stop_server
"${CB[@]}" compare --ref $O/ref2048.jsonl --ref-alt $O/ref256.jsonl --cases $CASES --test $O/conc-b*.jsonl --report $O/report.json > $O/compare.log 2>&1; echo "compare: exit $?"
W=()
while read n a b; do W+=(--window $n $a $b); done < $O/windows.txt
python3 tools/cbspectrace.py $O/server-batched.log $W --json $O/spectrace.json > $O/spectrace.txt 2>&1; echo "spectrace: exit $?"
pgrep -fl "intarsia.* serve" || echo "no intarsia serve left"
