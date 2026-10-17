#!/bin/zsh
# Part B: identity with --spec ngram AND prefix reuse (incl. decode-written KV) on together. One exclusive endor slot.
#   usage: slot_partB.sh OUTDIR     env: CHAINS (100), COPY (0.5), BS ("2 3 8")
# 1. solo reference: --max-batch 1 --spec ngram --prefill-chunk 256 (prefix cache on); cbreuse gen builds the chains and records the solo results
# 2. batched: --max-batch 8 --spec ngram --prefill-chunk 256, reuse ON, INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1; B concurrent clients, one chain each
# 3. compare each B against the reproduced-predecessor solo references
set -u
setopt shwordsplit
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; mkdir -p $O
SD=${0:A:h}; source $SD/env.sh
BIN=${BIN:-$HEAD}
M=$HOME/models/Qwen3-8B
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
SPID=""; MPID=""
hyg() { { echo "--- $1 $(date '+%H:%M:%S')"; ps -axo pid,pcpu,rss,etime,comm | sort -k2 -nr | head -12 | cut -c1-150; sysctl vm.swapusage; } >> $O/hygiene.txt 2>&1; }
start_server() {
  local name=$1 port=$2; shift 2
  $BIN serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $port "$@" > $O/server-$name.log 2>&1 &
  SPID=$!
  for i in {1..180}; do
    curl -sf http://127.0.0.1:$port/health > /dev/null 2>&1 && { echo "server $name pid $SPID ready on $port"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "server $name died"; tail -5 $O/server-$name.log; return 1; }
    sleep 1
  done
  echo "server $name not ready"; return 1
}
stop_server() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; [ -n "$SPID" ] && wait $SPID 2>/dev/null; SPID=""; }
cleanup() { stop_server; sleep 4; [ -n "$MPID" ] && kill $MPID 2>/dev/null; hyg "end (servers stopped)"; $SD/end_sample.sh $O; }
trap cleanup EXIT INT TERM
hyg "start"
macmon pipe -i 1000 > $O/macmon.jsonl 2>/dev/null & MPID=$!
sleep 5
cd $W
CR=(uv run tools/cbreuse.py)
CHAINS=${CHAINS:-100}; COPY=${COPY:-0.5}
nowms() { python3 -c 'import time;print(int(time.time()*1000))'; }
echo "fleetq job $FLEETQ_JOB; binary $BIN ($(shasum -a 256 $BIN | cut -c1-16)); git $(git rev-parse --short HEAD)" | tee $O/env.txt
start_server ref 8101 --prefill-chunk 256 --max-batch 1 --spec ngram || exit 3
curl -s http://127.0.0.1:8101/v1/intarsia/plan > $O/plan-ref.json
curl -s http://127.0.0.1:8101/metrics > $O/metrics-ref-before.txt
"${CR[@]}" gen --url http://127.0.0.1:8101 --chains $CHAINS --copy-frac $COPY --out-dir $O/chains > $O/gen.log 2>&1; echo "gen: exit $?"
curl -s http://127.0.0.1:8101/metrics > $O/metrics-ref-after.txt
stop_server
INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1 start_server batched 8101 --max-batch 8 --prefill-chunk 256 --spec ngram || exit 3
curl -s http://127.0.0.1:8101/v1/intarsia/plan > $O/plan-batched.json
: > $O/windows.txt
for B in ${BS:-2 3 8}; do
  curl -s http://127.0.0.1:8101/metrics > $O/metrics-b$B-before.txt
  T0=$(nowms)
  "${CR[@]}" run --url http://127.0.0.1:8101 --chains $O/chains/chains.jsonl --b $B --out $O/conc-b$B.jsonl > $O/run-b$B.log 2>&1; echo "run B=$B: exit $?"
  T1=$(nowms)
  curl -s http://127.0.0.1:8101/metrics > $O/metrics-b$B-after.txt
  echo "b$B $T0 $T1" >> $O/windows.txt
done
curl -s http://127.0.0.1:8101/health > $O/health-end.json
stop_server
for B in ${BS:-2 3 8}; do
  "${CR[@]}" compare --ref $O/chains/ref.jsonl --chains $O/chains/chains.jsonl --test $O/conc-b$B.jsonl --report $O/report-b$B.json > $O/compare-b$B.log 2>&1; echo "compare B=$B: exit $?"
done
WIN=()
while read n a b; do WIN+=(--window $n $a $b); done < $O/windows.txt
python3 tools/cbspectrace.py $O/server-batched.log $WIN --json $O/spectrace.json > $O/spectrace.txt 2>&1; echo "spectrace: exit $?"
gzip -f $O/server-batched.log $O/server-ref.log
pgrep -fl "intarsia.* serve" || echo "no intarsia serve left"
