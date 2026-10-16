#!/bin/zsh
# Job G2: concurrent-vs-solo identity with the interleaving of decode and 256-token prefill chunks (the only place chunk 256 runs in
# serving), staged uploads on. References: BASE binary, solo, --max-batch 1, --prefill-chunk 2048 and 256. Test: NEW binary,
# --max-batch 8, B = 4 and 8 clients, stream on/off. Same method as cb-wire's gate3.sh. Inside an exclusive slot.
# gate-conc.sh <outdir> <new binary> <base binary>
set -u
setopt shwordsplit
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; NEWBIN=$2; BASEBIN=$3; mkdir -p $O
W=/opt/code/intarsiaml-cbgap
M=$HOME/models/Qwen3-8B
SPID=""
start_server() { local name=$1 port=$2 bin=$3; shift 3
  $bin serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --port $port "$@" > $O/server-$name.log 2>&1 &
  SPID=$!
  for i in {1..180}; do
    curl -sf http://127.0.0.1:$port/health > /dev/null 2>&1 && { echo "server $name pid $SPID ready on $port"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "server $name died"; tail -5 $O/server-$name.log; return 1; }
    sleep 1
  done; echo "server $name not ready"; return 1; }
stop_server() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; [ -n "$SPID" ] && wait $SPID 2>/dev/null; SPID=""; }
trap stop_server EXIT
cd $W
CB=(uv run tools/cbidentity.py)
for C in 2048 256; do
  start_server ref$C 8101 $BASEBIN --prefill-chunk $C --max-batch 1 || exit 3
  "${CB[@]}" run --mode solo --url http://127.0.0.1:8101 --token-ids on --out $O/ref$C.jsonl > $O/run-ref$C.log 2>&1; echo "ref $C: exit $?"
  stop_server
done
INTARSIA_CB_NO_REUSE=1 start_server batched 8101 $NEWBIN --max-batch 8 || exit 3
for B in 4 8; do for SM in on off; do
  "${CB[@]}" run --mode concurrent --url http://127.0.0.1:8101 --token-ids on --b $B --stream $SM --arrival-scale 0.25 --out $O/conc-mb8-b$B-$SM.jsonl > $O/run-conc-mb8-b$B-$SM.log 2>&1; echo "conc mb8 B=$B $SM: exit $?"
done; done
curl -s http://127.0.0.1:8101/metrics > $O/metrics-end.txt
stop_server
"${CB[@]}" compare --ref $O/ref2048.jsonl --ref-alt $O/ref256.jsonl --test $O/conc-mb8-*.jsonl --report $O/report.json > $O/compare.log 2>&1; echo "compare: exit $?"
echo CONC_DONE
