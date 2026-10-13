#!/usr/bin/env bash
# The vLLM arm of one round (runs as a client in a small fleetq slot while vLLM is up; the same `bench speed` call as the
# ours arms). ab-vllm.sh <round>
set -u
P=$HOME/intarsia-perf; AB=${AB_DIR:-$P/ab}; RND=$1
BIN=${BENCH_BIN:-$P/dev3/target/acc/release/intarsia}
SIZES=${SIZES:-128,1024,4096,16384}; GEN=${GEN:-128}; REPS=${REPS:-5}
HOST=${VLLM_HOST:-100.120.88.104}
out=$AB/vllm-r$RND; rm -rf $out; mkdir -p $out; echo '0 0' > $out/inflight; touch $out/inflight.run
( mr=0; mw=0; while [ -e $out/inflight.run ]; do
    m=$(curl -s http://$HOST:8000/metrics)
    r=$(echo "$m" | awk '/^vllm:num_requests_running/ {print int($2)}' | tail -1); w=$(echo "$m" | awk '/^vllm:num_requests_waiting/ {print int($2)}' | tail -1)
    [ "${r:-0}" -gt $mr ] && mr=${r:-0}; [ "${w:-0}" -gt $mw ] && mw=${w:-0}; echo "$mr $mw" > $out/inflight; sleep 1; done ) > /dev/null 2>&1 &
$BIN bench speed --url http://$HOST:8000 --model guidance --sizes $SIZES --gen $GEN --reps $REPS --warmup 1 --thinking off --label prefillcuda-vllm-r$RND --out-dir $out > $out/bench.log 2>&1; echo $? > $out/bench.rc
rm -f $out/inflight.run; sleep 2
tail -n 12 $out/bench.log | cut -c1-200
