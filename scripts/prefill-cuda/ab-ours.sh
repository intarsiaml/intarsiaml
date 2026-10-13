#!/usr/bin/env bash
# One round's "ours" arms on spark1 (inside scripts/prefill-cuda-window.sh: vLLM is down). Same procedure as
# scripts/phase1b-acceptance.sh stage_ab: a fresh `intarsia serve` per arm, the bench is the only client (1 Hz poller records
# the maximum running/waiting seen), `bench speed --warmup 1 --thinking off`, --profile-every 0, graphs required for ours-new.
#   ab-ours.sh <round> <arm:bin:extra-serve-flags> ...      e.g. new:$NEW: old:$OLD: newc:$NEW:--prefill-chunk=2048
# Env: SIZES (128,1024,4096,16384) GEN (128) REPS (5) AB_DIR (~/intarsia-perf/ab) MAX_SEQ (20480)
set -u
P=$HOME/intarsia-perf
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
SIZES=${SIZES:-128,1024,4096,16384}; GEN=${GEN:-128}; REPS=${REPS:-5}; MAX_SEQ=${MAX_SEQ:-20480}
AB=${AB_DIR:-$P/ab}; PORT=8090; RND=$1; shift; BENCH_BIN=${BENCH_BIN:-$P/dev/target/acc/release/intarsia}
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
for spec in "$@"; do
  arm=${spec%%:*}; rest=${spec#*:}; bin=${rest%%:*}; flags=${rest#*:}; flags=${flags//=/ }
  out=$AB/ours-$arm-r$RND; rm -rf $out; mkdir -p $out
  echo "== round $RND arm $arm ($bin $flags) $(date +%T)"
  nohup $bin serve --backend cuda --model $M --port $PORT --host 127.0.0.1 --served-model-name intarsia-35b --max-seq $MAX_SEQ --mem-budget 45 --profile-every 0 $flags > $out/serve.log 2>&1 < /dev/null &
  spid=$!
  up=0; for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }; sleep 3; done
  [ $up = 1 ] || { echo "server did not come up, see $out/serve.log"; kill $spid; exit 1; }
  echo '0 0' > $out/inflight; touch $out/inflight.run
  ( mr=0; mw=0; while [ -e $out/inflight.run ]; do
      m=$(curl -s http://127.0.0.1:$PORT/metrics)
      r=$(echo "$m" | awk '/^vllm:num_requests_running/ {print int($2)}' | tail -1); w=$(echo "$m" | awk '/^vllm:num_requests_waiting/ {print int($2)}' | tail -1)
      [ "${r:-0}" -gt $mr ] && mr=${r:-0}; [ "${w:-0}" -gt $mw ] && mw=${w:-0}; echo "$mr $mw" > $out/inflight; sleep 1; done ) > /dev/null 2>&1 &
  req=""; case "$arm" in new*) req="--require-graphs";; esac
  $BENCH_BIN bench speed --url http://127.0.0.1:$PORT --model intarsia-35b --sizes $SIZES --gen $GEN --reps $REPS --warmup 1 --thinking off --label prefillcuda-$arm-r$RND --out-dir $out $req > $out/bench.log 2>&1; echo $? > $out/bench.rc
  rm -f $out/inflight.run; sleep 2
  kill $spid; for i in $(seq 1 30); do kill -0 $spid 2>/dev/null || break; sleep 1; done; kill -9 $spid 2>/dev/null
  tail -n 12 $out/bench.log | cut -c1-200
  sleep 3
done
echo "== ab-ours round $RND done $(date +%T)"
