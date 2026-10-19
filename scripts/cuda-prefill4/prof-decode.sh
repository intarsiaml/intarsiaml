#!/usr/bin/env bash
# Decode-step kernel profile of the 35B at a short and a long context (spark1 side; run inside one EXCLUSIVE fleetq slot
# with vLLM up, its GPU idle: relative shares only):
#   fleetq run --host spark1 --exclusive --mem 34 --label "..." -- scripts/cuda-prefill4/remote.sh p4-dev scripts/cuda-prefill4/prof-decode.sh
# Per context (CHARS: prompt characters of tools/prompts/corpus_docs.txt, default "3400 54000": ~1k and ~16k tokens at ~3.3 characters per token): graph
# decode of 64 greedy tokens under nsys (--cuda-graph-trace=node), summarised per step by scripts/cuda-decode/nsys_steps.py.
# Env for the engine: PROF_ENV ("A=1 B=2"). Output ~/intarsia-decfuse/out/p4prof-dec-<chars>/
set -u
P=$HOME/intarsia-decfuse
T=${TREE:-p4-dev}
BIN=$P/$T/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
NSYS=$(ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 34 )) || { echo "ABORT: memory"; exit 1; }
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv
wc -c $P/$T/tools/prompts/corpus_docs.txt
for chars in ${CHARS:-3400 54000}; do
  O=$P/out/p4prof-dec-$chars; rm -rf $O; mkdir -p $O
  head -c $chars $P/$T/tools/prompts/corpus_docs.txt > $O/prompt.txt
  echo "== $chars chars, env '${PROF_ENV:-}' $(date +%T)"
  env ${PROF_ENV:-INTARSIA_X=0} $BIN generate -m $M --backend cuda --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos \
    --max-tokens 64 -p "$(cat $O/prompt.txt)" > $O/plain.out 2> $O/plain.err
  grep -aE "finish=" $O/plain.err | head -1 | cut -c1-230
  env ${PROF_ENV:-INTARSIA_X=0} $NSYS profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $O/nsys $BIN generate -m $M --backend cuda \
    --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --max-tokens 64 -p "$(cat $O/prompt.txt)" > $O/nsys.out 2> $O/nsys.err
  grep -aE "finish=" $O/nsys.err | head -1 | cut -c1-230
  $NSYS stats -r cuda_gpu_trace -f csv --force-export=true -o $O/trace $O/nsys.nsys-rep > $O/stats.log 2>&1 || tail -3 $O/stats.log
  python3 $P/$T/scripts/cuda-decode/nsys_steps.py $O/trace_cuda_gpu_trace.csv --skip 4 --top 14 --json $O/steps.json | tee $O/steps.txt
  rm -f $O/nsys.sqlite $O/nsys.nsys-rep $O/trace_cuda_gpu_trace.csv
done
