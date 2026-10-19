#!/usr/bin/env bash
# Decode-step kernel profile of the 35B, per arm and context (spark1 side; ONE EXCLUSIVE fleetq slot with vLLM up and its GPU
# idle: relative shares only). Copy of scripts/cuda-prefill4/prof-decode.sh with arms and a clock log:
#   fleetq run --host spark1 --exclusive --mem 34 --label "..." -- scripts/cuda-prefill4/remote.sh r5-pipe scripts/cuda-round5/prof-decode.sh
# ARMS: ";"-separated "name:ENV=V ENV=V" (default "pipe:INTARSIA_X=0;base:INTARSIA_CUDA_SPLITKV_PIPE=0"), CHARS (prompt
# characters of tools/prompts/corpus_docs.txt, default "3400 54000": ~1k and ~16k tokens). Graph decode of 64 greedy tokens
# under nsys (--cuda-graph-trace=node), summarised per step by scripts/cuda-decode/nsys_steps.py; nvidia-smi samples SM
# clock / power every 200 ms during each run. Output ~/intarsia-decfuse/out/r5prof-<arm>-<chars>/
set -u
P=$HOME/intarsia-decfuse
T=${TREE:-r5-pipe}
BIN=$P/$T/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
NSYS=$(ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 34 )) || { echo "ABORT: memory"; exit 1; }
IFS=';' read -r -a arms <<< "${ARMS:-pipe:INTARSIA_X=0;base:INTARSIA_CUDA_SPLITKV_PIPE=0}"
for chars in ${CHARS:-3400 54000}; do
  for a in "${arms[@]}"; do
    name=${a%%:*}; envs=${a#*:}
    O=$P/out/r5prof-$name-$chars; rm -rf $O; mkdir -p $O
    head -c $chars $P/$T/tools/prompts/corpus_docs.txt > $O/prompt.txt
    echo "== $name, $chars chars, env '$envs' $(date +%T)"
    nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader -lms 200 > $O/clocks.csv 2>&1 &
    smi=$!
    env $envs $NSYS profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $O/nsys $BIN generate -m $M --backend cuda \
      --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --max-tokens 64 --show-ids -p "$(cat $O/prompt.txt)" > $O/nsys.out 2> $O/nsys.err
    kill $smi 2>/dev/null
    grep -aE "finish=" $O/nsys.err | head -1 | cut -c1-230
    grep -a '^ids:' $O/nsys.err | md5sum | sed 's/^/ids md5 /'
    awk -F', ' '$3+0 > 50 {split($1,a," "); s+=a[1]; n++} END {if (n) printf "SM clock while busy (>50%% util): %.0f MHz over %d samples\n", s/n, n}' $O/clocks.csv
    $NSYS stats -r cuda_gpu_trace -f csv --force-export=true -o $O/trace $O/nsys.nsys-rep > $O/stats.log 2>&1 || tail -3 $O/stats.log
    python3 $P/$T/scripts/cuda-decode/nsys_steps.py $O/trace_cuda_gpu_trace.csv --skip 4 --top 16 --json $O/steps.json | tee $O/steps.txt | grep -E 'steps total|span|splitkv|combine'
    rm -f $O/nsys.sqlite $O/nsys.nsys-rep $O/trace_cuda_gpu_trace.csv
  done
done
