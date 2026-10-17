#!/usr/bin/env bash
# Window 1 (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): the base tree (perf/cuda-norm-gemv 85823f7).
#  1. nsys kernel table of one ~4096-token prefill at chunk 2048, default config, alone on the GPU.
#  2. base prefill KL21 against the vLLM golden at three prefill chunks (the base's own perturbation spread):
#     35b-short and 35b-long, --mode prefill. Two lanes in parallel (deterministic, GPU sharing does not change bits).
# Output: ~/intarsia-decfuse/out/w1
set -u
P=$HOME/intarsia-decfuse
BASE=$P/base/target/acc/release/intarsia
O=$P/out/w1; rm -rf $O; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
DEADLINE=$(( $(date +%s) + ${W_BUDGET_MIN:-7} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
echo "base: $($BASE --version 2>&1 | head -1)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee $O/clocks.txt
sha256sum $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors | cut -c1-16

# 1. profile (alone)
head -c 14000 $P/base/docs/accuracy.md > $O/prompt.txt
echo "== nsys $(date +%T)"
GEN="$BASE generate -m $M35 --backend cuda --mem-budget 29 --max-seq 8192 --prefill-chunk 2048 --profile-every 0 --max-tokens 1 -p \"\$(cat $O/prompt.txt)\""
nsys profile -t cuda --force-overwrite true -o $O/nsys bash -c "$GEN" > $O/nsys.out 2> $O/nsys.err
grep -aE "^finish=|^steps=|prompt" $O/nsys.err | head -5 | cut -c1-250
nsys stats -r cuda_gpu_kern_sum -f csv --force-export=true -o $O/kern $O/nsys.nsys-rep > $O/stats.log 2>&1 || tail -3 $O/stats.log
rm -f $O/nsys.sqlite
echo "== profile done $(date +%T), $(left) min left"

# 2. base accuracy at three chunks, two lanes
acc() { # acc <tag> <golden> <chunk> <extra args>
  local tag=$1 g=$2 chunk=$3; shift 3
  local t0=$(date +%s)
  timeout 330 $BASE accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty "$@" > $O/acc-$tag.log 2>&1
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill ' $O/acc-$tag.log | head -1 | cut -c1-200)" >> $O/lanes.log
}
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
( acc short-c2048 $S 2048; acc short-c256 $S 256; acc short-c1024 $S 1024; acc long-c1024 $L 1024 ) &
( acc long-c2048 $L 2048; acc long-c256 $L 256 ) &
wait
echo "== w1 done $(date +%T), $(left) min left"
cat $O/lanes.log
