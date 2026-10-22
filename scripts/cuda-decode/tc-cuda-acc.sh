#!/usr/bin/env bash
# Lane tc: KL of the tail merge on spark1 (CUDA, 35B NVFP4), started from endor INSIDE an exclusive fleetq slot (vLLM stays up):
#   fleetq run --host spark1 --exclusive --mem 30 --label "tc: CUDA tail merge accuracy" -- bash scripts/cuda-decode/tc-cuda-acc.sh <out dir>
# Same arms as tc-cuda.sh (trees tc-base / tc-dev). base writes a self-golden of the long golden (5 prompts, 4064 .. 12064 tokens;
# at the CUDA recipe chunk of 2048 only the 4631-token prompt has a tail that folds: 2048 + 2048 + 535 -> 2048 + 2583); off and on
# compare against it. Prefill mode (the merge only changes the prefill pass).
set -uo pipefail
OUT=${1:?out dir}; TM=${TM:-1024}
mkdir -p "$OUT"
ssh spark1 "TM=$TM bash -s" > "$OUT/cuda-acc.log" 2>&1 <<'EOS'
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L=$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
mkdir -p $HOME/tc-out
run() { # run <arm> <tree> <env> <extra args...>
  local arm=$1 tree=$2 envx=$3; shift 3
  local t0=$(date +%s)
  env $envx timeout 1500 $HOME/intarsia-deploy/$tree/target/release/intarsia accuracy --model $M --backend cuda --mode prefill --max-seq 16384 --mem-budget 28 --allow-unstamped --allow-dirty "$@" < /dev/null > $HOME/tc-out/acc-$arm.log 2>&1
  echo "acc $arm exit $? $(( $(date +%s) - t0 )) s"
}
echo "MemAvailable $(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo) GB; golden $(sha256sum $L | cut -c1-16)"
run base tc-base "X=1" --golden $L --write-self-golden $HOME/tc-out/S-long.safetensors
run off tc-dev "X=1" --self-golden $HOME/tc-out/S-long.safetensors
run on tc-dev "INTARSIA_PREFILL_TAIL_MERGE=$TM" --self-golden $HOME/tc-out/S-long.safetensors
for a in off on; do echo "== $a"; grep -aE "^(long_|chat_reuse|OVERALL)|SELF-STATS|self-golden|rror" $HOME/tc-out/acc-$a.log | cut -c1-200; done
EOS
cat "$OUT/cuda-acc.log"
rsync -a spark1:tc-out/acc-base.log spark1:tc-out/acc-off.log spark1:tc-out/acc-on.log "$OUT/" < /dev/null
