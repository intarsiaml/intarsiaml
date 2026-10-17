#!/usr/bin/env bash
# Window A (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): accuracy and gates of the FP4 MoE kernel
# (INTARSIA_CUDA_MOE_NVFP4=1) against the vLLM golden (sha b6b51e1b1699c66c), build nv-dev; build nv-base = origin/main.
#   lane 1: nv-dev prefill KL21 on 35b-short and 35b-long at chunks 256 and 2048 (the base spread was measured at 256 / 1024 / 2048)
#   lane 2: nv-base prefill KL21 at chunk 2048 (short, long) to confirm the base numbers on this commit, then the prefix-reuse
#           gate (`accuracy --mode reuse`, long golden, chunk 2048) for nv-dev with the kernel on and for nv-base
#   lane 3 (after lane 1): greedy ids, 256 tokens, 700-token prompt: dev kernel on eager vs graph, dev kernel off vs base
# Steps are skipped when the budget (W_BUDGET_MIN, default 9) is near. Output ~/intarsia-decfuse/out/wA
set -u
P=$HOME/intarsia-decfuse
DEV=$P/nv-dev/target/acc/release/intarsia
BASE=$P/nv-base/target/acc/release/intarsia
O=$P/out/wA; rm -rf $O; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
DEADLINE=$(( $(date +%s) + ${W_BUDGET_MIN:-9} * 60 ))
left() { echo $(( DEADLINE - $(date +%s) )); }   # seconds
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
echo "dev: $($DEV --version 2>&1 | head -1)  base: $($BASE --version 2>&1 | head -1)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee $O/clocks.txt
sha256sum $S | cut -c1-16
echo "== start $(date +%T)"

acc() { # acc <tag> <bin> <golden> <chunk> <env>
  local tag=$1 bin=$2 g=$3 chunk=$4 envs=$5 t0=$(date +%s)
  (( $(left) > 70 )) || { echo "acc $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env $envs timeout $(( $(left) < 330 ? $(left) : 330 )) $bin accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty > $O/acc-$tag.log 2>&1
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-$tag.log | head -1 | cut -c1-150) | $(grep -c moe_experts_nvfp4_mma_bf16 $O/acc-$tag.log) lines name the fp4 kernel" >> $O/lanes.log
}
reuse() { # reuse <tag> <bin> <env>
  local tag=$1 bin=$2 envs=$3 t0=$(date +%s)
  (( $(left) > 120 )) || { echo "reuse $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env $envs timeout $(( $(left) < 240 ? $(left) : 240 )) $bin accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty > $O/reuse-$tag.log 2>&1
  echo "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aiE 'reuse' $O/reuse-$tag.log | grep -aiE 'pass|fail|skip|identical|diverg|ok' | head -3 | cut -c1-200 | tr '\n' '|')" >> $O/lanes.log
}
gen() { # gen <tag> <bin> <env> [extra args]
  local tag=$1 bin=$2 envs=$3; shift 3
  (( $(left) > 60 )) || { echo "gen $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env $envs timeout 150 $bin generate -m $M35 --backend cuda --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c 3000 $P/nv-dev/tools/prompts/corpus_docs.txt)" --max-tokens 256 "$@" > $O/gen-$tag.out 2> $O/gen-$tag.err
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  echo "gen $tag: $(grep -aE 'finish=|steps=' $O/gen-$tag.err | head -2 | cut -c1-160 | tr '\n' '|') prompt $(grep -aEo 'prompt[ _]tokens?[=: ]+[0-9]+' $O/gen-$tag.err | head -1)" >> $O/lanes.log
}
(
  acc dev-short-c256 $DEV $S 256 INTARSIA_CUDA_MOE_NVFP4=1
  acc dev-short-c2048 $DEV $S 2048 INTARSIA_CUDA_MOE_NVFP4=1
  acc dev-long-c2048 $DEV $L 2048 INTARSIA_CUDA_MOE_NVFP4=1
  acc dev-long-c256 $DEV $L 256 INTARSIA_CUDA_MOE_NVFP4=1
  echo "== lane 1 done, $(left) s left $(date +%T)" >> $O/lanes.log
  gen dev-fp4-eager $DEV INTARSIA_CUDA_MOE_NVFP4=1 --no-graphs
  gen dev-fp4-graph $DEV INTARSIA_CUDA_MOE_NVFP4=1
  gen dev-off-graph $DEV INTARSIA_X=0
  gen base-graph $BASE INTARSIA_X=0
) &
(
  acc base-long-c2048 $BASE $L 2048 INTARSIA_X=0
  acc base-short-c2048 $BASE $S 2048 INTARSIA_X=0
  reuse dev-fp4 $DEV INTARSIA_CUDA_MOE_NVFP4=1
  reuse base $BASE INTARSIA_X=0
  echo "== lane 2 done, $(left) s left $(date +%T)" >> $O/lanes.log
) &
wait
echo "== done, $(left) s left $(date +%T)"
cat $O/lanes.log
for a in dev-fp4-eager dev-fp4-graph dev-off-graph base-graph; do printf '%s: %s ids\n' $a "$(tr -cd ',' < $O/gen-$a.ids 2>/dev/null | wc -c)"; done
cmp -s $O/gen-dev-fp4-eager.ids $O/gen-dev-fp4-graph.ids && echo "FP4 eager == FP4 graph ids" || echo "FP4 eager != FP4 graph ids"
cmp -s $O/gen-dev-off-graph.ids $O/gen-base-graph.ids && echo "DEV(kernel off) == BASE ids" || echo "DEV(kernel off) != BASE ids"
cmp -s $O/gen-dev-fp4-graph.ids $O/gen-base-graph.ids && echo "FP4 == BASE ids" || echo "FP4 != BASE ids (expected: activation quantisation)"
