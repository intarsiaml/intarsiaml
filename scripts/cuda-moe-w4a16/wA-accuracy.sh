#!/usr/bin/env bash
# Window A (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): accuracy and gates of the W4A16 MoE kernel
# (default on in w4-dev) against the vLLM golden (sha b6b51e1b1699c66c); w4-base = perf/cuda-moe-nvfp4-mma (2aff2199), FP4 off.
#   lane 1: dev prefill KL21 on 35b-short and 35b-long at chunk 2048 (the production chunk: every MoE call is T = 2048), greedy
#           ids (256 tokens, ~1500-token prompt): dev eager vs graph, dev with INTARSIA_CUDA_MOE_W4A16=0, base; 8B short golden
#   lane 2: prefix-reuse gate (`accuracy --mode reuse`, long golden, chunk 2048) for dev and base, 8B long golden, base KL21
#           at chunk 2048 (short, long) to confirm the base numbers on this box
# Steps are skipped when the budget (W_BUDGET_MIN, default 9) is near. Output ~/intarsia-decfuse/out/w4A
set -u
P=$HOME/intarsia-decfuse
DEV=$P/w4-dev/target/acc/release/intarsia
BASE=$P/w4-base/target/acc/release/intarsia
O=$P/out/w4A; rm -rf $O; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
DEADLINE=$(( $(date +%s) + ${W_BUDGET_MIN:-9}*60 + 30 ))
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
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-$tag.log | head -1 | cut -c1-150) | $(grep -c moe_experts_w4a16_bf16 $O/acc-$tag.log) lines name the w4a16 kernel, $(grep -c moe_experts_mma_bf16 $O/acc-$tag.log) the grouped one" >> $O/lanes.log
}
acc8() { # acc8 <tag> <golden file>
  local tag=$1 t0=$(date +%s)
  (( $(left) > 90 )) || { echo "acc8 $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  timeout $(( $(left) < 240 ? $(left) : 240 )) $DEV accuracy --model $M8 --backend cuda --golden $GD/$2 --mode both --decode-positions 16 --max-seq 4096 --mem-budget 40 --gate-class bf16 --allow-dirty > $O/acc8-$tag.log 2>&1
  echo "acc8 $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E 'verdict|golden gate|^prefill |^decode|PASS|FAIL' $O/acc8-$tag.log | head -4 | cut -c1-160 | tr '\n' '|')" >> $O/lanes.log
}
reuse() { # reuse <tag> <bin> <env>
  local tag=$1 bin=$2 envs=$3 t0=$(date +%s)
  (( $(left) > 120 )) || { echo "reuse $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env $envs timeout $(( $(left) < 240 ? $(left) : 240 )) $bin accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty > $O/reuse-$tag.log 2>&1
  echo "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aiE 'reuse' $O/reuse-$tag.log | grep -aiE 'pass|fail|skip|identical|diverg|ok' | head -3 | cut -c1-200 | tr '\n' '|')" >> $O/lanes.log
}
gen() { # gen <tag> <bin> <env> [extra args]; ~1500-token prompt: above the 1024-token minimum of the W4A16 kernel
  local tag=$1 bin=$2 envs=$3; shift 3
  (( $(left) > 60 )) || { echo "gen $tag SKIPPED: $(left) s left" >> $O/lanes.log; return; }
  env $envs timeout 150 $bin generate -m $M35 --backend cuda --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c 5000 $P/w4-dev/tools/prompts/corpus_docs.txt)" --max-tokens 256 "$@" > $O/gen-$tag.out 2> $O/gen-$tag.err
  grep -a '^ids:' $O/gen-$tag.err > $O/gen-$tag.ids
  echo "gen $tag: $(grep -aE 'finish=|steps=' $O/gen-$tag.err | head -2 | cut -c1-160 | tr '\n' '|')" >> $O/lanes.log
}
(
  acc dev-short-c2048 $DEV $S 2048 INTARSIA_X=0
  gen dev-v2-eager $DEV INTARSIA_X=0 --no-graphs
  gen dev-v2-graph $DEV INTARSIA_X=0
  acc dev-long-c2048 $DEV $L 2048 INTARSIA_X=0
  gen dev-off-graph $DEV INTARSIA_CUDA_MOE_W4A16=0
  acc8 short qwen3-8b-fp32.safetensors
  gen base-graph $BASE INTARSIA_X=0
  echo "== lane 1 done, $(left) s left $(date +%T)" >> $O/lanes.log
) &
(
  reuse dev $DEV INTARSIA_X=0
  reuse base $BASE INTARSIA_X=0
  acc base-short-c2048 $BASE $S 2048 INTARSIA_X=0
  acc base-long-c2048 $BASE $L 2048 INTARSIA_X=0
  acc8 long qwen3-8b-long-fp32.safetensors
  echo "== lane 2 done, $(left) s left $(date +%T)" >> $O/lanes.log
) &
wait
echo "== done, $(left) s left $(date +%T)"
cat $O/lanes.log
for a in dev-v2-eager dev-v2-graph dev-off-graph base-graph; do printf '%s: %s ids\n' $a "$(tr -cd ',' < $O/gen-$a.ids 2>/dev/null | wc -c)"; done
cmp -s $O/gen-dev-v2-eager.ids $O/gen-dev-v2-graph.ids && echo "v2 eager == v2 graph ids" || echo "v2 eager != v2 graph ids"
cmp -s $O/gen-dev-v2-graph.ids $O/gen-dev-off-graph.ids && echo "v2 == v1 (kernel off) ids" || echo "v2 != v1 (kernel off) ids"
cmp -s $O/gen-dev-v2-graph.ids $O/gen-base-graph.ids && echo "dev == BASE ids" || echo "dev != BASE ids"
