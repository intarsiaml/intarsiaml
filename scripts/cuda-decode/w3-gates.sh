#!/usr/bin/env bash
# Window 3 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): the accuracy and graph gates of the dev build.
#   acc35     35b-short accuracy against the vLLM golden (sha b6b51e1b) on base AND dev, then a bitwise comparison of the two
#             self-golden dumps (top-64 log-probs of every prefill and decode row)
#   acc8      the two 8B fp32 goldens on dev
#   graph     256 greedy tokens, graph replay vs --no-graphs, ids must be identical (dev)
# Steps are skipped when the deadline is near (W3_BUDGET_MIN, default 14).
set -u
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
O=$P/out/w3; mkdir -p $O
DEV=$P/dev/target/acc/release/intarsia; BASE=$P/base/target/acc/release/intarsia
DEADLINE=$(( $(date +%s) + ${W3_BUDGET_MIN:-14} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
sha256sum $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors | cut -c1-16

SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60 --prefill-chunk 256"
acc35() { # acc35 <tag> <bin>
  need 4 "accuracy 35b-short $1" || return 0
  rm -f $O/$1.pos.tsv
  INTARSIA_ACC_POS_DUMP=$O/$1.pos.tsv $2 accuracy --model $M35 --backend cuda --golden $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors $SHORT \
    --write-baseline $O/$1.baseline.json --write-self-golden $O/$1.dump.safetensors --allow-dirty > $O/$1.log 2>&1
  echo "exit $?"; grep -E "verdict|golden gate|failing per-prompt|^prefill |^decode" $O/$1.log | head -8 | cut -c1-230
}
acc35 dev $DEV

acc8() { # acc8 <tag> <golden file>
  need 2 "accuracy 8B $1" || return 0
  $DEV accuracy --model $M8 --backend cuda --golden $GD/$2 --mode both --decode-positions 16 --max-seq 4096 --mem-budget 40 --gate-class bf16 --allow-dirty > $O/$1.log 2>&1
  echo "exit $?"; grep -E "verdict|golden gate|^prefill |^decode|PASS|FAIL" $O/$1.log | head -10 | cut -c1-230
}
gen35() { # gen35 <tag> <bin> [extra args]
  local tag=$1 bin=$2; shift 2
  $bin generate -m $M35 --backend cuda --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p 'Write a long story about a robot who learns to paint.' --max-tokens 256 "$@" > $O/gen35-$tag.out 2> $O/gen35-$tag.err
  grep -a '^ids:' $O/gen35-$tag.err > $O/gen35-$tag.ids
  grep -aE "finish=|steps=" $O/gen35-$tag.err | cut -c1-230
}
if need 3 "graph vs eager, 256 tokens"; then
  echo "--- eager (--no-graphs)"; gen35 eager $DEV --no-graphs
  echo "--- graph"; gen35 graph $DEV
  echo "--- base build, graph"; gen35 base $BASE
  echo "ids: $(tr -cd ',' < $O/gen35-eager.ids | wc -c) / $(tr -cd ',' < $O/gen35-graph.ids | wc -c) commas"
  if [ -s $O/gen35-eager.ids ] && cmp -s $O/gen35-eager.ids $O/gen35-graph.ids; then
    echo "GRAPH == EAGER: the 256 generated token ids are identical" | tee $O/graphproof.txt
  else
    echo "GRAPH != EAGER: token ids DIFFER" | tee $O/graphproof.txt; cmp $O/gen35-eager.ids $O/gen35-graph.ids | head -3
  fi
fi
if [ -s $O/gen35-base.ids ] && cmp -s $O/gen35-base.ids $O/gen35-graph.ids; then
  echo "BASE == DEV: the 256 generated token ids of the base build and the dev build are identical" | tee -a $O/graphproof.txt
else
  echo "BASE != DEV: token ids differ" | tee -a $O/graphproof.txt
fi
acc8 8b-short qwen3-8b-fp32.safetensors
acc8 8b-long qwen3-8b-long-fp32.safetensors
# last (lowest priority if the window is cut short): the base build's accuracy run and the bitwise comparison of the dumps
acc35 base $BASE
if [ -s $O/base.dump.safetensors ] && [ -s $O/dev.dump.safetensors ]; then
  python3 -I $D/cmp_dump.py $O/base.dump.safetensors $O/dev.dump.safetensors | tee $O/cmp.txt
  cmp $O/base.pos.tsv $O/dev.pos.tsv && echo "per-position TSV identical" | tee -a $O/cmp.txt
fi
echo "== w3 done $(date +%T), $(left) min left"
