#!/usr/bin/env bash
# No-op / identity window (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): is tree B bit-identical to tree A on CUDA?
#   w-noop.sh <treeA> <treeB>          trees under ~/intarsia-decfuse (A = reference, B = candidate); output ~/intarsia-decfuse/out/noop-<B>
# Correctness runs (two lanes in parallel: they are deterministic, GPU sharing does not change a bit; no timing is taken from them):
#   lane 1  35b-short accuracy against the vLLM golden on A and on B with self-golden dumps, then cmp_dump.py + position TSV
#   lane 2  256 greedy ids of the 8B and the 35B, graphs and --no-graphs, on A and on B (all eight must be identical per model),
#           optional NOFUSION=1: B with INTARSIA_NO_FUSION=1 (35B and 8B) must match too; the two 8B fp32 goldens on B
# Then, alone on the GPU (clean timing): 35B decode A/B, two interleaved rounds (SIZES 128,1024; GEN 128; REPS 5), --require-graphs.
# Env: ROUNDS (2), ROUND_MIN (3: minutes a round needs), SIZES, REPS, W_BUDGET_MIN (default 14).
set -u
A=${1:?tree A}; B=${2:?tree B}
# optional 3rd arg: flags, comma separated: nofusion (add the INTARSIA_NO_FUSION=1 runs of B), nolanes, nospeed
FLAGS=,${3:-},
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=$HOME/intarsiaml-golden
O=$P/out/noop-$B; rm -rf $O; mkdir -p $O
BA=$P/$A/target/acc/release/intarsia; BB=$P/$B/target/acc/release/intarsia
DEADLINE=$(( $(date +%s) + ${W_BUDGET_MIN:-14} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
sha256sum $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors | cut -c1-16
echo "A=$A $($BA --version 2>&1 | head -1)"; echo "B=$B $($BB --version 2>&1 | head -1)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60 --prefill-chunk 256"
acc35() { # acc35 <tag> <bin>
  INTARSIA_ACC_POS_DUMP=$O/$1.pos.tsv $2 accuracy --model $M35 --backend cuda --golden $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors $SHORT \
    --write-baseline $O/$1.baseline.json --write-self-golden $O/$1.dump.safetensors --allow-dirty > $O/acc35-$1.log 2>&1
  echo "acc35 $1 exit $?" >> $O/lane1.log
  grep -E "verdict|golden gate|failing per-prompt|^prefill |^decode" $O/acc35-$1.log | head -8 | cut -c1-230 >> $O/lane1.log
}
acc8() { # acc8 <tag> <golden file>
  $BB accuracy --model $M8 --backend cuda --golden $GD/$2 --mode both --decode-positions 16 --max-seq 4096 --mem-budget 25 --gate-class bf16 --allow-dirty > $O/acc8-$1.log 2>&1
  echo "acc8 $1 exit $?" >> $O/lane2.log
  grep -E "verdict|golden gate|^prefill |^decode|PASS|FAIL" $O/acc8-$1.log | head -10 | cut -c1-230 >> $O/lane2.log
}
gen() { # gen <35b|8b> <tag> <bin> <envs-or-"-"> [extra args]
  local m=$1 tag=$2 bin=$3 envs=$4 mp mb; shift 4; [ "$envs" = - ] && envs=""
  if [ $m = 35b ]; then mp=$M35; mb=29; else mp=$M8; mb=25; fi
  env $envs $bin generate -m $mp --backend cuda --mem-budget $mb --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p 'Write a long story about a robot who learns to paint.' --max-tokens 256 "$@" > $O/gen$m-$tag.out 2> $O/gen$m-$tag.err
  grep -a '^ids:' $O/gen$m-$tag.err > $O/gen$m-$tag.ids
  echo "gen $m $tag: $(grep -aE 'finish=|steps=' $O/gen$m-$tag.err | cut -c1-200 | tr '\n' ' ')" >> $O/lane2.log
}
echo "== lanes start $(date +%T)"
if [[ $FLAGS != *,nolanes,* ]]; then
( acc35 A $BA; acc35 B $BB; echo "lane1 done $(date +%T)" >> $O/lane1.log ) &
(
  gen 35b A-graph $BA -; gen 35b B-graph $BB -; gen 35b A-eager $BA - --no-graphs; gen 35b B-eager $BB - --no-graphs
  gen 8b A-graph $BA -; gen 8b B-graph $BB -; gen 8b A-eager $BA - --no-graphs; gen 8b B-eager $BB - --no-graphs
  if [[ $FLAGS == *,nofusion,* ]]; then gen 35b B-nofusion $BB INTARSIA_NO_FUSION=1; gen 8b B-nofusion $BB INTARSIA_NO_FUSION=1; fi
  acc8 8b-short qwen3-8b-fp32.safetensors; acc8 8b-long qwen3-8b-long-fp32.safetensors
  echo "lane2 done $(date +%T)" >> $O/lane2.log
) &
wait
fi
echo "== lanes done $(date +%T), $(left) min left"
cat $O/lane1.log $O/lane2.log
for m in 35b 8b; do
  ref=$O/gen$m-A-graph.ids; n=$(tr -cd ',' < $ref | wc -c)
  for t in B-graph A-eager B-eager B-nofusion; do
    f=$O/gen$m-$t.ids; [ -e $f ] || continue
    if [ -s $ref ] && cmp -s $ref $f; then echo "IDS $m $t == A-graph ($n commas)"; else echo "IDS $m $t DIFFERS from A-graph"; cmp $ref $f | head -2; fi
  done
done | tee $O/idproof.txt
if [ -s $O/A.dump.safetensors ] && [ -s $O/B.dump.safetensors ]; then
  python3 -I $D/cmp_dump.py $O/A.dump.safetensors $O/B.dump.safetensors | tee $O/cmp.txt
  cmp $O/A.pos.tsv $O/B.pos.tsv && echo "per-position TSV identical" | tee -a $O/cmp.txt
fi
if [[ $FLAGS != *,nospeed,* ]]; then
  export SIZES=${SIZES:-128,1024} REPS=${REPS:-5} OUT=$O/ab
  for r in $(seq 1 ${ROUNDS:-2}); do
    if (( $(left) < ${ROUND_MIN:-3} )); then echo "SKIP A/B round $r: $(left) min left"; continue; fi
    if (( r % 2 == 1 )); then bash $D/w-ab.sh 35b $r A:$A B:$B; else bash $D/w-ab.sh 35b $r B:$B A:$A; fi
  done
fi
echo "== w-noop done $(date +%T), $(left) min left"
