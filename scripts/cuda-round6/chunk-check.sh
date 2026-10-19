#!/usr/bin/env bash
# Is the CUDA 35B prefill chunk-invariant between 2048 and 4096 / 8192? spark1 side, one fleetq slot with vLLM up (one 35B
# process at a time; the kernels are deterministic, so the bits do not depend on what else runs):
#   REMOTE_ENV="TREE=r6-base" fleetq run --host spark1 --mem 45 --label "..." -- scripts/cuda-round6/sync-scripts.sh scripts/cuda-round6/chunk-check.sh
# 1. greedy ids (GEN tokens) and the top-20 logits of the last prompt position for prompts of CHARS characters of
#    tools/prompts/corpus_docs.txt, at each chunk of CHUNKS; compared against the first chunk;
# 2. (ACC=1) KL21 of 35b-long at each chunk of ACC_CHUNKS (the base value at 2048 is round 5's 4.576e-2).
# Output ~/intarsia-decfuse/out/${OUTNAME:-r6chunk}
set -u
P=$HOME/intarsia-decfuse
T=${TREE:-r6-base}
BIN=$P/$T/target/acc/release/intarsia
O=$P/out/${OUTNAME:-r6chunk}; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 44 )) || { echo "ABORT: memory"; exit 1; }
echo "bin: $($BIN --version 2>&1 | head -1)"
CHUNKS=${CHUNKS:-2048 4096 8192}
for chars in ${CHARS:-20000 40000 54000}; do
  head -c $chars $P/$T/tools/prompts/corpus_docs.txt > $O/prompt-$chars.txt
  for c in $CHUNKS; do
    t0=$(date +%s)
    timeout 600 $BIN generate -m $M35 --backend cuda --mem-budget 40 --max-seq 20480 --prefill-chunk $c --profile-every 0 \
      --ignore-eos --show-ids --show-logits 20 --max-tokens ${GEN:-128} -p "$(cat $O/prompt-$chars.txt)" > $O/gen-$chars-c$c.out 2> $O/gen-$chars-c$c.err
    rc=$?
    grep -a '^ids:' $O/gen-$chars-c$c.err > $O/gen-$chars-c$c.ids
    grep -a '^top logits' $O/gen-$chars-c$c.err > $O/gen-$chars-c$c.top
    echo "gen $chars chars chunk $c exit $rc $(( $(date +%s) - t0 )) s: $(grep -aE 'finish=' $O/gen-$chars-c$c.err | head -1 | cut -c1-150)"
  done
  first=${CHUNKS%% *}
  for c in $CHUNKS; do
    [[ $c == "$first" ]] && continue
    i=$(cmp -s $O/gen-$chars-c$first.ids $O/gen-$chars-c$c.ids && echo identical || echo DIFFER)
    l=$(cmp -s $O/gen-$chars-c$first.top $O/gen-$chars-c$c.top && echo identical || echo DIFFER)
    echo "  $chars chars: chunk $c vs $first: ids $i, top-20 logits $l"
  done
done
if [[ "${ACC:-1}" == 1 ]]; then
  for c in ${ACC_CHUNKS:-4096}; do
    t0=$(date +%s)
    timeout 900 $BIN accuracy --model $M35 --backend cuda --golden $L --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $c --allow-dirty > $O/acc-long-c$c.log 2>&1
    echo "acc long chunk $c exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-long-c$c.log | head -1 | cut -c1-150)"
  done
fi
echo "== done $(date +%T)"
