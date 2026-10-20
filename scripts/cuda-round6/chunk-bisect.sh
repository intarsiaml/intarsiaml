#!/usr/bin/env bash
# Which kernel makes the CUDA 35B prefill chunk-variant (2048 vs 4096)? spark1 side, one small slot with vLLM up (bits only:
# the kernels are deterministic, contention does not change them):
#   REMOTE_ENV="TREE=r6-base" fleetq run --host spark1 --mem 45 --label "..." -- scripts/cuda-round6/sync-scripts.sh scripts/cuda-round6/chunk-bisect.sh
# For each environment in ENVS (";"-separated; each applied to BOTH chunk arms), generate GEN greedy tokens and the top-20
# logits of the last prompt position for a CHARS-character prompt at chunk 2048 and 4096 and compare them.
set -u
P=$HOME/intarsia-decfuse
T=${TREE:-r6-base}
BIN=$P/$T/target/acc/release/intarsia
O=$P/out/${OUTNAME:-r6bisect}; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 44 )) || { echo "ABORT: memory"; exit 1; }
echo "-- chunk-check c8192 load line:"; grep -aE '^loaded|chunk' $P/out/r6chunk/gen-20000-c8192.err | head -3 | cut -c1-300
head -c ${CHARS:-20000} $P/$T/tools/prompts/corpus_docs.txt > $O/prompt.txt
IFS=';' read -r -a envs <<< "${ENVS:-INTARSIA_X=0;INTARSIA_CUDA_FP8_GEMM_REG=0;INTARSIA_CUDA_MOE_W4A16=0;INTARSIA_CUDA_GDN_SPLIT=0;INTARSIA_MOE_BM=64}"
i=0
for e in "${envs[@]}"; do
  i=$((i + 1))
  for c in 2048 4096; do
    env $e timeout 600 $BIN generate -m $M35 --backend cuda --mem-budget 40 --max-seq 20480 --prefill-chunk $c --profile-every 0 \
      --ignore-eos --show-ids --show-logits 20 --max-tokens ${GEN:-8} -p "$(cat $O/prompt.txt)" > $O/g$i-c$c.out 2> $O/g$i-c$c.err
    grep -a '^ids:' $O/g$i-c$c.err > $O/g$i-c$c.ids; grep -a '^top logits' $O/g$i-c$c.err > $O/g$i-c$c.top
  done
  a=$(cmp -s $O/g$i-c2048.ids $O/g$i-c4096.ids && echo identical || echo DIFFER)
  b=$(cmp -s $O/g$i-c2048.top $O/g$i-c4096.top && echo identical || echo DIFFER)
  echo "env '$e': chunk 4096 vs 2048: ids $a, top-20 logits $b $(date +%T)"
done
echo "== done $(date +%T)"
