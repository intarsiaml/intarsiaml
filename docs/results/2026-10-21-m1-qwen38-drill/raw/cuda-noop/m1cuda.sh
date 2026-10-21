#!/bin/bash
# runs ON spark1. build: CUDA release builds of base 4be7b7d4 and head 92794701 (cpu only). check: Qwen3-8B bf16 no-op,
# base then head (one at a time, ~18 GB), accuracy short + long and 256 greedy ids on the short and 4k prompts.
set -u
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda
N=$HOME/m1cuda; O=$N/out
M=$HOME/models/Qwen3-8B; G=$HOME/intarsiaml-golden
case "$1" in
build)
  for v in base head; do
    (cd $N/$v && CARGO_TARGET_DIR=$N/target-$v nice cargo build --release -q -p intarsia-cli --features cuda -j 10) || { echo "build $v FAILED"; exit 2; }
    echo "built $v $(date +%T)"
  done ;;
check)
  date +%T
  for v in base head; do
    B=$N/target-$v/release/intarsia
    $B accuracy --model $M --backend cuda --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $O/$v-acc-short.out 2> $O/$v-acc-short.err; echo "$v accshort rc=$? $(date +%T)"
    $B accuracy --model $M --backend cuda --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $O/$v-acc-long.out 2> $O/$v-acc-long.err; echo "$v acclong rc=$? $(date +%T)"
    for p in short 4k; do
      $B generate --model $M --backend cuda --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
        --prompt "$(cat $N/prompt-$p.txt)" > $O/$v-gen-$p.out 2> $O/$v-gen-$p.err; echo "$v gen-$p rc=$? $(date +%T)"
    done
  done
  cd $O
  for f in acc-short.out acc-long.out gen-short.out gen-4k.out; do
    cmp -s base-$f head-$f && echo "$f: identical" || echo "$f: DIFFERS"
  done
  for p in short 4k; do
    b=$(grep '^ids:' base-gen-$p.err | md5sum | cut -c1-8); h=$(grep '^ids:' head-gen-$p.err | md5sum | cut -c1-8)
    echo "ids $p: base $b head $h $([ "$b" = "$h" ] && echo identical || echo DIFFER) (n=$(grep '^ids:' base-gen-$p.err | wc -w))"
  done
  grep -h "OVERALL\|golden gate" base-acc-*.out
  date +%T ;;
esac
