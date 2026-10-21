#!/bin/bash
# runs ON tatooine inside the radiance-down window: Qwen3-8B bf16, base on card 0 and head on card 1 in parallel.
# accuracy short + long against the fp32 goldens (bf16 gate), 256 greedy ids on the short and the 4k prompt.
set -u
O=/tmp/m1noop-out
M=/models/Qwen3-8B; G=/models/golden
variant() {  # tree card
  local v=$1 c=$2
  local RUN="env INTARSIA_REPO=/tmp/m1noop-$v INTARSIA_TARGET_VOL=m1noop-target-$v HIP_VISIBLE_DEVICES=$c /tmp/m1noop-head/deploy/tatooine/dev.sh /target/release/intarsia"
  $RUN accuracy --model $M --backend hip --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $O/$v-acc-short.out 2> $O/$v-acc-short.err; echo "$v accshort rc=$? $(date +%T)"
  $RUN accuracy --model $M --backend hip --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $O/$v-acc-long.out 2> $O/$v-acc-long.err; echo "$v acclong rc=$? $(date +%T)"
  for p in short 4k; do
    $RUN generate --model $M --backend hip --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
      --prompt "$(cat /tmp/m1noop-prompt-$p.txt)" > $O/$v-gen-$p.out 2> $O/$v-gen-$p.err; echo "$v gen-$p rc=$? $(date +%T)"
  done
}
date +%T
variant base 0 &
variant head 1 &
wait
cd $O
for f in acc-short.out acc-long.out gen-short.out gen-4k.out; do
  if cmp -s base-$f head-$f; then echo "$f: identical"; else echo "$f: DIFFERS"; fi
done
for p in short 4k; do
  b=$(grep '^ids:' base-gen-$p.err | md5sum | cut -c1-8); h=$(grep '^ids:' head-gen-$p.err | md5sum | cut -c1-8)
  echo "ids $p: base $b head $h $([ "$b" = "$h" ] && echo identical || echo DIFFER) (n=$(grep '^ids:' base-gen-$p.err | wc -w))"
done
grep -h "golden gate\|OVERALL" base-acc-*.out | head -8
date +%T
