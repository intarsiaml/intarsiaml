#!/usr/bin/env bash
# Lane mpf G0 on CUDA ON spark1: the lane head (target-dev) against origin/main (target-base). `plan show` (JSON, text) and 256 greedy ids
# after a short and a ~3.5k-token prompt, Qwen3-8B, Gemma-4-26B-A4B NVFP4 and Qwen3.6-35B-A3B NVFP4.
#   8B and Gemma, no flag: dev == base (ids, plan): nothing but the 35B recipe turns MTP on.
#   35B: base no flag (plain decode) against dev --spec off and dev INTARSIA_SPEC_DEFAULT=0: ids and plan identical (the opt-outs);
#        against dev no flag (the new default, --spec mtp): ids identical (exactness of the default), plan reported (not required).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "mpf: G0 cuda" -- ssh spark1 'bash -s' < scripts/mpf/cuda-g0.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-mpf; H=/home/nvidia/.cache/huggingface/hub; O=$R/out-g0; mkdir -p $O
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
MG=$(ls -d $H/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/* | head -1)
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
PD=$R/src/scripts/tp2-hip/noop/prompts; fails=0
DEV=$R/target-dev/release/intarsia; BASE=$R/target-base/release/intarsia
# one <tag> <binary> <model-key> <model> <extra flags...>: ids for short and 4k, plan show json/text
one() {
  local tag=$1 b=$2 m=$3 M=$4; shift 4
  for p in short 4k; do
    env ${ENVX:-} $b generate -m $M --backend cuda --mem-budget 40 --max-seq 8192 "$@" --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
      -p "$(cat $PD/$p.txt)" --max-tokens 256 < /dev/null > /dev/null 2> $O/gen-$m-$p-$tag.err
    grep -a '^ids:' $O/gen-$m-$p-$tag.err > $O/gen-$m-$p-$tag.ids
  done
  env ${ENVX:-} $b plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 "$@" --json < /dev/null > $O/plan-$m-$tag.json 2> /dev/null
  env ${ENVX:-} $b plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 "$@" < /dev/null > $O/plan-$m-$tag.txt 2> /dev/null
}
# cmp <model-key> <tagA> <tagB> <need-plan 1|0>
cmpt() {
  local m=$1 A=$2 B=$3 plan=$4
  for p in short 4k; do
    if cmp -s $O/gen-$m-$p-$A.ids $O/gen-$m-$p-$B.ids && [ -s $O/gen-$m-$p-$A.ids ]; then echo "PASS ids $m $p $A == $B ($(tr ',' '\n' < $O/gen-$m-$p-$B.ids | wc -l) ids, md5 $(md5sum < $O/gen-$m-$p-$B.ids | cut -c1-8))"
    else echo "FAIL ids $m $p $A vs $B"; fails=$((fails+1)); fi
  done
  for k in json txt; do
    if cmp -s $O/plan-$m-$A.$k $O/plan-$m-$B.$k && [ -s $O/plan-$m-$A.$k ]; then echo "PASS plan $m $k $A == $B byte-identical ($(wc -c < $O/plan-$m-$B.$k) B)"
    elif [ $plan = 1 ]; then echo "FAIL plan $m $k $A vs $B ($(diff $O/plan-$m-$A.$k $O/plan-$m-$B.$k | wc -l) diff lines)"; fails=$((fails+1))
    else echo "INFO plan $m $k $A vs $B differ ($(diff $O/plan-$m-$A.$k $O/plan-$m-$B.$k | wc -l) diff lines)"; fi
  done
}
for mm in 8:$M8 g:$MG; do
  m=${mm%%:*}; M=${mm#*:}
  one base $BASE $m $M; one dev $DEV $m $M; cmpt $m base dev 1
done
one base $BASE 35 $M35
one dev-off $DEV 35 $M35 --spec off
ENVX=INTARSIA_SPEC_DEFAULT=0 one dev-envoff $DEV 35 $M35
one dev $DEV 35 $M35
cmpt 35 base dev-off 1; cmpt 35 base dev-envoff 1; cmpt 35 base dev 0
grep -a "default of recipe" $O/gen-35-short-dev.err | head -1
echo "g0 fails=$fails"
