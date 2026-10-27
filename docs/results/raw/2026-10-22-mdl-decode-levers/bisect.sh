#!/bin/zsh
# mdl lane: which fold costs decode speed? Interleaved rounds of main, head (all folds), head with one fold left out
# (INTARSIA_METAL_NO_FOLD=<kernel>), head with all left out. 1k context, 129 tokens, 35B, ONE exclusive endor slot.
# usage: bisect.sh <main-binary> <head-binary> <out-dir>
MAIN=$1; HEAD=$2; O=$3; HERE=${0:A:h}; M35=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
PROMPT=$(cat $HERE/prompt-1k.txt); mkdir -p $O; S=$O/summary.txt; date >> $S
ALL=matmul_fp8_pair_gemv,rmsnorm_add_matmul_fp8_pair_gemv,sigmoid_mul_matmul_fp8_gemv,gated_rmsnorm_matmul_fp8_gemv,rmsnorm_add_matmul_pair_bf16_gemv,matmul_nvfp4_glu_gemv,shared_down_combine_gemv
run() { tag=$1; bin=$2; nofold=$3
  INTARSIA_METAL_NO_FOLD=$nofold $bin generate -m $M35 --backend metal --profile-every 0 --temperature 0 --ignore-eos --max-seq 8192 -p "$PROMPT" --max-tokens 129 > $O/$tag.out 2> $O/$tag.err
  echo "$tag: $(grep -E 'finish=' $O/$tag.err | tail -1 | sed 's/.*decode/decode/')" | tee -a $S; }
for r in 1 2 3; do
  run main-$r $MAIN ""
  run head-$r $HEAD ""
  run alloff-$r $HEAD $ALL
  if [ $r -le 2 ]; then for f in ${(s:,:)ALL}; do run off-$f-$r $HEAD $f; done; fi
done
for t in main head alloff; do cmp -s $O/main-1.out $O/$t-1.out && echo "ids $t == main" || echo "ids $t DIFFER" ; done | tee -a $S
date >> $S
