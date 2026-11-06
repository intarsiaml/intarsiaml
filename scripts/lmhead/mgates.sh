#!/bin/zsh
# Lane lmhead-fp8, Metal no-op gates (variable unset): plan show (text + JSON), 256 greedy ids (short and ~3.5k prompts, default flags) of the
# Qwen3.8-27B (slot "8" of noop.sh) and the Qwen3.6-35B-A3B NVFP4 (slot "35"), head vs base binary; kernels test summary of both.
#   fleetq run --host endor --exclusive --mem 40 --expect 40 --label "..." -- zsh scripts/lmhead/mgates.sh <out dir>
set -u
R=${0:A:h:h:h}; O=${1:?out dir}; mkdir -p $O
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
zsh $R/scripts/lmhead/mrebuild.sh | tee $O/rebuild.txt
env | grep -q INTARSIA_LMHEAD_QUANT && { echo "INTARSIA_LMHEAD_QUANT is set"; exit 2; }
for t in base head; do
  B=$R/target/release/intarsia; [[ $t == base ]] && B=$R/.scratch/base/target/release/intarsia
  RUN="$B" bash $R/scripts/tp2-hip/noop/noop.sh run $O $t metal $M27 $M35 $R/scripts/tp2-hip/noop/prompts
  $B kernels test --backend metal < /dev/null > $O/kernels-$t.out 2>&1; echo "kernels $t exit $?"
  tail -4 $O/kernels-$t.out
done
bash $R/scripts/tp2-hip/noop/noop.sh cmp $O base head | tee $O/cmp.txt
diff <(grep -v -E "ms|time|[0-9]+\.[0-9]+ s" $O/kernels-base.out) <(grep -v -E "ms|time|[0-9]+\.[0-9]+ s" $O/kernels-head.out) > $O/kernels.diff && echo "kernels test output: identical (timing lines excluded)" || echo "kernels test output: DIFFERS, see kernels.diff"
