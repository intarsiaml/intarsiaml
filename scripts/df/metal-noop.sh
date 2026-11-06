#!/bin/zsh
# Lane df, Metal fix 1 no-op gates (one exclusive slot): with INTARSIA_METAL_MXFP4_MR unset, `plan show --json` (8B, 27B, 35B) and
# `kernels list` / `kernels test` (metal) of the lane binary against the main binary (2abdc2475); plus the 27B plan with the knob set.
#   fleetq run --host endor --exclusive --mem 40 --expect 30 --label "..." -- zsh scripts/df/metal-noop.sh <lane bin> <base bin> <out>
set -u
B=${1:?lane binary}; A=${2:?base binary}; O=${3:?out dir}; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
typeset -A MOD; MOD=(8b /Users/josef/models/Qwen3-8B 27b /Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4 35b /Users/josef/models/Qwen3.6-35B-A3B-NVFP4)
for t in 8b 27b 35b; do
  for side in base lane; do
    bin=$A; [[ $side == lane ]] && bin=$B
    $bin plan show --backend metal --model $MOD[$t] --max-seq 4096 --json > $O/plan-$t-$side.json 2> $O/plan-$t-$side.err < /dev/null
    say "plan $t $side exit $?"
  done
  cmp -s $O/plan-$t-base.json $O/plan-$t-lane.json && say "SAME plan $t" || say "DIFF plan $t"
done
INTARSIA_METAL_MXFP4_MR=1 $B plan show --backend metal --model $MOD[27b] --max-seq 4096 --json > $O/plan-27b-lane-mr1.json 2> $O/plan-27b-lane-mr1.err < /dev/null
cmp -s $O/plan-27b-lane.json $O/plan-27b-lane-mr1.json && say "SAME plan 27b knob unset vs 1" || say "DIFF plan 27b knob unset vs 1"
for side in base lane; do
  bin=$A; [[ $side == lane ]] && bin=$B
  $bin kernels list --backend metal > $O/klist-$side.txt 2>&1 < /dev/null
  $bin kernels test --backend metal > $O/ktest-$side.txt 2>&1 < /dev/null
  say "kernels test $side exit $? $(tail -2 $O/ktest-$side.txt | tr '\n' '|' | cut -c1-200)"
done
cmp -s $O/klist-base.txt $O/klist-lane.txt && say "SAME kernels list" || say "DIFF kernels list"
diff <(grep -vE '[0-9]+\.[0-9]+ ?(ms|us|s)\b' $O/ktest-base.txt) <(grep -vE '[0-9]+\.[0-9]+ ?(ms|us|s)\b' $O/ktest-lane.txt) > $O/ktest.diff && say "SAME kernels test (timing lines dropped)" || say "DIFF kernels test: $(wc -l < $O/ktest.diff) lines"
say done
