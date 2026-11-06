#!/bin/zsh
# Lane dm no-op gates (one slot): plan show --json (8B, 27B, 35B; default spec, --spec off, --spec mtp) and kernels list / test, lane vs base.
#   zsh scripts/dm/noop.sh <lane bin> <base bin> <out>
set -u
B=${1:?lane}; A=${2:?base}; O=${3:?out}; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
typeset -A MOD; MOD=(8b /Users/josef/models/Qwen3-8B 27b /Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4 35b /Users/josef/models/Qwen3.6-35B-A3B-NVFP4)
for t in 8b 27b 35b; do for sp in default off mtp; do
  args=(); [[ $sp != default ]] && args=(--spec $sp)
  for side in base lane; do bin=$A; [[ $side == lane ]] && bin=$B
    $bin plan show --backend metal --model $MOD[$t] --max-seq 4096 --json $args > $O/plan-$t-$sp-$side.json 2> $O/plan-$t-$sp-$side.err < /dev/null; done
  cmp -s $O/plan-$t-$sp-base.json $O/plan-$t-$sp-lane.json && say "SAME plan $t spec $sp" || say "DIFF plan $t spec $sp ($(wc -c < $O/plan-$t-$sp-lane.json) bytes)"
done; done
for side in base lane; do bin=$A; [[ $side == lane ]] && bin=$B
  $bin kernels list --backend metal > $O/klist-$side.txt 2>&1 < /dev/null
  $bin kernels test --backend metal > $O/ktest-$side.txt 2>&1 < /dev/null; say "kernels test $side exit $? $(tail -1 $O/ktest-$side.txt | cut -c1-160)"; done
diff $O/klist-base.txt $O/klist-lane.txt > $O/klist.diff && say "SAME kernels list" || say "kernels list diff: $(grep '^[<>]' $O/klist.diff | tr '\n' ' ' | cut -c1-300)"
diff <(grep -vE '[0-9]+\.[0-9]+ ?(ms|us|s)\b' $O/ktest-base.txt) <(grep -vE '[0-9]+\.[0-9]+ ?(ms|us|s)\b' $O/ktest-lane.txt) > $O/ktest.diff && say "SAME kernels test" || say "kernels test diff $(grep -c '^[<>]' $O/ktest.diff) lines"
say done
