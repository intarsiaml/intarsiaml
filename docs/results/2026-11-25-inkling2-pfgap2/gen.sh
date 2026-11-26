#!/bin/sh
# ink-pfgap2 step 1b: same cb-trace question through `intarsia generate --max-tokens 1`, a fresh process per run (the path the 11-21 trace used).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
W=/opt/code/intarsiaml/.worktrees/ink-pfgap2; O=$W/.scratch/gen; mkdir -p $O
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
B=$W/target/release/intarsia; [ -x $B ] || { echo "GUARD: no binary"; exit 1; }
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
C=$W/crates/intarsia-models/examples/data/bench_corpus.txt
echo "tree $(git -C $W rev-parse --short=9 HEAD) $(git -C $W status --porcelain --untracked-files=no | wc -l | tr -d ' ') changed $(date +%T)"
n=0
for spec in 4k:12500 4k:12500 8k:23500 8k:23500 16k:46500 16k:46500; do
  sz=${spec%%:*}; ch=${spec#*:}; n=$((n + 1))
  P=$(python3 -c "import sys;t=open(sys.argv[1]).read();print('Request $n.\n'+t[$n*700:$n*700+$ch]+' Summarize the text above.')" "$C")
  s=$(python3 -c 'import time; print(time.clock_gettime(time.CLOCK_UPTIME_RAW))')
  env INTARSIA_METAL_PROFILE=cb:$O/cb-$n-$sz.jsonl $B generate --model $CK --backend metal --spec off --max-seq 20480 --max-tokens 1 --temperature 0 -p "$P" > $O/out-$n-$sz.txt 2> $O/err-$n-$sz.txt
  e=$(python3 -c 'import time; print(time.clock_gettime(time.CLOCK_UPTIME_RAW))')
  echo "run $n $sz start $s end $e rc $?"
  tail -4 $O/err-$n-$sz.txt | cut -c1-200
done
echo "== done $(date +%T)"
