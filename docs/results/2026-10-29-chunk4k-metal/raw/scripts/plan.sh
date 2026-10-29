#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/chunk4k
B=.lane/target/release/intarsia; M=$HOME/Models/Qwen3.6-35B-A3B-NVFP4; O=.lane/plan; mkdir -p $O
for ms in 8192 20480; do
$B plan show --backend metal --model $M --mem-budget 40 --max-seq $ms --json </dev/null > $O/new-$ms.json 2>$O/new-$ms.err; echo "new $ms exit $?"
$B plan show --backend metal --model $M --mem-budget 40 --max-seq $ms --prefill-chunk 2048 --json </dev/null > $O/old-$ms.json 2>$O/old-$ms.err; echo "old $ms exit $?"
diff $O/old-$ms.json $O/new-$ms.json > $O/diff-$ms.txt; echo "diff $ms: $(wc -l < $O/diff-$ms.txt) lines"
done
