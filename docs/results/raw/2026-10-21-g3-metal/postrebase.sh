#!/bin/zsh
# G3: Metal crate tests + kernels test on the tree rebased onto origin/main 76635aa4 (after M1's mxfp4 merge #42).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687
O=$W/docs/results/raw/2026-10-21-g3-metal/postrebase; mkdir -p $O
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer INTARSIA_CPU_FREE_MEM_GB=60
cd $W
echo "head $(git rev-parse --short HEAD) $(date -Iseconds)" | tee $O/info.txt
INTARSIA_TEST_REAL_MODEL=1 cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $O/cargo-test-metal.txt 2>&1
echo "cargo test metal exit $?" | tee -a $O/info.txt
grep -h "test result" $O/cargo-test-metal.txt | awk '{p+=$4; f+=$6} END {print "total passed", p, "failed", f}' | tee -a $O/info.txt
$W/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/models/Qwen3-8B > $O/kernels-test.txt 2>&1
echo "kernels test exit $?" | tee -a $O/info.txt; tail -1 $O/kernels-test.txt | tee -a $O/info.txt
