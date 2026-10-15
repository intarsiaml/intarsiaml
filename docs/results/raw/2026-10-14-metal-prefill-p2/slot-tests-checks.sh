#!/bin/zsh
# P2 final tests and correctness checks in one slot: cargo tests of the metal crate, then checks.sh.
# usage: slot-tests-checks.sh <outdir> <pre-binary> <new-binary>
O=$1; PRE=$2; NEW=$3
W=/opt/code/intarsiaml/.claude/worktrees/agent-a6a0312aabf864646
mkdir -p $O/cargo
cd $W
cargo test -p intarsia-backend-metal --release > $O/cargo/cargo-test-metal.log 2>&1; echo "cargo test metal exit $?"
cargo test -p intarsia-backend-metal --release --test flash_prefill -- --nocapture > $O/cargo/flash-test.log 2>&1; echo "flash tests exit $?"
grep -h "worst error" $O/cargo/flash-test.log
$W/docs/results/raw/2026-10-14-metal-prefill-p2/checks.sh $O $PRE $NEW
