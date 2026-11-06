#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/flip-metal-mtp
export CARGO_TARGET_DIR=$PWD/.scratch/target-test
cargo test --workspace --no-fail-fast > .scratch/logs/cargo-test-workspace.txt 2>&1
echo "cargo test exit $?"
grep -E "^test result|FAILED|failed" .scratch/logs/cargo-test-workspace.txt | awk '/^test result/{p+=$4; f+=$6; i+=$8} /FAILED|failed/ && !/^test result/{print} END{print "passed",p,"failed",f,"ignored",i}' | tail -15
