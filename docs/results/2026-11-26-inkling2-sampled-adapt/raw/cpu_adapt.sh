#!/bin/bash
# lane ink-sampled-adapt: the adapt tiny suite + core unit tests on cpu (endor, fleetq --load cpu)
set -u
W=/opt/code/intarsiaml/.worktrees/ink-sampled-adapt
export CARGO_TARGET_DIR=$W/target
cd $W
L=$W/.scratch/logs/cpu-$(date +%H%M)
mkdir -p $L
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
cargo test --release -q -p intarsia-core --lib spec:: > $L/core_spec.txt 2>&1; echo "core spec:: rc $? | $(grep -E '^test result' $L/core_spec.txt)"
cargo test --release -q -p intarsia-models --test inkling2_dspark_sampled_adapt -- --test-threads=2 --nocapture > $L/adapt.txt 2>&1
echo "inkling2_dspark_sampled_adapt rc $? | $(grep -E '^test result' $L/adapt.txt)"
grep -E "^(table|aggressive|drafts|fixed):" $L/adapt.txt
grep -E "panicked" -A3 $L/adapt.txt | head
echo "logs $L"
