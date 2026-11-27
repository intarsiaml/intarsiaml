#!/bin/bash
# lane ink-sampled-adapt: cpu-backend tiny suites + doc tests on endor (fleetq --load cpu)
set -u
W=/opt/code/intarsiaml/.worktrees/ink-sampled-adapt
export CARGO_TARGET_DIR=$W/target
cd $W
L=$W/.scratch/logs/cpu-$(date +%H%M)
mkdir -p $L
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
cargo build --release -p intarsia-models -p intarsia-cli --tests 2>&1 | grep -E '^(warning: unused|error)' -A5 | head -40
for t in inkling2_dspark_sampled_adapt inkling2_dspark_sampled inkling2_dspark_sampled_gumbel inkling2_dspark inkling2_dspark_cut inkling2_verify_softcap recipes_doc; do
  cargo test --release -q -p intarsia-models --test $t -- --test-threads=4 > $L/$t.txt 2>&1
  echo "$t rc $? | $(grep -E '^test result' $L/$t.txt | tr '\n' ' ')"
done
cargo test --release -q -p intarsia-cli --test docs_env_vars > $L/docs_env_vars.txt 2>&1; echo "docs_env_vars rc $? | $(grep -E '^test result' $L/docs_env_vars.txt)"
cargo test --release -q -p intarsia-core --lib > $L/core_lib.txt 2>&1; echo "core lib rc $? | $(grep -E '^test result' $L/core_lib.txt)"
echo "logs $L"
