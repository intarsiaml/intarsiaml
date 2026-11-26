#!/usr/bin/env bash
# endor --load cpu: build the four tiny Inkling-2 test binaries (release, with the Metal backend) of the branch, no run.
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-spec
export CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/ink-hip-spec/target
cargo test --release -p intarsia-models --test inkling2 --test inkling2_dspark --test inkling2_dspark_cut --test inkling2_verify_softcap --no-run 2>&1 | tail -6
