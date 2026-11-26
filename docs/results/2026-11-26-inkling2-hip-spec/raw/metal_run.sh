#!/usr/bin/env bash
# endor --exclusive (brief 4): the tiny Inkling-2 suite on the Metal backend, tiny fixtures only (no real model).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-spec
export CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/ink-hip-spec/target
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "start $(date +%T) git $(git rev-parse --short HEAD)"
env INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_dspark --test inkling2_dspark_cut --test inkling2_verify_softcap 2>&1 | grep -aE '^test |test result|Running|panicked' 
echo "end $(date +%T)"
