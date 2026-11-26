#!/usr/bin/env bash
# endor --load cpu: host checks of the HIP default flip (dspark rule tests, load dspark_default test, hip type-check, docs env vars).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-spec
export CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/ink-hip-spec/target
mkdir -p .scratch/logs
echo "[host] core dspark rule tests $(date +%T)"; cargo test -p intarsia-core --lib dspark 2>&1 | grep -aE '^test |test result|panicked' | head -10
echo "[host] models load tests $(date +%T)"; cargo test -p intarsia-models --lib dspark_default 2>&1 | grep -aE '^test |test result|panicked' | head -10
echo "[host] models lib all $(date +%T)"; cargo test -p intarsia-models --lib 2>&1 | grep -aE 'test result|panicked|FAILED' | head -6
echo "[host] recipe tests (core lib all) $(date +%T)"; cargo test -p intarsia-core --lib 2>&1 | grep -aE 'test result|panicked|FAILED' | head -6
echo "[host] hip type-check $(date +%T)"; INTARSIA_HIP_CHECK_ONLY=1 cargo check -p intarsia-backend-hip -p intarsia-models --tests 2>&1 | grep -aE '^error' -A5 | head -20; echo "check exit ${PIPESTATUS[0]}"
echo "[host] docs_env_vars $(date +%T)"; cargo test -p intarsia-cli --test docs_env_vars 2>&1 | grep -aE 'test result|panicked' | head -4
echo "[host] done $(date +%T)"
