#!/usr/bin/env bash
# endor, inside a fleetq --load cpu slot: host checks of the branch (no GPU).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-spec
export CARGO_TARGET_DIR=/opt/code/intarsiaml/.worktrees/ink-hip-spec/target
mkdir -p .scratch/logs
step() { echo "[host] $1 $(date +%T)"; }
step "hip type-check (cfg ik_hip, tests)"
INTARSIA_HIP_CHECK_ONLY=1 cargo check -p intarsia-backend-hip --tests > .scratch/logs/check-hip.log 2>&1; echo "exit $?"; grep -aE '^(error|warning: unused)' -A5 .scratch/logs/check-hip.log | head -40
step "hip lib tests (stub)"
cargo test -p intarsia-backend-hip --lib > .scratch/logs/test-hip-lib.log 2>&1; echo "exit $?"; grep -aE '^test result|FAILED|panicked' .scratch/logs/test-hip-lib.log | head
step "core lib tests (dspark rule, spec_graph, inkling2)"
cargo test -p intarsia-core --lib > .scratch/logs/test-core.log 2>&1; echo "exit $?"; grep -aE '^test result|FAILED|panicked' .scratch/logs/test-core.log | head
step "models lib tests (load: dspark_default)"
cargo test -p intarsia-models --lib > .scratch/logs/test-models-lib.log 2>&1; echo "exit $?"; grep -aE '^test result|FAILED|panicked' .scratch/logs/test-models-lib.log | head
for t in inkling2 inkling2_dspark inkling2_dspark_cut inkling2_verify_softcap; do
  step "cpu test $t"
  cargo test -p intarsia-models --test $t > .scratch/logs/test-cpu-$t.log 2>&1; echo "exit $?"; grep -aE '^test result|FAILED|panicked' .scratch/logs/test-cpu-$t.log | head
done
step "docs_env_vars"
cargo test -p intarsia-cli --test docs_env_vars > .scratch/logs/test-docs-env.log 2>&1; echo "exit $?"; grep -aE '^test result|FAILED|panicked' .scratch/logs/test-docs-env.log | head
step done
