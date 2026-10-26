#!/usr/bin/env bash
# Lane mxr, ON tatooine inside a fleetq job (--load cpu): cargo-build the dev tree (~/intarsiaml-mxr-dev) and the base tree
# (~/intarsiaml-mxr-dev/.base = origin/main) into bind-mounted target dirs on /data, plus the dev tests the gate window runs
# (compiled, not run), then run the GPU-free HIP tests. Incremental, re-runnable. usage: cbuild.sh [dev|base|both] [notests]
set -uo pipefail
what=${1:-both}
mkvol() { # <name> <dir>
  mkdir -p "$2" && chmod 0777 "$2"
  docker volume inspect "$1" >/dev/null 2>&1 || docker volume create --driver local -o type=none -o o=bind -o device="$2" "$1" >/dev/null
}
mkvol intarsia-mxr-target-dev /data/intarsiaml-mxr-target-dev
mkvol intarsia-mxr-target-base /data/intarsiaml-mxr-target-base
R=$HOME/intarsiaml-mxr-dev
D=$R/deploy/tatooine/dev.sh
dv() { INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-mxr-target-dev $D "$@" < /dev/null; }
bs() { INTARSIA_REPO=$R/.base INTARSIA_TARGET_VOL=intarsia-mxr-target-base $D "$@" < /dev/null; }
if [ "$what" != base ]; then
  echo "== dev $(date +%T): intarsia"; dv cargo build --release --locked --features hip -p intarsia-cli 2>&1 | grep -aE "^(error|warning: unused)|Finished|error\[" | tail -8
  echo "== dev $(date +%T): tests (compile only)"
  dv cargo test --release --no-run -p intarsia-backend-hip --test rows_exact --test rows_registry --test admission --test quick_coverage 2>&1 | grep -aE "^error|Finished|Executable" | tail -8
  dv cargo test --release --no-run -p intarsia-models --test batch_cuda_real 2>&1 | grep -aE "^error|Finished|Executable" | tail -3
  if [ "${2:-}" != notests ]; then
    echo "== dev $(date +%T): GPU-free tests"
    dv cargo test --release -p intarsia-backend-hip --test quick_coverage 2>&1 | grep -aE "^test |test result|panicked" | tail -4
    dv cargo test --release -p intarsia-backend-hip --lib 2>&1 | grep -aE "test result|panicked|FAILED" | tail -6
  fi
fi
if [ "$what" != dev ]; then
  echo "== base $(date +%T): intarsia"; bs cargo build --release --locked --features hip -p intarsia-cli 2>&1 | grep -aE "^error|Finished" | tail -3
  bs cargo test --release --no-run -p intarsia-models --test batch_cuda_real 2>&1 | grep -aE "^error|Finished" | tail -3
fi
df -h /data | tail -1
echo "BUILD DONE $(date +%T)"
