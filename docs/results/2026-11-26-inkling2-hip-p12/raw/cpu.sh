#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (no GPU): build the cli as .scratch/bin/intarsia-<name> plus the hip test binaries
# (--no-run), then the CPU tests the lane touches (intarsia-core p12, the hip crate's kernel-table tests, docs_env_vars).
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-p12"
NAME=${1:?name}; shift
R=docs/results/2026-11-26-inkling2-hip-p12/raw
bash $R/build.sh "$NAME" "$@" || exit $?
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-p12 INTARSIA_CONTAINER_NAME=ink-hip-p12-build
trap 'docker rm -f ink-hip-p12-build >/dev/null 2>&1' EXIT
deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-core --lib p12 < /dev/null 2>&1 | grep -aE "^test |test result|panicked|error" | tail -12
deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-backend-hip --lib < /dev/null 2>&1 | grep -aE "test result|panicked|FAILED|error\[" | tail -6
deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-cli --test docs_env_vars < /dev/null 2>&1 | grep -aE "test result|panicked|FAILED|error\[|missing" | tail -6
