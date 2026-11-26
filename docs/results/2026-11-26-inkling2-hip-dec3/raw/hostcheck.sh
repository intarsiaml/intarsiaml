#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (no GPU use): the host checks of lane ink-hip-dec3 in the dev image, target dir
# /target/hostcheck of the lane's volume (so the HIP build cache stays): the HIP crate type-checked with its tests (check-only),
# its stub build's tests (INTARSIA_NO_HIP=1: no kernels, the GPU tests skip), and the pre-push hook's two docs-sync tests.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-dec3"
mkdir -p logs
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-dec3 INTARSIA_CONTAINER_NAME=ink-hip-dec3-host
trap 'docker rm -f ink-hip-dec3-host >/dev/null 2>&1' EXIT
run() { echo "[host] $*"; deploy/tatooine/dev.sh env CARGO_TARGET_DIR=/target/hostcheck "$@" < /dev/null 2>&1 | grep -aE "^test result|error(\[|:)|warning: unused|FAILED|panicked|Finished|passed|failed" | tail -25; echo "[host] exit ${PIPESTATUS[0]}"; }
run env INTARSIA_HIP_CHECK_ONLY=1 cargo check -j8 -p intarsia-backend-hip --tests
run env INTARSIA_NO_HIP=1 cargo test -j8 -p intarsia-backend-hip
run cargo test -j8 --locked -p intarsia-cli --test docs_env_vars
run cargo test -j8 --locked -p intarsia-models --test recipes_doc
