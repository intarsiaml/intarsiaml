#!/usr/bin/env bash
# mhead: the CPU suites of the gates (run on the Mac inside a fleetq slot):
# fleetq run --host endor --load cpu --mem 10 --expect 30 --label "mhead: cpu suites" -- scripts/mhead/cpu-suites.sh
set -uo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
R=$REPO/docs/results/2026-11-16-mhead/raw; mkdir -p "$R"; cd "$REPO"
cargo test --release -p intarsia-core --lib > "$R/cpu-core-lib.log" 2>&1; echo "core lib exit $?"
cargo test --release -p intarsia-models -p intarsia-cli --lib --tests --no-fail-fast > "$R/cpu-models-cli.log" 2>&1; echo "models + cli exit $?"
