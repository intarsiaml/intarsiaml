#!/usr/bin/env bash
# Lane FX5b: build the Metal CLI in its own target dir.
#   fleetq run --host endor --mem 6 --expect 25 --label "fx5b metal build" -- scripts/fx5b/build.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/opt/code/target-fx5b}
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -5
