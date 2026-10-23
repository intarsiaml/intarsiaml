#!/usr/bin/env bash
# Lane FX5c: build the Metal CLI (also hosts the CPU oracle backend) in its own target dir.
#   fleetq run --host endor --mem 6 --expect 25m --label "fx5c metal build" -- scripts/fx5c/metal-build.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/opt/code/target-fx5c}
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -5
ls -la "$CARGO_TARGET_DIR/release/intarsia"
