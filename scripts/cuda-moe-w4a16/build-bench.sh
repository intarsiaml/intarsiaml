#!/usr/bin/env bash
# Build tree $TREE (default w4-dev) on spark1 (vLLM up) and run the microbench configs:  build-bench.sh "<T list>" "<cfg>" ...
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE=${TREE:-w4-dev}
"$S/../cuda-decode/build.sh" "$TREE" --tests
"$S/bench.sh" "$@"
