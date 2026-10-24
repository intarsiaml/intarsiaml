#!/usr/bin/env bash
# Lane lt: build the flip PR head (lt-dev) and main (lt-main), stamped, on one Spark, plus lt-dev's test binaries. Inside a fleetq slot (--load cpu).
# Usage: build2.sh <host> <dev src> <dev stamp> <main src> <main stamp>
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST=$1 SEED_TREE=lt-base "$D/build.sh" lt-dev "$2" "$3"
HOST=$1 SEED_TREE=lt-base "$D/build.sh" lt-main "$4" "$5"
ssh -n -o BatchMode=yes "$1" "export FLEETQ_JOB=$FLEETQ_JOB PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/intarsia-decfuse/lt-dev && { [ -d target/test ] || cp -a target/acc target/test; } && CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-backend-cuda -p intarsia-models -p intarsia-cli --no-run < /dev/null 2>&1 | grep -E '^error|Finished' | tail -5"
