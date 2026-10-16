#!/usr/bin/env bash
# CPU-slot build on tatooine (no GPU): the release CLI and the hip crate's test binaries, in the dev image.
#   fleetq acquire --host tatooine --mem 0 --label "..." --wait ; FLEETQ_JOB=<id> ssh tatooine 'FLEETQ_JOB=<id> bash ~/intarsiaml-hipperf/scripts/hip-decode/build.sh' ; fleetq release <id>
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-hipperf}"
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
$D cargo build --release -p intarsia-cli --features hip 2>&1 | grep -E "^(error|warning: unused)|Finished|-->|^\s+\|" | head -60
$D cargo test --release -p intarsia-backend-hip --no-run 2>&1 | grep -E "^error|Finished|-->|^\s+\|" | head -60
