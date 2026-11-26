#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job: build main's (80719483a, .scratch/main-src) tiny test target `inkling2` in the lane's
# volume, so window 5 can run main's tiny suite beside the branch's (window 4). Prints the test executable.
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-verify"
cd "$LANE"; mkdir -p logs
export INTARSIA_REPO=$LANE/.scratch/main-src INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-verify INTARSIA_CONTAINER_NAME=ink-hip-verify-build
find "$INTARSIA_REPO/crates" "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/recipes" -type f -exec touch {} +
"$LANE/deploy/tatooine/dev.sh" cargo test --release -j8 -p intarsia-models --test inkling2 --no-run < /dev/null > logs/build-main-tiny.log 2>&1
echo "main tiny build exit $?"; tail -2 logs/build-main-tiny.log
