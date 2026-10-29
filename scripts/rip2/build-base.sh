#!/usr/bin/env bash
# Lane rip2 (copied from scripts/rip): build origin/main (the A/B base) on spark2 under ~/intarsiaml-rip2/base -> ~/intarsiaml-rip2/bin/base (--load cpu):
#   fleetq run --host spark2 --load cpu --mem 8 --expect 30 --label "rip2: build base" -- scripts/rip2/build-base.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
H=${HOST:-spark2}; R="~/intarsiaml-rip2/base"; REF=${REF:-origin/main}
RPATH='set -o pipefail; export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SHA=$(git -C "$REPO" rev-parse "$REF")
rsh "mkdir -p $R ~/intarsiaml-rip2/bin; test -d $R/target || cp -a ~/intarsiaml-rip2/src/target $R/target; find $R -mindepth 1 -maxdepth 1 ! -name target -exec rm -rf {} +"
git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$H" "tar -x -C intarsiaml-rip2/base"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.toml' \) -print0 | xargs -0 touch; env INTARSIA_BUILD_STAMP=ref-${SHA:0:12} CARGO_BUILD_JOBS=16 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -4"
rsh "cp $R/target/release/intarsia ~/intarsiaml-rip2/bin/base; ~/intarsiaml-rip2/bin/base --version | head -1"
