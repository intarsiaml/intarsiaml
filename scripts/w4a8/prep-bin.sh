#!/usr/bin/env bash
# Lane W4A8, ON tatooine (cpu job, after build.sh): copy the release CLI out of the cargo target volume into the repo copy, where
# cat-window.sh's serve containers find it as /work/w4a8-intarsia.
set -euo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-w4a8 INTARSIA_TARGET_VOL=intarsia-cargo-w4a8
"$INTARSIA_REPO/deploy/tatooine/dev.sh" cp /target/release/intarsia /work/w4a8-intarsia
ls -l "$INTARSIA_REPO/w4a8-intarsia"
