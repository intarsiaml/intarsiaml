#!/usr/bin/env bash
# Lane mtp-snap-gpu (GPU check of PR #395 on HIP, 2026-11-16): helpers ON tatooine (sourced). Tree ~/intarsiaml-mtpsnap-gpu with its own cargo
# target volume; the two binaries it builds are /target/intarsia-pr and /target/intarsia-main in that volume.
TREE=${TREE:-$HOME/intarsiaml-mtpsnap-gpu}
export INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-mtpsnap-gpu}
D="$TREE/deploy/tatooine/dev.sh"
STAMP="$(cat "$TREE/.stamp")"
