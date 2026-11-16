#!/usr/bin/env bash
# Lane gdn27-speed: helpers ON tatooine (sourced). Tree ~/intarsiaml-gdn27-speed with its own cargo target volume.
TREE=${TREE:-$HOME/intarsiaml-gdn27-speed}
export INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-gdn27-speed} INTARSIA_MODELS=${INTARSIA_MODELS:-/mnt/models}
D="$TREE/deploy/tatooine/dev.sh"
STAMP="$(cat "$TREE/.stamp")"
