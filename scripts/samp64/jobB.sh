#!/usr/bin/env bash
# samp64 phase 2: existing knobs on the shipping config: --spec-k 1, INTARSIA_SPEC_K_MIN=1 (adaptive depth 1..2), INTARSIA_SPEC_COSTGATE=1. Sampled and greedy, 4k/32k/64k.
D="$(dirname "${BASH_SOURCE[0]}")"
C="0.7:4096,32768,65536:5 0:4096,32768,65536:5"
bash "$D/window.sh" w3 "--spec-k 1" - $C
bash "$D/window.sh" w4 "" "INTARSIA_SPEC_K_MIN=1" $C
bash "$D/window.sh" w5 "" "INTARSIA_SPEC_COSTGATE=1" $C
