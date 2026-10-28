#!/usr/bin/env bash
# Lane det w9 (ON tatooine, radiance down): candidate fixes for the stale all-reduce read, per-row tap on. INTARSIA_HIP_AR_DRAIN=6 (new:
# every thread reads back its last pushed word before the flag) and =1 (the existing system fence in every thread). Baseline (mode 3,
# w7): 3 events in 200 tapped fresh mtp processes.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w9 || exit 2
M="--spec mtp --spec-k 2"
TAP=1 MT=16 E="INTARSIA_HIP_AR_DRAIN=6" loop drain6 0,1 250 --tp 2 --devices 0,1 $M
TAP=1 MT=16 E="INTARSIA_HIP_AR_DRAIN=1" loop drain1 0,1 250 --tp 2 --devices 0,1 $M
log "w9 done"
