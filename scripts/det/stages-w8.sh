#!/usr/bin/env bash
# Lane det w8 (ON tatooine, radiance down): confirm the P2P all-reduce with the per-row tap. w7's tap arm caught 3 events in 200 fresh mtp processes,
# each one rank only, a few rows, at layer 2 of the 38-row prefill (the other rank equal): a one-sided wrong read in the all-reduce. w7's
# AR_DRAIN=5 (0/200) and --comm host (0/150) arms had no tap, so a shift that did not flip a token was invisible there. Same arms, tap on.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w8 || exit 2
M="--spec mtp --spec-k 2"
TAP=1 MT=16 E="INTARSIA_HIP_AR_DRAIN=5" loop drain5 0,1 300 --tp 2 --devices 0,1 $M
TAP=1 MT=16 loop host 0,1 200 --tp 2 --devices 0,1 --comm host $M
log "w8 done"
