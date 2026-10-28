#!/usr/bin/env bash
# Lane asp (ON tatooine, radiance down): plan show of the 3 models (default env and the lane's knobs off must print the same), and the
# model-level GPU tests that exercise the speculative verify, rollback, batching and tp=2 paths of this lane's change.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-asp/.asp/lib.sh"
D=dev
clear "gates" || exit 2
plan "$D" 0 8b "$M8"
plan "$D" 0,1 27b-tp2 "$M27" $TP2
plan "$D" 0 35b "$M35"
for t in batch batch_recurrent mtp hybrid fusion_p2 graphs hybrid_mxfp4 spec; do
  mtest "$D" "$t" 0 900 --test "$t"
done
for t in tp_hybrid tp_mtp tp_batch tp; do
  mtest "$D" "$t" 0,1 1200 --test "$t"
done
log "gates done"
