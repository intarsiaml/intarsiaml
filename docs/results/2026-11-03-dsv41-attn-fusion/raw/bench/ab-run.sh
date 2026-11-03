#!/bin/bash
# Alternating A/B of the unmodified attn_sparse_mma (base) and the ATTN_2BLK variant (var): PAIRS pairs, base first then var, each process runs the
# three shapes (+ T37, T11 for hashes) with REPS reps of ITERS launches. Run inside a fleetq slot (--quiet gpu,cpu) after ab-build.sh.
set -eu
OUT=${OUT:-$HOME/intarsia-dsv41-attnfuse/bench/ab}; PAIRS=${PAIRS:-6}; ITERS=${ITERS:-20}; REPS=${REPS:-15}
for i in $(seq 1 "$PAIRS"); do
  for v in base var; do
    echo "## pair $i $v $(date +%T)"
    "$OUT/ab_$v" "$ITERS" "$REPS"
  done
done
