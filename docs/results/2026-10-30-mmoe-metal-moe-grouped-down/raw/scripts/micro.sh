#!/bin/bash
# usage: micro.sh OUTDIR CONFIG...   CONFIG = off | <gv>   (inside a quiet fleetq slot). One process per config.
set -u
W=/opt/code/intarsiaml/.worktrees/mmoe; O=$1; shift; mkdir -p $O
BR=$W/.lane/bin-new/rows_exact
for c in "$@"; do
  if [[ $c == off* ]]; then E="INTARSIA_METAL_MOE_DN_GROUP=0"; else E="INTARSIA_METAL_MOE_DN_GROUP=9 INTARSIA_METAL_MOE_DN_GV=$c"; fi
  env INTARSIA_REQUIRE_GPU=1 $E $BR --ignored --nocapture --test-threads=1 timing_moe_rows16 > $O/wall-$c.log 2>&1
  env INTARSIA_REQUIRE_GPU=1 $E INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/ce1-$c.jsonl $BR --ignored --nocapture --test-threads=1 timing_moe_rows16 > $O/ce1-$c.log 2>&1
  echo "## $c"; grep TIMING $O/wall-$c.log
done
