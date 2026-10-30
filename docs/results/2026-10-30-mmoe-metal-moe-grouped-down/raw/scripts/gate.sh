#!/bin/bash
# usage: gate.sh OUTDIR GV...   (inside a fleetq slot) rows-exact gate with the grouped down on at T>=9, one process per variant
set -u
W=/opt/code/intarsiaml/.worktrees/mmoe; O=$1; shift; mkdir -p $O
BR=$W/.lane/bin-new/rows_exact
for gv in "$@"; do
  INTARSIA_REQUIRE_GPU=1 INTARSIA_METAL_MOE_DN_GROUP=9 INTARSIA_METAL_MOE_DN_GV=$gv $BR --test-threads=1 --nocapture moe_experts_rows_are_bitwise_the_one_token_call > $O/gate-gv$gv.log 2>&1
  echo "gv=$gv exit $? $(grep -m1 'metal moe: expert-grouped' $O/gate-gv$gv.log) | $(grep -E 'test result' $O/gate-gv$gv.log)"
done
