#!/bin/bash
# usage: gates.sh OUTDIR  (inside a fleetq slot): full rows_exact suite default / grouped on; negative controls
set -u
W=/opt/code/intarsiaml/.worktrees/mmoe; O=$1; mkdir -p $O
BR=$W/.lane/bin-new/rows_exact
export INTARSIA_REQUIRE_GPU=1
run() { local tag=$1; shift; env "$@" $BR --test-threads=1 --nocapture > $O/$tag.log 2>&1; echo "$tag exit $? | $(grep -E '^test result' $O/$tag.log)"; }
runx() { local tag=$1 t=$2; shift 2; env "$@" $BR --test-threads=1 --nocapture --exact $t > $O/$tag.log 2>&1; echo "$tag exit $? (want non-zero) | $(grep -m1 -E 'MoE|T=[0-9]+ routing' $O/$tag.log | cut -c1-200)"; }
G=moe_experts_rows_are_bitwise_the_one_token_call
run suite-default X=1
run suite-grouped-gv0 INTARSIA_METAL_MOE_DN_GROUP=9 INTARSIA_METAL_MOE_DN_GV=0
run suite-grouped-gv1 INTARSIA_METAL_MOE_DN_GROUP=9 INTARSIA_METAL_MOE_DN_GV=1
run suite-grouped-gv4 INTARSIA_METAL_MOE_DN_GROUP=9 INTARSIA_METAL_MOE_DN_GV=4
run suite-grouped-from-2 INTARSIA_METAL_MOE_DN_GROUP=2 INTARSIA_METAL_MOE_DN_GV=1
runx neg-moe_dn_g $G INTARSIA_METAL_MOE_DN_GROUP=9 INTARSIA_METAL_NEGCTL=moe_dn_g
runx neg-moe_dn $G X=1 INTARSIA_METAL_NEGCTL=moe_dn
runx neg-moe_gu $G X=1 INTARSIA_METAL_NEGCTL=moe_gu
runx neg-moe_dn-grouped $G INTARSIA_METAL_MOE_DN_GROUP=9 INTARSIA_METAL_NEGCTL=moe_dn
