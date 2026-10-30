#!/bin/bash
# usage: h1.sh OUTDIR CONFIG...  harness (batch_metal_real real-traffic) B=9,12,16, ce1 per-op, one process per config (off | gv). inside a quiet fleetq slot
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mmoe; O=$1; shift; mkdir -p $O
BR=$W/.lane/bin-new/batch_metal_real
export MM_BATCHES=9,12,16 MM_STEPS=32 INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/models/Qwen3.6-35B-A3B-NVFP4
for c in "$@"; do
  if [[ $c == off* ]]; then E="INTARSIA_METAL_MOE_DN_GROUP=0"; else E="INTARSIA_METAL_MOE_DN_GROUP=9 INTARSIA_METAL_MOE_DN_GV=$c"; fi
  echo "## $c $(date +%T)"
  env X=1 $E $BR --ignored real_35b_batch_step_real_traffic --nocapture --test-threads=1 > $O/plain-$c.log 2>&1; grep REALTRAFFIC $O/plain-$c.log
  env X=1 $E INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/ce1-$c.jsonl $BR --ignored real_35b_batch_step_real_traffic --nocapture --test-threads=1 > $O/ce1-$c.log 2>&1; grep REALTRAFFIC $O/ce1-$c.log
done
