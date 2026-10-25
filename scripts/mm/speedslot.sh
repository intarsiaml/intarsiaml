#!/bin/bash
# Lane mm speed comparison inside one exclusive fleetq slot (fleetq run gives FLEETQ_JOB): localeval main then branch, then the
# in-process real-traffic step time A B A B, then the branch's per-kernel B = 8 profile. Usage: speedslot.sh MAIN_BIN BRANCH_BIN
# MAIN_TEST_BIN BRANCH_TEST_BIN OUT
set -u
MAIN=$1; BR=$2; TM=$3; TB=$4; OUT=$5; mkdir -p "$OUT"
cd /opt/code/wt-mm
scripts/mm/speed.sh "$MAIN" mm-main-2c2d9941 > "$OUT/localeval-main.log" 2>&1
scripts/mm/speed.sh "$BR" mm-branch > "$OUT/localeval-branch.log" 2>&1
export INTARSIA_QWEN36_DIR=$HOME/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_REQUIRE_GPU=1
for rep in 1 2; do for arm in main branch; do
  b=$TM; [ $arm = branch ] && b=$TB
  echo "== $arm rep$rep"; $b --ignored --nocapture --test-threads=1 real_35b_batch_step_real_traffic 2>&1 | grep -E "REALTRAFFIC|panicked"
done; done > "$OUT/step-time.log"
INTARSIA_METAL_PROFILE=kernel:$OUT/k8-branch.jsonl MM_BATCHES=8 MM_STEPS=8 $TB --ignored --nocapture --test-threads=1 real_35b_batch_step_real_traffic > /dev/null 2>&1
python3 scripts/mm/kprof.py $OUT/k8-branch.jsonl 8 8 > "$OUT/k8-branch.txt"
gzip -f $OUT/k8-branch.jsonl
