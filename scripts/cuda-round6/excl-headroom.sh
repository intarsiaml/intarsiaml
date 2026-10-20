#!/usr/bin/env bash
# Round-6 headroom slot (Mac side): ONE exclusive fleetq slot on spark1 with vLLM up and idle (no --takes-down), then
#   1. nsys prefill profile of base (r6-base, chunk 2048) at ~1k / 4k / 16k tokens, and of chunk 4096 at 4k / 16k
#      (scripts/cuda-round6/prof-prefill.sh);
#   2. kernel microbenchmarks on r6-dev (moe at T 1024..8192, fp8 projections, GDN, tensor-core / DRAM-run roofs;
#      scripts/cuda-round6/dev.sh, ROUNDS=2);
#   3. vLLM's Marlin NvFp4 MoE at the same shapes inside the vLLM container (scripts/cuda-round6/marlin-bench.sh);
# and releases the slot on exit.   scripts/cuda-round6/excl-headroom.sh 2>&1 | tee <log>
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB=""
cleanup() { [[ -n "$JOB" ]] && { echo "$(date +%T) releasing $JOB"; fleetq release "$JOB" >/dev/null; }; }
trap cleanup EXIT INT TERM
AQ=$(fleetq acquire --host spark1 --exclusive --mem 34 --expect 30 --label "cuda-round6: headroom (prefill nsys 1k/4k/16k, kernel microbench, vLLM Marlin MoE bench; vLLM up, idle)" --wait 2>&1); rc=$?
echo "$AQ" | tail -3
# only a granted lease counts: a cancelled or dropped request exits non-zero and must not run anything
[[ $rc == 0 ]] || { echo "acquire failed (exit $rc): not running"; exit 1; }
JOB=$(echo "$AQ" | grep -Eo '#[0-9]+' | head -1 | tr -d '#')
[[ -n "$JOB" ]] || { echo "no job id"; exit 1; }
fleetq status --host spark1 | grep -q "#$JOB .*" || { echo "job $JOB not on spark1"; JOB=""; exit 1; }
export FLEETQ_JOB=$JOB
echo "$(date +%T) lease $JOB held"
STEP=${STEP:-prof,dev,marlin}
if [[ ",$STEP," == *",prof,"* ]]; then
  REMOTE_ENV="TREE=r6-base ARMS=base:INTARSIA_X=0 CHARS='3400 14000 54000'" "$S/sync-scripts.sh" scripts/cuda-round6/prof-prefill.sh
  REMOTE_ENV="TREE=r6-base ARMS=c4096:INTARSIA_X=0 CHUNK=4096 CHARS='14000 54000'" "$S/sync-scripts.sh" scripts/cuda-round6/prof-prefill.sh
fi
if [[ ",$STEP," == *",dev,"* ]]; then
  TREE=${DEV_TREE:-r6-exp} NOBUILD=1 STEPS=${DEV_STEPS:-moe,wq,gdn,peak} TESTS="${TESTS:---test fp8_gemm_reg --test gdn_split}" TEST_ENV="${TEST_ENV:-INTARSIA_FP8_REG_CFG=1}" MOE_T=1024,2048,4096,8192 WQ_M=1024,2048,4096 ROUNDS=2 WQ_ENVS="${WQ_ENVS:-INTARSIA_FP8_REG_CFG=0;INTARSIA_FP8_REG_CFG=1;INTARSIA_FP8_REG_CFG=2;INTARSIA_FP8_REG_CFG=3}" GDN_ENVS="${GDN_ENVS:-INTARSIA_GDN_EXP=0;INTARSIA_GDN_EXP=4;INTARSIA_GDN_EXP=8;INTARSIA_GDN_EXP=2}" "$S/dev.sh"
fi
if [[ ",$STEP," == *",marlin,"* ]]; then
  "$S/marlin-bench.sh" --tokens 1024,2048,4096,8192 --reps 10
fi
echo "$(date +%T) done"
