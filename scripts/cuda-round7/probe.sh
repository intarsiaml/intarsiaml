#!/usr/bin/env bash
# Round 7, one small slot (vLLM up): rebuild r7-dev and run the e4m3 GEMM tests (cuBLASLt default accumulation, then
# INTARSIA_LT_FAST_ACCUM=1) with the lt bench under both, then build r7-base from BASE_SRC (a plain export of origin/main).
#   BASE_SRC=<dir> fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round7/probe.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
STEPS=test "$S/dev.sh"
echo "======== FAST_ACCUM=1"
NOBUILD=1 TENV=INTARSIA_LT_FAST_ACCUM=1 STEPS=test,wq WQ_KERN=matmul_fp8_gemm_lt WQ_ENVS="INTARSIA_X=0;INTARSIA_LT_FAST_ACCUM=1" "$S/dev.sh"
if [[ -n "${BASE_SRC:-}" ]]; then
  echo "======== build r7-base from $BASE_SRC"
  ssh -n -o BatchMode=yes spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -c 'mkdir -p ~/intarsia-decfuse/r7-base; test -d ~/intarsia-decfuse/r7-base/target || cp -a ~/intarsia-decfuse/r7-dev/target ~/intarsia-decfuse/r7-base/target'"
  SRC_DIR="$BASE_SRC" INTARSIA_STAMP="head=4be7b7d4 (origin/main export) code-tree=base" "$REPO/scripts/cuda-decode/build.sh" r7-base 2>&1 | grep -E "^error|Finished|intarsia$|stamp"
fi
