#!/usr/bin/env bash
# Round-6 mutation check on spark1 (vLLM up, small slot), inside the caller's fleetq slot:
#   fleetq run --host spark1 --mem 6 --label "..." -- scripts/cuda-round6/mut.sh
# Seeds ~/intarsia-decfuse/r6-mbase from r6-dev's target dir, then scripts/cuda-prefill4/mutcheck.sh twice with
# scripts/cuda-round6/mutate.py: the flash mutants (tests/flash_bq.rs + tests/attention.rs, racecheck on the flash lines of
# race-targets.txt) and the MoE-threshold mutants (the backend's lib tests: the registry pins; no racecheck).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; D=~/intarsia-decfuse; mkdir -p \$D/r6-mbase; test -d \$D/r6-mbase/target || cp -a \$D/r6-dev/target \$D/r6-mbase/target"
P=$S/mutate.py
FLM=$(python3 -I "$P" --list attention_flash.cu); QMM=$(python3 -I "$P" --list quantmoe.rs)
echo "===== flash mutants: $FLM"
MUT_PY=$P KFILE=kernels/cuda/attention_flash.cu TESTS="--test flash_bq --test attention" MIN_PASS=3 BASE_TREE=r6-mbase MUT_TREE=r6-mut \
  "$REPO/scripts/cuda-prefill4/mutcheck.sh" $FLM
r1=$?
echo "===== MoE threshold mutants: $QMM"
MUT_PY=$P KFILE=crates/intarsia-backend-cuda/src/quantmoe.rs TESTS="--lib" MIN_PASS=13 BASE_TREE=r6-mbase MUT_TREE=r6-mut RACECHECK=0 NOBUILD=1 \
  "$REPO/scripts/cuda-prefill4/mutcheck.sh" $QMM
r2=$?
echo "===== mutcheck exits: flash $r1, threshold $r2"
