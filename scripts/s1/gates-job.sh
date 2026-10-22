#!/usr/bin/env bash
# Lane S1 gates on spark1 (vLLM up, non-exclusive), as one fleetq run with --expect (backfill):
#   fleetq run --host spark1 --mem 40 --expect 105 --label "..." -- scripts/s1/gates-job.sh
# build s1-base (origin/main export) + s1-dev, then on the box: kernel tests, kernels test, nsys 4k prefill per arm, greedy
# ids + plans, 35B KL21 accuracy + prefix reuse per arm (scripts/s1/remote.sh). Output -> docs/results/.../raw/gates-*.out
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$S/../../docs/results/2026-10-22-s1-spark-prefill/raw"; mkdir -p "$LOG"
echo "== build $(date +%T)"
HOST=spark1 SEED_TREE=s1-dev CARGO_BUILD_JOBS=12 bash "$S/build.sh" 2>&1 | tee "$LOG/gates-build.out"
echo "== remote $(date +%T)"
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB OUTNAME=s1G STEPS='${GSTEPS:-tests ktest prof ids acc}'; bash ~/intarsia-decfuse/s1-dev/scripts/s1/remote.sh" 2>&1 | tee "$LOG/gates-remote.out"
echo "== done $(date +%T)"
