#!/usr/bin/env bash
# Lane S1: rebuild s1-dev on HOST (default spark2) and run `intarsia kernels test --backend cuda` (full and --quick), plus the
# new-kernel GPU tests, inside a fleetq run. Output -> docs/results/2026-10-22-s1-spark-prefill/raw/ktest-<host>.out
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H=${HOST:-spark2}
OUT="$S/../../docs/results/2026-10-22-s1-spark-prefill/raw/ktest-$H.out"
{
  HOST=$H STEPS=dev bash "$S/build.sh"
  ssh -n -o BatchMode=yes $H "export FLEETQ_JOB=$FLEETQ_JOB OUTNAME=s1K STEPS='tests ktest' BENCH_CFGS='0'; bash ~/intarsia-decfuse/s1-dev/scripts/s1/remote.sh"
  ssh -n -o BatchMode=yes $H "~/intarsia-decfuse/s1-dev/target/acc/release/intarsia kernels test --backend cuda --quick 2>&1 | grep -aE 'gemm_reg|glu_gemm|cublas|kernels,|FAIL|NO CASES|SKIPPED' | head -20"
} 2>&1 | tee "$OUT"
