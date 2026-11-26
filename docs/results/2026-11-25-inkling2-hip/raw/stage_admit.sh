#!/usr/bin/env bash
# Stage: admission of the given ops on HIP against the cpu reference (`intarsia kernels test`). Usage: stage_admit.sh op [op ..]
set -uo pipefail
R=$HOME/intarsiaml-ink-hip/docs/results/2026-11-25-inkling2-hip/raw
fail=0
for op in "$@"; do
  echo "== kernels test --backend hip --op $op ($(date '+%T'))"
  bash "$R/run.sh" /target/release/intarsia kernels test --backend hip --op "$op" 2>&1 | tail -25
  rc=${PIPESTATUS[0]}
  echo "== $op exit $rc"
  [ "$rc" -ne 0 ] && fail=1
done
exit $fail
