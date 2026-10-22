#!/usr/bin/env bash
# Lane S1: the whole spark2 exclusive job, from this Mac:
#   fleetq run --host spark2 --exclusive --takes-down retro --mem 100 --expect 75 --label "..." -- scripts/s1/window-job.sh
# 1. build s1-base / s1-dev on spark2 (vLLM still up), 2. the vLLM-down window (scripts/s1/window.sh: 00:00-07:00 EDT,
# agent-harness check, vLLM restored in the EXIT trap) running scripts/s1/wALL.sh with localeval on base / dev / cub.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$S/../../docs/results/2026-10-22-s1-spark-prefill/raw"; mkdir -p "$LOG"
echo "== build $(date +%T)"
HOST=spark2 bash "$S/build.sh" 2>&1 | tee "$LOG/w-build.out"
ssh -n spark2 'for t in s1-base s1-dev; do test -x ~/intarsia-decfuse/$t/target/acc/release/intarsia || exit 1; done' || { echo "build missing"; exit 1; }
echo "== window $(date +%T)"
LOCALEVAL_ARMS="base dev cub" bash "$S/window.sh" ${WIN_MIN:-55} "S1 spark2 A/B + localeval (base/dev/cub)" '~/intarsia-decfuse/s1-dev/scripts/s1/wALL.sh' 2>&1 | tee "$LOG/w-window.out"
