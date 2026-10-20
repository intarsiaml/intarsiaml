#!/usr/bin/env bash
# Round-6 window 2 driver (Mac side): waits until NOT_BEFORE (HHMMSS, at least 5 min after window 1's vLLM came back), then
# scripts/cuda-prefill4/wC-localeval.sh with the spark1 side scripts/cuda-round6/wB2.sh (chunk-4096 opt-in A/B, 2 rounds,
# then localeval of dev).   NOT_BEFORE=212530 scripts/cuda-round6/w2.sh <log>
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG=${1:?log}
until [ "$(date +%H%M%S)" -ge "${NOT_BEFORE:-0}" ]; do sleep 10; done
ARMS="dev" REMOTE=/home/nvidia/intarsia-decfuse/r6-scripts/scripts/cuda-round6/wB2.sh \
  LABEL="cuda-round6 W2: opt-in chunk 4096 A/B x2 + localeval dev (vLLM down)" OUTNAME=r6C LE_PREFIX=intarsia-cuda-round6 \
  DOWNTIME_LOG="$REPO/docs/results/2026-10-19-cuda-round6/raw/downtime.log" \
  "$REPO/scripts/cuda-prefill4/wC-localeval.sh" 15 "$LOG"
