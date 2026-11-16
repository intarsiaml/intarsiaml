#!/usr/bin/env bash
# dv27 smoke 2: does --max-batch 2 + INTARSIA_BATCHED_MTP=2 (no batched verify) draft on the 27B at 2 streams, and are replies identical V65536 vs V0?
set -uo pipefail
D=$(dirname "${BASH_SOURCE[0]}")
export OUT=$(git -C "$D" rev-parse --show-toplevel)/docs/results/2026-11-15-dv27/raw/smoke GEN=128
LOGNAME_=smoke-b2n MODE=b2n ARMS="V65536.s V0.s" CELLS="en:default:1" "$D/sweep.sh"
