#!/usr/bin/env bash
# dv27 smoke: (1) does the recipe default (no spec flags) equal the explicit V65536 flag set, (2) is acceptance recorded on the single-stream path,
# (3) is seeded T=1.0 identical between V65536 and V0 (full head) on that path, (4) does --max-batch 2 + the opt-in env draft on the 27B.
set -uo pipefail
D=$(dirname "${BASH_SOURCE[0]}")
export OUT=$(git -C "$D" rev-parse --show-toplevel)/docs/results/2026-11-15-dv27/raw/smoke GEN=128
LOGNAME_=smoke-s1 MODE=s1 ARMS="D.s V65536.s V0.s" CELLS="en:default:1 cjk:off:1" "$D/sweep.sh"
LOGNAME_=smoke-b2 MODE=b2 ARMS="V65536.s V0.s" CELLS="en:default:1" STREAMS=2 "$D/sweep.sh"
