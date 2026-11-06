#!/usr/bin/env bash
# Lane lmhead-fp8-default, ONE HIP window on tatooine (radiance down for the ids runs only; the two builds are cpu work before hdrive stops radiance):
#   scripts/lmhead/hwin.sh        (acquires the slot itself, releases it on every exit path; hdrive.sh restores radiance and checks it answers)
# Builds the head tree (this worktree) and the base tree (.scratch/base = main af6aa5226) on tatooine, then at tp2 with the recipe's MTP k2 default:
#   texts (18 prompts x 256 greedy tokens, md5) of base (no env), base + INTARSIA_LMHEAD_QUANT=fp8 (the opt-in lane's behaviour), head (no env = the new
#   default), head + INTARSIA_LMHEAD_QUANT=none (the off switch); then `plan show` / `kernels test` base vs head (hgates.sh).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."
OUTS=$(fleetq acquire --host tatooine --takes-down radiance --mem 30 --quiet gpu,cpu --expect ${EXPECT:-55} --label "lmhead-fp8-default: HIP window (builds, 27B ids default vs fp8 vs off, plan/kernels gates; radiance down <= 55 min)" --wait)
echo "$OUTS"; J=$(echo "$OUTS" | grep -oE '#[0-9]+' | head -1 | tr -d '#'); [ -n "$J" ] || exit 1; echo "job=$J"
export FLEETQ_JOB=$J W=def CAP=${CAP:-2400}
trap 'fleetq release $J' EXIT
export OUT=$PWD/docs/results/2026-11-06-lmhead-fp8-default/raw/hip-window
mkdir -p "$OUT"
scripts/lmhead/tbuild.sh lmhead > "$OUT/build-head.log" 2>&1; echo "build head exit $?"
scripts/lmhead/tbuild.sh lmhbase "$PWD/.scratch/base" > "$OUT/build-base.log" 2>&1; echo "build base exit $?"
export ITEMS="base.off.2.mtp.1:texts base.fp8.2.mtp.1:texts head.off.2.mtp.1:texts head.none.2.mtp.1:texts head.off.2.mtp.1:gates"
scripts/lmhead/hdrive.sh
echo "driver exit $?"
