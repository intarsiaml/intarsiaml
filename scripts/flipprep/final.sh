#!/usr/bin/env bash
# flipprep final check on the merged head (main + #380 + #385 + the flip): build, then proof.sh with arms C (recipe defaults, no env) and R (--spec off).
# fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 35 --label "flipprep: merged-head check" -- scripts/flipprep/final.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
D="$(dirname "${BASH_SOURCE[0]}")"; REPO="$(git -C "$D" rev-parse --show-toplevel)"
REVS="comb=${HEAD_REV:-flipprep-35b}" "$D/build.sh" || exit 1
ARMS="C R" OUT="$REPO/docs/results/2026-11-15-flipprep/raw/proof-merged" "$D/proof.sh"
