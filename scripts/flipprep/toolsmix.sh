#!/usr/bin/env bash
# flipprep: a tool request (constrained by the tools grammar, so it never drafts) beside a plain chat request (which drafts), 2 streams, greedy,
# against servers A (main defaults) and F (flip defaults): the replies of both must be identical across arms.
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 20 --label "flipprep: tools + plain mixed pairs" -- scripts/flipprep/toolsmix.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=${OUT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-15-flipprep/raw/gates}
LOGNAME_=toolsmix; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"; source "$S/arms.sh"
for a in ${TARMS:-A F}; do
  arm_setup $a
  if start_serve "toolsmix-$a" "$ENVS" --max-batch 2 $ARGS; then
    python3 "$S/tools.py" --url "$URL" --model guidance --arm "$a" --out "$OUT/tools" --metrics "http://$HN:$PORT/metrics" --temperature 0 --streams 2 --mix 2>&1 | tee -a "$LOG"
    finish_serve "toolsmix-$a"
  fi
done
