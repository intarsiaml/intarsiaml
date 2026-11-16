#!/usr/bin/env bash
# flipprep: the same 24 tool requests with `tool_choice: none` (the tools array is in the prompt, the reply is not constrained, so it DRAFTS): MTP acceptance on
# tool-flavoured traffic, where <tool_call> / <think> / <|im_end|> (ids >= 248044) can never be drafted by a 131072-row head. Arms A (main), F (flip), F0 (flip, full head), greedy, 1 stream.
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 25 --label "flipprep: tool_choice none acceptance" -- scripts/flipprep/toolsnone.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=${OUT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-15-flipprep/raw/gates}
LOGNAME_=toolsnone; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"; source "$S/arms.sh"
for a in ${TARMS:-A F F0}; do
  arm_setup $a
  if start_serve "toolsnone-$a" "$ENVS" --max-batch 2 $ARGS; then
    python3 "$S/tools.py" --url "$URL" --model guidance --arm "$a" --out "$OUT/tools" --metrics "http://$HN:$PORT/metrics" --temperature 0 --streams 1 --tool-choice none 2>&1 | tee -a "$LOG"
    finish_serve "toolsnone-$a"
  fi
done
