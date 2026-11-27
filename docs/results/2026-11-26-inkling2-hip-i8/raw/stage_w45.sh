#!/usr/bin/env bash
# Windows 4 / 5 (lane ink-hip-i8): stage_speed.sh once per prompt with its own rep list, so the radiance budget can give 16k fewer reps.
# env: SPEC=off|default; PLAN="4k:1 2 3 4,16k:1 2" (prompt:reps, comma separated)
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-i8/docs/results/2026-11-26-inkling2-hip-i8/raw
IFS=',' read -ra items <<< "${PLAN:?PLAN}"
for it in "${items[@]}"; do
  p=${it%%:*}; reps=${it#*:}
  SPEC=${SPEC:-off} PROMPTS=$p REPS="$reps" bash "$R/stage_speed.sh"
done
echo "[stage] all done $(date '+%T')"
