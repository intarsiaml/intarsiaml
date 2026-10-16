#!/bin/zsh
# Idle-gap probe over the weight-residency candidates. probe.sh <outdir> <reps> <modes...>; gaps 0.3/3/10/40 s after the mode's setup.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/resid
R=/opt/code/intarsiaml-residency/docs/results/raw/2026-10-16-metal-weight-residency
O=$1; N=$2; shift 2; mkdir -p $O
source $R/waitidle.sh
for rep in $(seq 1 $N); do for mode in "$@"; do for gap in 300 3000 10000 40000; do
  wait_idle || touch $O/$mode-g$gap-$rep.contaminated
  $SC/wire_probe /Users/josef/models/Qwen3-8B $mode 0 $gap > $O/$mode-g$gap-$rep.txt 2>&1
done; done; done
echo done
