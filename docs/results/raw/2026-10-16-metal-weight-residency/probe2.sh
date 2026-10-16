#!/bin/zsh
# Decay horizon and keep-warm variants. probe2.sh <outdir> <reps> "<mode> <gap_ms> <KEEP_MS> <KEEP_WHAT>"...
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/resid
R=/opt/code/intarsiaml-residency/docs/results/raw/2026-10-16-metal-weight-residency
O=$1; N=$2; shift 2; mkdir -p $O
source $R/waitidle.sh
CFGS=("$@")
for rep in $(seq 1 $N); do for cfg in "${CFGS[@]}"; do
  c=(${=cfg})
  tag=${c[1]}-g${c[2]}-k${c[3]}-${c[4]}-$rep
  wait_idle || touch $O/$tag.contaminated
  KEEP_MS=${c[3]} KEEP_WHAT=${c[4]} $SC/wire_probe /Users/josef/models/Qwen3-8B ${c[1]} 0 ${c[2]} > $O/$tag.txt 2>&1
done; done
echo done
