#!/usr/bin/env bash
# Window 4 stage (lane ink-hip-pf2): main2 (origin/main 40b85a391, the new prefill defaults) and pf2d (the branch merged on it),
# 4k / 16k prefill, one process per configuration, at the defaults: main2; pf2d INTARSIA_HIP_INK_PF2 0 / 3 / 7 / 19 / 23 (the
# default) / 23 with ZP 8, 32, 64 / 151 (bit 128 on); pf2d default and main2 again. Card 0. Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pf2
S=$REPO/.scratch
M=/models/Inkling-2-Mini
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
P16="$P4
$P4
$P4
$P4"
G="generate -m $M --backend hip --max-seq 20480 --spec off --ignore-eos --profile-every 0"
now() { date '+%T'; }
line() { grep -aoE "prefill [0-9]+ ms \([0-9.]+ tok/s\)|Error.*|panicked.*" | head -2 | tr '\n' ' '; }
for cfg in "main2 INK=main" "pf2d INTARSIA_HIP_INK_PF2=0" "pf2d INTARSIA_HIP_INK_PF2=3" "pf2d INTARSIA_HIP_INK_PF2=7" "pf2d INTARSIA_HIP_INK_PF2=19" \
           "pf2d INTARSIA_HIP_INK_PF2=23" "pf2d INTARSIA_HIP_INK_PF2_ZP=8" "pf2d INTARSIA_HIP_INK_PF2_ZP=32" "pf2d INTARSIA_HIP_INK_PF2_ZP=64" \
           "pf2d INTARSIA_HIP_INK_PF2=151" "pf2d INK=default" "main2 INK=main"; do
  set -- $cfg
  for p in 4k 16k; do
    [ $p = 4k ] && PR="$P4" || PR="$P16"
    echo "GEN $p $cfg | $(bash "$S/run.sh" env $2 /work/.scratch/bin/intarsia-$1 $G --max-tokens 2 -p "$PR" 2>&1 | line) [$(now)]"
  done
done
echo "[stage] done $(now)"
