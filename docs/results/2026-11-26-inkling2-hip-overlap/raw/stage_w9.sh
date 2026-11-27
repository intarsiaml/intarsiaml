#!/usr/bin/env bash
# Window W9 (lane ink-hip-overlap2), the stage window.sh runs under CAP=80 (radiance budget: 2:08 left of 16:00): the Inkling HIP
# bitwise items the regate windows did not cover on the merged head o6, both cards at once.
#   card 0: o6 INTARSIA_OVERLAP=1 INTARSIA_OVERLAP_PACE_MS=0, batched (--max-batch 16 --max-seq 4608 --kv-fraction 0.9): 8 and 16 at once
#   card 1: o6 INTARSIA_OVERLAP=1 INTARSIA_OVERLAP_PACE_MS=300, same: 16 at once; then the hip tiny suite inkling2 on the o6 tree
# Replies compared afterwards with W4's solo replies (the first lane's S3 card-1 pace check, on o6).
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib2.sh"
L=$REPO/logs/w9; mkdir -p "$L"
(
  SENV="INTARSIA_OVERLAP=1 INTARSIA_OVERLAP_PACE_MS=0" WAITS=35 srv 0 pace-p0-o6 o6 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  [ "$UP" != 0 ] && { BW_CPUS=10,11,22,23 bw pace-p0-o6 conc conc:8:0; curl -sf -m 5 "http://$IP:8133/metrics" | grep -E '^intarsia_batch_overlap_steps_total ' | sed 's/^/[pace-p0-o6] /'; }
  stop 0
) > "$L/card0.out" 2>&1 &
(
  SENV="INTARSIA_OVERLAP=1 INTARSIA_OVERLAP_PACE_MS=300" WAITS=35 srv 1 pace-p300-o6 o6 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  [ "$UP" != 0 ] && { bw pace-p300-o6 conc; curl -sf -m 5 "http://$IP:8133/metrics" | grep -E '^intarsia_batch_overlap_steps_total ' | sed 's/^/[pace-p300-o6] /'; }
  stop 1
  tiny 1 o6 inkling2
) > "$L/card1.out" 2>&1 &
wait
cat "$L/card0.out" "$L/card1.out"
echo "[w9] done $(now)"
