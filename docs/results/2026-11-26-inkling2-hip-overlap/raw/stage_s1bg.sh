#!/usr/bin/env bash
# Background stage of window S1 (lane ink-hip-overlap) on card 1 beside card 0's paired speed visits: the md5 cells O4 left
# (o4 spec off, main5 DSpark), then G2 short / long for o4 and main5, then gate 1's poisoned run on o4 (INTARSIA_HIP_POISON=ff:
# every new device allocation 0xff, NaN in bf16 / f32; overlap on by default) at 16 at once and 4 x 4.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/s1; mkdir -p "$L"
md5s 1 o4 "--spec off"
md5s 1 main5 ""
stop 1
for b in o4 main5; do for g in short long; do g2 1 $b $g; done; done
stop 1
if (( $(left) > 60 )); then
  SENV="INTARSIA_HIP_POISON=ff" WAITS=$(( $(left) - 40 )) srv 1 poison-o4 o4 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  [ "$UP" != 0 ] && bw poison-o4 conc conc:4:0 conc:4:1 conc:4:2 conc:4:3
fi
stop 1
echo "[bg] done $(now)"
