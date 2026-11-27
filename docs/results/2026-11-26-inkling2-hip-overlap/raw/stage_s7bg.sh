#!/usr/bin/env bash
# Card 1 of window S7 (lane ink-hip-overlap2): Inkling-2 greedy md5 (short / 4k x 64 / 256), o6 and main6, `--spec off` and the
# DSpark default; then G2 short / long for o6 and main6.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib2.sh"
L=$REPO/logs/s7; mkdir -p "$L"
md5s 1 o6 "--spec off"; md5s 1 o6 ""; md5s 1 main6 "--spec off"; md5s 1 main6 ""
stop 1
for b in o6 main6; do for g in short long; do g2 1 $b $g; done; done
stop 1
echo "[bg] done $(now)"
