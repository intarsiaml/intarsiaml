#!/usr/bin/env bash
# Card 1 of window S8 (lane ink-hip-overlap2): hip tiny suites on the o6 tree (inkling2_batch, inkling2_packed, inkling2), then
# o6's solo server (W4's config: --max-seq 4608 --spec off, no --max-batch; its first start admits the P12 plan) with the 16 real
# prompts one at a time, compared with W4's solo replies afterwards.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib2.sh"
L=$REPO/logs/s8; mkdir -p "$L"
tiny 1 o6 inkling2_batch inkling2_packed inkling2
if (( $(left) > 90 )); then
  WAITS=$(( $(left) - 45 )) srv 1 solo-o6 o6 --max-seq 4608 --spec off
  [ "$UP" != 0 ] && bw solo-o6 solo
fi
stop 1
echo "[bg] done $(now)"
