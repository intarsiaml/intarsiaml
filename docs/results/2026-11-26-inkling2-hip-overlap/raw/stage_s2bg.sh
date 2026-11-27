#!/usr/bin/env bash
# Background stage of window S2 (lane ink-hip-overlap) on card 1: what S1 left, then the solo server (no --max-batch) of o4 and main5:
# the 16 real prompts one at a time (serve_bitwise.py solo), compared offline with W4's solo replies.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/s2; mkdir -p "$L"
for b in o4 main5; do
  (( $(left) < 60 )) && { echo "[solo-$b] SKIP ($(left) s left)"; continue; }
  SENV="" WAITS=$(( $(left) - 45 )) srv 1 solo-$b $b --max-seq 4608
  [ "$UP" != 0 ] && bw solo-$b solo
  stop 1
done
echo "[bg] done $(now)"
