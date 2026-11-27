#!/usr/bin/env bash
# Background stage of window S3 (lane ink-hip-overlap) on card 1: the pace trade-off on o4 (the coordinator's ask: TTFT p50 vs ITL
# p99 at 2 or 3 settings), the real prompts at 8 (wave 0) and 16 at once, plain (replies compared with solo): main5, then o4 at pace
# 0, 150 (default) and 300 ms. Card 1 beside card 0's visits: compare these arms with each other, not with card 0's numbers.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/s3; mkdir -p "$L"
for arm in "main5;;m" "o4;INTARSIA_OVERLAP_PACE_MS=0;p0" "o4;;p150" "o4;INTARSIA_OVERLAP_PACE_MS=300;p300"; do
  IFS=';' read -r b env tag <<< "$arm"
  (( $(left) < 50 )) && { echo "[pace-$tag] SKIP ($(left) s left)"; continue; }
  SENV="$env" WAITS=40 srv 1 pace-$tag $b --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  [ "$UP" != 0 ] && bw pace-$tag conc:8:0 conc
  stop 1
done
echo "[bg] done $(now)"
