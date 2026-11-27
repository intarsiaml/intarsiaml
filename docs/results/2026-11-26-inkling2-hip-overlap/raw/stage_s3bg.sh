#!/usr/bin/env bash
# Background stage of window S3 (lane ink-hip-overlap) on card 1: the pace trade-off (the coordinator's ask: TTFT p50 vs ITL p99 at
# 2 or 3 settings), on the merged head o5 (revised 21:10: o5 instead of o4, so the same runs are the merged head's batched bitwise
# check): the real prompts at 8 (wave 0) and 16 at once, plain (replies compared with W4's solo replies): main5, then o5 at pace 0,
# 150 (default) and 300 ms. Card 1 beside card 0's visits: compare these arms with each other, not with card 0's numbers.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/s3; mkdir -p "$L"
for arm in "main5;;m" "o5;INTARSIA_OVERLAP_PACE_MS=0;p0" "o5;;p150" "o5;INTARSIA_OVERLAP_PACE_MS=300;p300"; do
  IFS=';' read -r b env tag <<< "$arm"
  (( $(left) < 50 )) && { echo "[pace-$tag] SKIP ($(left) s left)"; continue; }
  SENV="$env" WAITS=40 srv 1 pace-$tag $b --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  [ "$UP" != 0 ] && { bw pace-$tag conc:8:0 conc; curl -sf -m 5 "http://$IP:8133/metrics" > "$L/pace-$tag-metrics-end.txt" 2>/dev/null; echo "[pace-$tag] $(grep -E '^intarsia_batch_overlap_steps_total ' "$L/pace-$tag-metrics-end.txt")"; }
  stop 1
done
echo "[bg] done $(now)"
