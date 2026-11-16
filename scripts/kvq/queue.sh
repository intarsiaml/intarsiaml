#!/bin/bash
# Lane kvq: the four measuring jobs, one fleetq slot each (null test first). bash 3.2 safe.
W=/opt/code/intarsiaml/.worktrees/kvq
R=$W/docs/results/2026-11-16-kvq/raw
FQ=${FQ:-fleetq}
cd "$W" || exit 1
for cell in "35 off 3 40" "35 default 3 45" "27 default 2 80" "27 off 2 75"; do
  set -- $cell
  m=$1; spec=$2; reps=$3; mins=$4
  $FQ run --host endor --quiet gpu,cpu --expect "$mins" --mem 32 --label "kvq: ABBA-BAAB ${m}B spec=$spec" -- \
    python3 scripts/kvq/ab.py --model "$m" --spec "$spec" --reps "$reps" --out "$R/ab-$m-$spec" > "$R/ab-$m-$spec.log" 2>&1
  echo "job $m $spec exit $?" >> "$R/queue.log"
done
echo "## queue done" >> "$R/queue.log"
