#!/bin/zsh
# Lane mfp8split: per-kernel split timing of the fp8 rows kernel (tests/rows_exact.rs timing_chunkx, groups fp8_35b and fp8_head_27b) for M = 2..13,
# every split. Run under fleetq --quiet gpu,cpu. usage: timing.sh <tag> [split env value]  -> raw/timing/<tag>.log
# (`cur` is the M-row call through `launch`: the old halves with split env 0, the plan with 1.)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mfp8split; O=$W/docs/results/2026-11-17-mfp8split/raw/timing; mkdir -p $O
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
w=0; while [ "$(th)" != 0 ] && [ $w -lt 180 ]; do sleep 5; w=$((w+5)); done; echo "# thermal $(th) after ${w}s wait" > $O/$1.log
INTARSIA_METAL_FP8_ROWS_SPLIT=${2:-1} CHUNKX_GROUPS=fp8_35b,fp8_head_27b CHUNKX_MAXM=13 $W/.base/bin-tests/rows_exact --ignored --nocapture timing_chunkx 2>&1 | grep -a "CHUNKX\|panicked\|test result" >> $O/$1.log
echo "## done $1 $(grep -c CHUNKX $O/$1.log) lines thermal $(th)"
