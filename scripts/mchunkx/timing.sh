#!/bin/zsh
# Lane mchunkx: run the per-kernel split timing (tests/rows_exact.rs timing_chunkx) from the copied test binary. Run under fleetq --quiet gpu,cpu.
# usage: timing.sh <tag> [groups comma list]   -> raw/timing/<tag>.log
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mchunkx; O=$W/docs/results/2026-11-17-mchunkx/raw/timing; mkdir -p $O
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
w=0; while [ "$(th)" != 0 ] && [ $w -lt 180 ]; do sleep 5; w=$((w+5)); done; echo "# thermal $(th) after ${w}s wait" > $O/$1.log
CHUNKX_GROUPS=${2:-} $W/.base/bin-tests/rows_exact --ignored --nocapture timing_chunkx 2>&1 | grep -a "CHUNKX\|panicked\|test result" >> $O/$1.log
echo "## done $1 $(grep -c CHUNKX $O/$1.log) lines thermal $(th)"
