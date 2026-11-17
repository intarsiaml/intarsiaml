#!/bin/zsh
# Lane msm192 (from scripts/msmallm/bench.sh): naxd_bench (scratch path vs in-kernel decode forms per call), per-round times too.
# Run under fleetq --quiet gpu,cpu. usage: bench.sh <out.txt> [VAR=val ...]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/msm192; O=$1; shift
COOL=${COOL:-0}; w=0; th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
while [ $(th) != 0 ] && [ $w -lt $COOL ]; do sleep 5; w=$((w+5)); done   # COOL=<s>: wait up to <s> for thermal pressure 0 first
echo "cooled ${w}s" > $O
echo "thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}') $(date +%H:%M:%S)" >> $O
env X=1 "$@" $W/target/base/naxd_bench >> $O 2>&1
echo "thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}') $(date +%H:%M:%S)" >> $O
