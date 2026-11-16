#!/bin/zsh
# Lane mprefill: naxd_bench (scratch path vs in-kernel decode forms per call). Run under fleetq --quiet gpu,cpu.
# usage: bench.sh <out.txt> [VAR=val ...]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mprefill; O=$1; shift
echo "thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}') $(date +%H:%M:%S)" > $O
env X=1 "$@" $W/target/release/examples/naxd_bench >> $O 2>&1
echo "thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}') $(date +%H:%M:%S)" >> $O
