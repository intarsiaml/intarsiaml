#!/bin/zsh
# usage: slot.sh "<label>" <command...>
# Acquires an exclusive endor slot, checks that the job is RUNNING before anything runs, runs the command inside it,
# releases via trap (also on error / interrupt).
set -u
LABEL="$1"; shift
R=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g1
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "$DEVELOPER_DIR" ] && [ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR empty or missing" >&2; exit 3; }
OUT=$(fleetq acquire --host endor --exclusive --mem 20 --label "metal spec G1: $LABEL" --wait 2>&1)
echo "$OUT"
ID=$(echo "$OUT" | grep -Eo '#?[0-9]+' | head -1 | tr -d '#')
[ -n "$ID" ] || { echo "no job id" >&2; exit 4; }
trap 'fleetq release '"$ID"' >/dev/null 2>&1; echo "released $ID $(date)" >&2; echo "released $ID $(date)" >> '"$R"'/slots.log' EXIT INT TERM
STATE=$(fleetq show "$ID" 2>&1 | head -1)
echo "$STATE" | grep -q " running on endor" || { echo "job $ID is not running: $STATE" >&2; exit 5; }
echo "slot $ID RUNNING ($LABEL) $(date): $STATE" | tee -a $R/slots.log
export FLEETQ_JOB=$ID
"$@"
