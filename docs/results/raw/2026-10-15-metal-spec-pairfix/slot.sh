#!/bin/zsh
# usage: slot.sh "<step>" <command...>   exclusive endor slot, RUNNING check, release via trap
set -u
STEP="$1"; shift
R=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-pairfix
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "$DEVELOPER_DIR" ] && [ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR empty or missing" >&2; exit 3; }
OUT=$(fleetq acquire --host endor --exclusive --mem 20 --label "metal spec pair fix" --wait 2>&1)
echo "$OUT"
ID=$(echo "$OUT" | grep -Eo '#?[0-9]+' | head -1 | tr -d '#')
[ -n "$ID" ] || { echo "no job id" >&2; exit 4; }
trap 'fleetq release '"$ID"' >/dev/null 2>&1; echo "released $ID $(date)" >&2; echo "released $ID ($STEP) $(date)" >> '"$R"'/slots.log' EXIT INT TERM
STATE=$(fleetq show "$ID" 2>&1 | head -1)
echo "$STATE" | grep -q " running on endor" || { echo "job $ID is not running: $STATE" >&2; exit 5; }
echo "slot $ID RUNNING ($STEP) $(date): $STATE" | tee -a $R/slots.log
export FLEETQ_JOB=$ID
"$@"
