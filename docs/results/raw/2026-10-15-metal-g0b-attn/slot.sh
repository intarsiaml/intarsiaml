#!/bin/zsh
# usage: slot.sh "<label>" <command...>   (acquires an exclusive endor slot, runs the command inside it, releases via trap)
set -u
LABEL="$1"; shift
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "$DEVELOPER_DIR" ] && [ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR empty or missing" >&2; exit 3; }
OUT=$(fleetq acquire --host endor --exclusive --mem 20 --label "metal G0b attn: $LABEL" --wait 2>&1)
echo "$OUT"
ID=$(echo "$OUT" | grep -Eo '#?[0-9]+' | head -1 | tr -d '#')
[ -n "$ID" ] || { echo "no job id" >&2; exit 4; }
fleetq show "$ID" 2>&1 | head -1 | grep -q " running on endor" || { echo "job $ID is not running (cancelled?): $(fleetq show $ID 2>&1 | head -1)" >&2; exit 5; }
trap 'fleetq release '"$ID"' >/dev/null 2>&1; echo "released $ID" >&2' EXIT INT TERM
echo "slot $ID $(date)" >> /opt/code/intarsiaml-g0b/docs/results/raw/2026-10-15-metal-g0b-attn/slots.log
export FLEETQ_JOB=$ID
"$@"
