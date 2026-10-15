#!/bin/zsh
# usage: slot.sh <label> <expected-minutes> <script> [args...]   -- acquire an exclusive endor fleetq slot, run the script in it, release
LABEL=$1; EXPECT=$2; shift 2
JSON=$(fleetq acquire --host endor --exclusive --mem 24 --label "$LABEL" --expect $EXPECT --wait --json)
ID=$(echo "$JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("id") or d.get("job",{}).get("id"))')
echo "fleetq job $ID ($LABEL) acquired $(date)"
cleanup() { fleetq release $ID; echo "fleetq job $ID released $(date)"; }
trap cleanup EXIT
FLEETQ_JOB=$ID "$@"
echo "script exit $? $(date)"
