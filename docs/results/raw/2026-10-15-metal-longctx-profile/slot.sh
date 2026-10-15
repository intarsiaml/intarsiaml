#!/bin/zsh
# usage: slot.sh <label> <expect-minutes> <script> [args...]: exclusive endor slot (60 GB), run the script, release on any exit
: ${DEVELOPER_DIR:?}
LABEL=$1; EXPECT=$2; shift 2
JSON=$(fleetq acquire --host endor --exclusive --mem 60 --label "$LABEL" --expect $EXPECT --wait --json)
ID=$(echo "$JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("id") or d.get("job",{}).get("id"))')
echo "fleetq job $ID ($LABEL) acquired $(date)" | tee -a /opt/code/intarsiaml-longctx/docs/results/raw/2026-10-15-metal-longctx-profile/fleetq-jobs.txt
trap "fleetq release $ID; echo fleetq job $ID released \$(date)" EXIT INT TERM
fleetq status --host endor | grep -E "#$ID " | tee -a /opt/code/intarsiaml-longctx/docs/results/raw/2026-10-15-metal-longctx-profile/fleetq-jobs.txt
fleetq status --host endor | grep -E "RUNNING #$ID " >/dev/null || { echo "job $ID not RUNNING, abort"; exit 4; }
FLEETQ_JOB=$ID "$@"
echo "script exit $? $(date)"
