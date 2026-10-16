#!/bin/zsh
# Continue inside the already-held fleetq job (slot.sh of job 2009 was killed without releasing so that the lease survives), release on exit.
: ${DEVELOPER_DIR:?}
ID=$1; shift
trap "fleetq release $ID; echo fleetq job $ID released \$(date)" EXIT INT TERM
fleetq status --host endor | grep -E "RUNNING #$ID " || { echo "job $ID not RUNNING, abort"; exit 4; }
FLEETQ_JOB=$ID "$@"
echo "script exit $? $(date)"
