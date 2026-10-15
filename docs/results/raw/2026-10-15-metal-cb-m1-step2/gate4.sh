#!/bin/zsh
# cb-m1 step 2 gate 4 (throughput), one exclusive endor slot: usage gate4.sh <outdir> <rev>
# localeval 0.3.0 (feat/decode-aggregate): speed at ctx 1k, --gen 256, N = 1 (the sweep) and 2,4,8 streams, server --profile-every 0,
# --max-batch 1 (LocalEngine) vs --max-batch 8. Inside the slot: FLEETQ_JOB is inherited by localeval.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; REV=$2; mkdir -p $O
HERE=${0:A:h}
source $HERE/serve_lib.sh
for MB in ${MBS:-1 8}; do
  start_server mb$MB 8201 --max-batch $MB || exit 3
  (cd /opt/code/deepeval-decode-agg && uv run localeval speed --url http://127.0.0.1:8201/v1 --model Qwen3-8B --sizes 1024 --gen 256 --reps 3 --concurrency 2,4,8 --label cbwire-mb$MB) > $O/localeval-mb$MB.log 2>&1; echo "localeval mb$MB: exit $?"
  curl -s http://127.0.0.1:8201/metrics > $O/metrics-mb$MB.txt
  stop_server
done
