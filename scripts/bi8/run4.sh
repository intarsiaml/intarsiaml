#!/usr/bin/env bash
# Lane bi8 short check (spark2, --quiet gpu,cpu, < 3 min on the GPU): the merged tree (after #229) served greedy vs the cold reference,
# --max-batch 16, 1 / 4 / 8 / 16 streams x 2 reps on one server.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2; HN=spark-fa00; PORT=8197
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:?}; mkdir -p "$OUT"; REF=${REF:?}
M=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
SPID=$(rsh "nohup ~/intarsiaml-bi8/bin/intarsia serve --backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 36 --profile-every 0 --max-batch 16 > ~/intarsiaml-bi8/out/serve-merged.log 2>&1 < /dev/null & echo \$!")
for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && break; sleep 2; done
python3 "$D/greedy.py" run "http://$HN:$PORT/v1" 1,4,8,16 2 "$OUT/greedy-mb16-merged.json" > /dev/null
python3 "$D/greedy.py" compare "$REF" "$OUT/greedy-mb16-merged.json" | tee "$OUT/greedy-mb16-merged.compare" | tail -1
