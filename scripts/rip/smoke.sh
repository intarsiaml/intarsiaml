#!/usr/bin/env bash
# Lane rip: a < 3 min smoke of the job scripts' untried parts (spark2, --quiet gpu,cpu): one 8B server start, one greedy.py rep at
# 1 stream, one tiny localeval call (checks it runs inside this slot), one `plan show` of each binary.
#   OUT=... fleetq run --host spark2 --quiet gpu,cpu --mem 30 --expect 4 --label "rip: smoke" -- scripts/rip/smoke.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2; HN=spark-fa00; PORT=8197
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:?}; mkdir -p "$OUT"
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
B=/home/nvidia/intarsiaml-rip/bin
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
for t in base intarsia; do rsh "$B/$t plan show --backend cuda --model $M8 --max-seq 8192 --mem-budget 40 < /dev/null 2>&1 | tail -3"; done
SPID=$(rsh "nohup $B/intarsia serve --backend cuda --model $M8 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 > ~/intarsiaml-rip/out/serve-smoke.log 2>&1 < /dev/null & echo \$!")
trap 'rsh "kill $SPID 2>/dev/null; sleep 3; kill -9 $SPID 2>/dev/null"' EXIT
for i in $(seq 1 90); do curl -sf "http://$HN:$PORT/health" >/dev/null && break; sleep 2; done
python3 "$D/../bi8/greedy.py" run "http://$HN:$PORT/v1" 1 1 "$OUT/greedy-smoke.json" > /dev/null && python3 -c "import json; d=json.load(open('$OUT/greedy-smoke.json')); print('greedy.py ok', list(d)[:3], len(next(iter(d.values()))))"
(cd /opt/code/deepeval && uv run localeval speed --url "http://$HN:$PORT/v1" --model intarsia --thinking off --sizes 128 --gen 8 --reps 1 --label rip-smoke --no-unim --recipe smoke 2>&1 | tail -6)
rsh "grep -iE 'row-invariant|error' ~/intarsiaml-rip/out/serve-smoke.log | head -3"
echo "== smoke done"
