#!/usr/bin/env bash
# cvatflip churn gate (runs ON the box): what a K / V table "pair" of the captured verify attention (lane cvat) is keyed on, and whether the table grows with the number of requests.
# One `serve --max-batch 1 --spec mtp --spec-k 2` (knob unset = on; INTARSIA_SPEC_TRACE=1 prints `verify attention rows: captured table regions N of 1024` at every capture
# registration), N (default 50) distinct sequential greedy requests (different prompts and lengths, short replies), then the count lines and the engagement notes.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 25 --label "cvatflip churn" -- ssh spark1 'N=50 bash -s' < scripts/cvatflip/churn.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cvatflip; O=${O:-$R/out/churn}; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8299}; N=${N:-50}
nohup bash $R/src/scripts/cvatflip/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!
pat() { echo "$R/bin/[a-z0-9-]+ serve.*--port $PORT"; }
stop() { pkill -f "$(pat)" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "$(pat)" >/dev/null || break; sleep 1; done; }
trap 'stop; kill $WD 2>/dev/null' EXIT
stop
INTARSIA_SPEC_TRACE=1 nohup $BIN serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 \
  --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve.log 2>&1 < /dev/null &
for i in $(seq 1 120); do pgrep -f "$(pat)" >/dev/null || { echo "serve died"; tail -5 $O/serve.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
echo "up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
ok=0
for i in $(seq 1 $N); do
  # a distinct prompt per request (another topic and a varying filler length), greedy, 48 tokens
  body=$(python3 - "$i" <<'PY'
import json, sys
i = int(sys.argv[1])
topics = ["rivers", "chess openings", "bread baking", "tide tables", "volcanoes", "typography", "bee colonies", "rust ownership", "railway timetables", "glaciers"]
p = f"Request {i}: write a short paragraph about {topics[i % len(topics)]}. " + "Context line. " * (i * 7 % 120)
print(json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 48, "temperature": 0, "stream": False}))
PY
)
  curl -sf -m 120 -H 'Content-Type: application/json' -d "$body" http://127.0.0.1:$PORT/v1/chat/completions > $O/r$i.json && ok=$((ok+1))
done
echo "requests ok: $ok of $N"
echo "table size lines (count, line):"; grep -a 'captured table regions' $O/serve.log | sort | uniq -c
echo "first / last: $(grep -a 'captured table regions' $O/serve.log | head -1) / $(grep -a 'captured table regions' $O/serve.log | tail -1)"
echo "notes: $(grep -a 'verify attention rows:' $O/serve.log | grep -av 'table regions' | sort | uniq -c | tr '\n' ';')"
echo "metrics: $(curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec_(steps|accepted)|graph_)' | sed 's/{[^}]*}//' | tr '\n' ' ')"
