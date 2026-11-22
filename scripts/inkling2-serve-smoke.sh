#!/bin/bash
# one fleetq slot: Metal server on the real checkpoint, the API smoke, then the localeval smoke cells (inside this slot)
cd /opt/code/intarsiaml/.worktrees/ink-serve
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
PORT=8097
./target/release/intarsia serve --model "$M" --port $PORT --served-model-name inkling-2-mini --no-supervisor > .scratch/server.log 2>&1 &
SP=$!
for i in $(seq 1 200); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -20 .scratch/server.log; exit 1; }
  sleep 3
done
echo "server up after $((i*3)) s"
python3 tools/inkling2_serve_smoke.py http://127.0.0.1:$PORT/v1 .scratch/smoke-out.json
( set -a; . ~/.secrets.env 2>/dev/null; set +a; cd /opt/code/deepeval
  for S in none json_schema; do
    uv run localeval quality --url http://127.0.0.1:$PORT/v1 --model inkling-2-mini --bench gsm8k,ifeval --limit 2 --structured $S --judge none --no-unim --label ink-serve-metal-smoke-$S --recipe "intarsia serve --model Inkling-2-Mini --port $PORT (ink/serve, Metal)" 2>&1 | tail -25
  done )
kill $SP; wait $SP 2>/dev/null
tail -5 .scratch/server.log
