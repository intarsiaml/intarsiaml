#!/bin/bash
# Lane mm speed arm (run inside a held fleetq slot: FLEETQ_JOB set). Usage: scripts/mm/speed.sh BIN LABEL [--max-batch N]
# Starts `intarsia serve` (35B, metal) on 127.0.0.1:18431, runs localeval speed (1k prompt, 512 generated, 3 reps,
# 1 / 4 / 8 / 16 streams), stops the server.
set -u
BIN=$1; LABEL=$2; MB=${3:-8}
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; PORT=18431
CMD="$BIN serve --backend metal --model $M --max-batch $MB --served-model-name intarsia --max-seq 8192 --host 127.0.0.1 --port $PORT"
$CMD > /opt/code/wt-mm/.mm/serve-$LABEL.log 2>&1 &
SP=$!
for i in $(seq 1 180); do curl -s -m 2 http://127.0.0.1:$PORT/v1/models > /dev/null && break; sleep 1; done
echo "server up after ${i}s (pid $SP)"
cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024 --gen 512 --reps 3 \
  --concurrency 4,8,16 --label "$LABEL" --recipe "$CMD" 2>&1 | tail -30
kill $SP; wait $SP 2>/dev/null
