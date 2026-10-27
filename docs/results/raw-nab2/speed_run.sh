#!/bin/sh
# one fleetq slot: start the intarsia server, localeval speed (128/1k/4k) + MMLU limit 10, stop the server
W=/opt/code/intarsiaml/.worktrees/nab2
echo "FLEETQ_JOB=$FLEETQ_JOB"
$W/target/release/intarsia serve --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --max-seq 6144 --port 8123 --served-model-name lfm2.5-1.2b-instruct --no-ui > $W/raw/nab2/serve.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8123/v1/models >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:8123/v1/models | head -c 300; echo
R="intarsia serve --model LiquidAI/LFM2.5-1.2B-Instruct --backend metal --max-seq 6144 (nab2 branch lane-nab2/lfm2-shortconv)"
cd /opt/code/deepeval
uv run localeval speed --url http://127.0.0.1:8123/v1 --model lfm2.5-1.2b-instruct --sizes 128,1024,4096 --gen 128 --reps 3 --thinking off --label nab2-lfm2-metal --recipe "$R" 2>&1 | tail -40
uv run localeval quality --url http://127.0.0.1:8123/v1 --model lfm2.5-1.2b-instruct --bench mmlu --limit 10 --thinking off --label nab2-lfm2-metal-mmlu10 --recipe "$R" 2>&1 | tail -30
kill $SP; wait $SP 2>/dev/null
