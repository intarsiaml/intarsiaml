#!/bin/bash
# usage: [EXTRA="--max-batch 2"] serve.sh BIN TAG   (run on the Spark under one fleetq slot; kills the server at the end)
cd ~/intarsia-stream-burst
SNAP=$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
bin/$1 serve $EXTRA --backend cuda --model $SNAP --port 8193 --max-seq 32768 --mem-budget 40 > out/$2.server.log 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do a=$(free -g | awk 'NR==2{print $7}'); [ "$a" -lt 10 ] && { echo LOWMEM $a; kill -9 $SP; }; sleep 1; done ) &
for i in $(seq 1 200); do curl -sf localhost:8193/v1/models >/dev/null && break; sleep 2; done
cd out
python3 sse_probe.py 8193 $2-warm --max 32 >/dev/null
for r in 1 2; do
python3 sse_probe.py 8193 $2-nothink-$r --max 400
python3 sse_probe.py 8193 $2-think-$r --think --max 400
python3 sse_probe.py 8193 $2-tools-$r --tools --max 300
done
kill $SP; wait $SP 2>/dev/null; free -g | sed -n 2p
