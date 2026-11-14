#!/bin/bash
# usage: lan.sh BIN TAG   (on spark1, one fleetq slot): serve on 0.0.0.0, probe from spark2 over the LAN, then kill the server
cd ~/intarsia-stream-burst
SNAP=$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
bin/$1 serve --max-batch 2 --profile-every 0 --host 0.0.0.0 --backend cuda --model $SNAP --port 8193 --max-seq 32768 --mem-budget 40 > out/$2.server.log 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do a=$(free -g | awk 'NR==2{print $7}'); [ "$a" -lt 10 ] && { echo LOWMEM $a; kill -9 $SP; }; sleep 1; done ) &
for i in $(seq 1 200); do curl -sf localhost:8193/v1/models >/dev/null && break; sleep 2; done
P="cd ~/intarsia-stream-burst-probe && python3 sse_probe.py 8193 $2"
ssh spark2 "$P-warm --host spark1 --max 32" >/dev/null
for r in 1 2 3; do
ssh spark2 "$P-nothink-$r --host spark1 --max 400"
ssh spark2 "$P-think-$r --host spark1 --think --max 400"
done
kill $SP; wait $SP 2>/dev/null
