#!/bin/bash
# runs ON spark2 inside the slot: cuda_repro.sh LABEL BIN [ENV=VAL]; probe from $HOME/intarsia-hrr/probe
L=$1; BIN=$2; shift 2
D=$HOME/intarsia-hrr/probe; cd $D
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
P=$((18700 + RANDOM % 100))
env "$@" $BIN serve --model $M35 --backend cuda --max-seq 20000 --spec off --port $P --host 127.0.0.1 --no-ui > serve-$L.log 2>&1 &
S=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$P/v1/models >/dev/null && break; sleep 3; done
PROBE_ARGS=ab python3 hrr_reuse_probe.py $P probe-$L.json ab 2>&1 | tee probe-$L.txt
kill $S; wait $S 2>/dev/null
