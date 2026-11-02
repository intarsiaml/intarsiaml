#!/bin/bash
# usage: hrr_run.sh LABEL PORT [ENV=VAL]  (runs from worktree root)
cd /opt/code/intarsiaml/.worktrees/hrr
L=$1; P=$2; shift 2
env "$@" .scratch/target/release/intarsia serve --model ~/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --max-seq 20000 --spec off --port $P --host 127.0.0.1 --no-ui > .scratch/serve-$L.log 2>&1 &
S=$!
for i in $(seq 1 100); do curl -sf http://127.0.0.1:$P/v1/models >/dev/null && break; sleep 3; done
python3 .scratch/hrr_reuse_probe.py $P .scratch/probe-$L.json $PROBE_ARGS 2>&1 | tee .scratch/probe-$L.txt
kill $S; wait $S 2>/dev/null
