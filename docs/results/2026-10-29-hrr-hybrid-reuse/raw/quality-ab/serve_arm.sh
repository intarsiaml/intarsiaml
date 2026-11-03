#!/bin/bash
# serve_arm.sh ARM(A|B) -> starts serve (8093-like ports) + proxy; prints PIDs to .scratch/qa/arm-ARM.pids
cd /opt/code/intarsiaml/.worktrees/hrr
ARM=$1; KIND=$2; SP=$((19100+$(printf %d "'$ARM")))
if [ $ARM = B ]; then export INTARSIA_PREFIX_REUSE=exact; else unset INTARSIA_PREFIX_REUSE; fi
.scratch/target/release/intarsia serve --model ~/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --max-seq 20000 --spec off --port $SP --host 127.0.0.1 --no-ui --served-model-name intarsia-metal > .scratch/qa/serve-$ARM-$KIND.log 2>&1 &
S=$!
python3 .scratch/qa/proxy.py $((SP+100)) $SP .scratch/qa/reqlog-$ARM-$KIND.jsonl &
P=$!
echo "$S $P $((SP+100))" > .scratch/qa/arm-$ARM-$KIND.pids
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$((SP+100))/v1/models >/dev/null && break; sleep 3; done
echo "up $ARM serve=$S proxy=$P port=$((SP+100))"
wait
