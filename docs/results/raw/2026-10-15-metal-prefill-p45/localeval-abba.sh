#!/bin/zsh
# localeval speed, one server at a time, pre new new pre, inside the caller's fleetq slot (FLEETQ_JOB set).
#   localeval-abba.sh <outdir-name> <prefill chunk>
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p45
W=/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00
OUT=$W/docs/results/raw/2026-10-15-metal-prefill-p45/$1
CHUNK=$2
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $OUT
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
pgrep -fl zoom.us > $OUT/zoom.txt 2>&1
PORT=8160
for B in pre new new pre; do
  PORT=$((PORT+1))
  BIN=$S/intarsia-t2
  [ $B = pre ] && BIN=$S/intarsia-pre
  N=$(ls $OUT | grep -c "^p45-c$CHUNK-$B-.*-server.err")
  L=p45-c$CHUNK-$B-$((N+1))
  $BIN serve --model $M --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 \
    --prefill-chunk $CHUNK --profile-every 0 --no-supervisor --no-ui > $OUT/$L-server.out 2> $OUT/$L-server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo "server died"; break; }; sleep 2; done
  sleep 3
  echo "idle before $L: $(idle3)" | tee -a $OUT/idle.txt
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 \
    --thinking off --label $L) > $OUT/$L-localeval.out 2> $OUT/$L-localeval.err
  echo "$L localeval exit $?"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3
done
echo done
