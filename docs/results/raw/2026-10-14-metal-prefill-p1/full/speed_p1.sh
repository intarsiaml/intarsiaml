#!/bin/zsh
# Metal prefill P1, speed burst. Run inside ONE fleetq slot (FLEETQ_JOB set by the caller).
#   cp target/release/intarsia $S/intarsia-new ; cp $S/pre-target/release/intarsia $S/intarsia-pre
# pre = spike/metal 495a447, new = branch metal-p1-gemm. Order everywhere is pre, new, new, pre (ABBA), so a drift in the
# machine (thermal, other tenants) shows up as pre-1 vs pre-2 instead of being credited to the change.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7
R=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
PC=$W/docs/results/raw/2026-10-14-metal-phase-c
NEW=$S/intarsia-new
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/speed/plain $R/speed/trace $R/speed/localeval
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
binof() { [ $1 = pre ] && echo $PRE || echo $NEW; }

echo "== 0. idle"; sleep 5; idle3 | tee $R/speed/gpu-idle-start.txt

echo "== 1. tile pick sweep with the shipped rule"
NAX_ROUNDS=10 $W/target/release/examples/gemm_nax pick > $R/speed/bench-pick.txt 2>&1

echo "== 2. plain prefill (generate --max-tokens 1), chunk 256, ABBA x2 per prompt; first run discarded (page-in)"
$PRE generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $PC/prompt-4k.txt)" > $R/speed/plain/warm.out 2> $R/speed/plain/warm.err
for P in 1k 4k; do
  i=0
  for B in pre new new pre; do
    i=$((i+1))
    echo "idle $B $P #$i: $(idle3)" >> $R/speed/plain/idle.txt
    $(binof $B) generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 \
      --prompt "$(cat $PC/prompt-$P.txt)" > $R/speed/plain/$B-$P-$i.out 2> $R/speed/plain/$B-$P-$i.err
    echo "$B $P #$i: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/speed/plain/$B-$P-$i.err | tail -1)"
  done
done

echo "== 3. kernel-mode GPU traces (one command buffer per launch), new binary, chunk 256"
for P in 1k 4k; do
  INTARSIA_METAL_PROFILE=kernel:$R/speed/trace/kernel-$P-c256.jsonl $NEW generate --model $M --backend metal --max-seq 8192 --temperature 0 \
    --max-tokens 1 --profile-every 0 --prompt "$(cat $PC/prompt-$P.txt)" > $R/speed/trace/kernel-$P-c256.out 2> $R/speed/trace/kernel-$P-c256.err
  python3 -I $PC/analyze_prefill.py $R/speed/trace $P 256 > $R/speed/trace/analysis-$P-c256.txt 2>&1
done

echo "== 4. localeval speed, one server at a time, pre new new pre"
PORT=8110
for B in pre new new pre; do
  PORT=$((PORT+1))
  N=$(ls $R/speed/localeval | grep -c "^p1-$B-.*-server.err")
  L=p1-$B-$((N+1))
  $(binof $B) serve --model $M --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 \
    --profile-every 0 --no-supervisor --no-ui > $R/speed/localeval/$L-server.out 2> $R/speed/localeval/$L-server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo "server died"; break; }; sleep 2; done
  sleep 3
  echo "idle before $L: $(idle3)" | tee -a $R/speed/localeval/idle.txt
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 \
    --thinking off --label $L) > $R/speed/localeval/$L-localeval.out 2> $R/speed/localeval/$L-localeval.err
  echo "$L localeval exit $?"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3
done
echo "== end idle"; idle3 | tee $R/speed/gpu-idle-end.txt
echo done
