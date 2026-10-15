#!/bin/zsh
# F1 (NAX flash prefill attention) speed burst. pre = spike/metal 766c723 binary, new = branch metal-f1-nax-attn. Run inside ONE
# fleetq slot (FLEETQ_JOB set by the caller). ABBA everywhere (pre, new, new, pre) so a drift shows as pre-1 vs pre-2.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-ae91acc25118dab7a
R=$W/docs/results/raw/2026-10-15-metal-prefill-f1/speed
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
NEW=$S/intarsia-new
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
P4K=$PB/prompt-4k.txt
P16K=$PB/b2a/prompt-16k.txt
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/plain $R/trace-new $R/trace-pre $R/localeval
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
binof() { [ $1 = pre ] && echo $PRE || echo $NEW; }
promptof() { [ $1 = 4k ] && echo $P4K || echo $P16K; }

echo "== 0. idle, zoom"; sleep 5; idle3 | tee $R/gpu-idle-start.txt; pgrep -fl zoom.us | tee $R/zoom.txt; echo "zoom check done"

echo "== 0b. attention microbench (examples/flash_bench: per-call GPU time, one KV cache per layer, 36 layers)"
FB=$S/flash_bench
$FB --reps 8 flash nax 0 1024 3840 15872 > $R/bench-pos0-c256.txt 2>&1; cat $R/bench-pos0-c256.txt
$FB --reps 8 --chunk 1024 flash nax 0 1024 3840 15360 > $R/bench-pos0-c1024.txt 2>&1; cat $R/bench-pos0-c1024.txt
for T in 4096 16384; do for C in 256 1024; do $FB --reps 5 --total $T --chunk $C flash nax 2>&1 | grep -E "^total"; done; done | tee $R/bench-total.txt

echo "== 1. plain prefill (generate --max-tokens 1), ABBA, chunk 256 and 1024; first run discarded"
$PRE generate --model $M --backend metal --max-seq 20480 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $P4K)" > $R/plain/warm.out 2> $R/plain/warm.err
for P in 4k 16k; do
  for C in 256 1024; do
    i=0
    for B in pre new new pre; do
      i=$((i+1))
      echo "idle $B $P c$C #$i: $(idle3)" >> $R/plain/idle.txt
      $(binof $B) generate --model $M --backend metal --max-seq 20480 --temperature 0 --max-tokens 1 --profile-every 0 --prefill-chunk $C \
        --prompt "$(cat $(promptof $P))" > $R/plain/$B-$P-c$C-$i.out 2> $R/plain/$B-$P-c$C-$i.err
      echo "$B $P chunk $C #$i: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/plain/$B-$P-c$C-$i.err | tail -1)"
    done
  done
done

echo "== 2. kernel-mode GPU traces (one command buffer per launch)"
for B in new pre; do
  for P in 4k 16k; do
    for C in 256 1024; do
      INTARSIA_METAL_PROFILE=kernel:$R/trace-$B/kernel-$P-c$C.jsonl $(binof $B) generate --model $M --backend metal --max-seq 20480 --temperature 0 \
        --max-tokens 1 --profile-every 0 --prefill-chunk $C --prompt "$(cat $(promptof $P))" > $R/trace-$B/kernel-$P-c$C.out 2> $R/trace-$B/kernel-$P-c$C.err
      python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p2/analyze_prefill.py $R/trace-$B $P $C > $R/trace-$B/analysis-$P-c$C.txt 2>&1
      echo "$B $P c$C: $(grep -E 'attn|total|attention' $R/trace-$B/analysis-$P-c$C.txt | head -3 | tr '\n' ' ')"
    done
  done
done

echo "== 3. localeval speed, one server at a time, pre new new pre (sizes 128,1024,4096)"
PORT=8230
for B in pre new new pre; do
  PORT=$((PORT+1))
  N=$(ls $R/localeval | grep -c "^f1-$B-.*-server.err")
  L=f1-$B-$((N+1))
  $(binof $B) serve --model $M --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 \
    --profile-every 0 --no-supervisor --no-ui > $R/localeval/$L-server.out 2> $R/localeval/$L-server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo "server died"; break; }; sleep 2; done
  sleep 3
  echo "idle before $L: $(idle3)" | tee -a $R/localeval/idle.txt
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 \
    --thinking off --label $L) > $R/localeval/$L-localeval.out 2> $R/localeval/$L-localeval.err
  echo "$L localeval exit $?"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3
done
echo "== end idle"; idle3 | tee $R/gpu-idle-end.txt
echo done
