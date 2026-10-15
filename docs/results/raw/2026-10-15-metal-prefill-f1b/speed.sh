#!/bin/zsh
# F1b speed burst, ONE fleetq slot. pre = spike/metal 6271eec, f1 = F1 + merge, new = branch tip.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/f1b
W=/opt/code/intarsiaml/.claude/worktrees/agent-afe24a73270ed0c31
RAW=$W/docs/results/raw/2026-10-15-metal-prefill-f1b
R=$RAW/speed
mkdir -p $R
export INTARSIA_CPU_FREE_MEM_GB=60
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
date > $R/start.txt
echo "== 0. idle, zoom"; sleep 5; idle3 | tee $R/gpu-idle-start.txt; pgrep -fl zoom.us | tee $R/zoom.txt; echo "zoom check done"

echo "== 1. attention microbench (GPU time per call, one KV cache per layer, 36 layers); P2 = flash, F1 = flash-bench of 5c30da0, F1b = final"
for B in f1 final; do
  FB=$S/flash_bench-$B
  $FB --reps 8 flash nax 0 1024 3840 15872 > $R/bench-$B-pos0-c256.txt 2>&1; echo "-- $B chunk 256"; cat $R/bench-$B-pos0-c256.txt
  $FB --reps 8 --chunk 2048 flash nax 0 2048 4096 14336 > $R/bench-$B-pos0-c2048.txt 2>&1; echo "-- $B chunk 2048"; cat $R/bench-$B-pos0-c2048.txt
  for T in 4096 16384; do for C in 256 2048; do $FB --reps 5 --total $T --chunk $C flash nax 2>&1 | grep -E "^total" | sed "s/^/$B /"; done; done | tee $R/bench-$B-total.txt
done

echo "== 2. kernel-mode GPU traces (TFLOPS per kernel class)"
$RAW/trace.sh speed/trace new=$S/intarsia-final pre=$S/intarsia-base f1=$S/intarsia-f1

echo "== 3. plain prefill ABBA (8 runs per binary and config), pre vs new, chunk 256 and 2048, 4k and 16k"
$RAW/plain-abba.sh speed/plain "256,2048" "4k,16k" 4 A=$S/intarsia-base B=$S/intarsia-final
echo "== 3b. plain prefill ABBA, f1 vs new (4k and 16k, chunk 256 and 2048, 2 rounds)"
$RAW/plain-abba.sh speed/plain-f1 "256,2048" "4k,16k" 2 A=$S/intarsia-f1 B=$S/intarsia-final

echo "== 4. decode unchanged: tok/s and step period (cb trace, 129 tokens)"
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
M=/Users/josef/models/Qwen3-8B
mkdir -p $R/decode
for P in short 4k; do
  for B in pre new new pre; do
    BIN=$S/intarsia-final; [ $B = pre ] && BIN=$S/intarsia-base
    F=$R/decode/$B-$P-$RANDOM
    INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
      --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $PB/prompt-$P.txt)" > $F.out 2> $F.err
    echo "$B $P: $(python3 -I $PB/b2b/period.py $F.jsonl 364 128 2>&1 | tail -1) $(grep -o '([0-9.]* tok/s)' $F.err | tail -1)"
  done
done
echo "== end idle"; idle3 | tee $R/gpu-idle-end.txt
date >> $R/start.txt
echo done
