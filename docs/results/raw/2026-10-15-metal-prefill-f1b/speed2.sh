#!/bin/zsh
# F1b speed burst part 2 (the first burst's sections 2, 3, 3b were skipped: the helper scripts were not executable), plus an
# alternating microbench of F1 and F1b and the final-tree test run. ONE fleetq slot.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/f1b
W=/opt/code/intarsiaml/.claude/worktrees/agent-afe24a73270ed0c31
RAW=$W/docs/results/raw/2026-10-15-metal-prefill-f1b
R=$RAW/speed
mkdir -p $R $RAW/gates/tests
export INTARSIA_CPU_FREE_MEM_GB=60
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
date > $R/start2.txt
echo "== 0. idle, zoom"; sleep 5; idle3 | tee $R/gpu-idle-start2.txt; pgrep -fl zoom.us | tee -a $R/zoom.txt; echo "zoom check done"

echo "== 1. alternating microbench F1 / F1b (pos0 3840 chunk 256; pos0 4096 and 14336 chunk 2048), 3 rounds each"
for r in 1 2 3; do
  for B in f1 final; do
    FB=$S/flash_bench-$B
    $FB --reps 8 nax 3840 > $R/bench2-$B-c256-r$r.txt 2>&1
    $FB --reps 8 --chunk 2048 nax 4096 14336 > $R/bench2-$B-c2048-r$r.txt 2>&1
    echo "$B r$r: c256 $(tail -1 $R/bench2-$B-c256-r$r.txt | tr -s ' ') | c2048 $(tail -2 $R/bench2-$B-c2048-r$r.txt | tr -s ' ' | tr '\n' ';')"
  done
done

echo "== 2. kernel-mode GPU traces (TFLOPS per kernel class)"
$RAW/trace.sh speed/trace new=$S/intarsia-final pre=$S/intarsia-base f1=$S/intarsia-f1

echo "== 3. plain prefill ABBA (4 rounds = 8 runs per binary and config), pre vs new, chunk 256 and 2048, 4k and 16k"
$RAW/plain-abba.sh speed/plain "256,2048" "4k,16k" 4 A=$S/intarsia-base B=$S/intarsia-final
python3 -I $RAW/summarize_plain.py $R/plain pre new | tee $R/plain-summary.txt
echo "== 3b. plain prefill ABBA, f1 vs new (2 rounds)"
$RAW/plain-abba.sh speed/plain-f1 "256,2048" "4k,16k" 2 A=$S/intarsia-f1 B=$S/intarsia-final
python3 -I $RAW/summarize_plain.py $R/plain-f1 f1 new | tee $R/plain-f1-summary.txt

echo "== 4. final-tree tests (flash_prefill*, metal4_fallback, full metal crate)"
cd $W
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $RAW/gates/tests/cargo-test-metal-final.txt 2>&1; echo "cargo test metal exit $?"
grep -E "^test result|FAILED|panicked" $RAW/gates/tests/cargo-test-metal-final.txt | head -30
cargo test -p intarsia-backend-metal --release --test flash_prefill_nax -- --test-threads=1 --nocapture > $RAW/gates/tests/flash_prefill_nax-final.txt 2>&1; echo "flash_prefill_nax exit $?"
grep -E "worst error|numerics|test result" $RAW/gates/tests/flash_prefill_nax-final.txt | head -30
echo "== end idle"; idle3 | tee $R/gpu-idle-end2.txt
date >> $R/start2.txt
echo done
