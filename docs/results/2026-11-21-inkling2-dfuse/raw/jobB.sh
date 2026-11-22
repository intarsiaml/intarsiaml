#!/bin/sh
# B: the per-step gates on the real model. Fused == unfused bit for bit at 1 row (greedy decode, 64 tokens, text md5) and at
# 16 rows (G2-short prompts prefilled in 16-row chunks: every pass runs the 16-row exact_rows paths; self-golden md5), for each
# switch alone and all three; kernel-mode traces prove the fused kernels ran; then G2 / G3 with all on (the no-metal4 fallback
# is tests/inkctrl_nometal4.rs: without the metal4.0 library the model itself does not load, InkInProj prefill needs NAX).
# pinned: builds the snapshot worktree .scratch/snap (detached at the tested commit), not the dev tree
cd /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/snap || exit 1
[ -d target ] || cp -c -R /opt/code/intarsiaml/.worktrees/ink-dfuse/target target
L=/opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/B; mkdir -p $L
cargo build --release --bin intarsia --features metal > $L/build_b.log 2>&1 || { echo "BUILD FAILED"; exit 1; }
# fail fast: the new kernels' bitwise tests (no stress) before any long run
cargo test --release -p intarsia-backend-metal --test inkctrl --test inkctrl_nometal4 -- --skip ink_route_stress --skip blend_tail_stress > $L/failfast.log 2>&1 || { echo "FAILFAST inkctrl tests failed"; tail -30 $L/failfast.log; exit 1; }
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
G2S=golden/oracle/inkling2-mini-oracle-w4a16-short-top20.safetensors
ON="INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_BLEND_TAIL=1"
sw() { case $1 in off) echo "INTARSIA_INK_X=0";; route) echo "INTARSIA_INK_ROUTE_FUSED=1";; msg) echo "INTARSIA_INK_MSG_PRE=1";; tail) echo "INTARSIA_INK_BLEND_TAIL=1";; all) echo "$ON";; esac; }
for m in off route msg tail all; do
  env $(sw $m) /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 --profile-every 0 > $L/r1_$m.out 2> $L/r1_$m.err
  env $(sw $m) /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/tmo.sh 900 $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --prefill-chunk 16 --golden $G2S --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden $L/r16_$m.safetensors > $L/r16_$m.log 2>&1
done
# the fused kernels ran: kernel-mode traces of the all-on and all-off runs (1 row: 9 tokens; 16 rows: the G2-short prefill)
for m in off all; do
  env $(sw $m) INTARSIA_METAL_PROFILE=kernel:$L/k1_$m.jsonl /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 9 --temperature 0 --profile-every 0 > $L/k1_$m.out 2> $L/k1_$m.err
  python3 -I /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/cnt.py $L/k1_$m.jsonl "1-row decode, $m" > $L/count.txt.$m 2>&1
  env $(sw $m) INTARSIA_METAL_PROFILE=kernel:$L/k16_$m.jsonl /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/tmo.sh 900 $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --prefill-chunk 16 --golden $G2S --max-mean-kl 1 --no-ref-noise --allow-dirty > $L/k16_$m.log 2>&1
  python3 -I /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/cnt.py $L/k16_$m.jsonl "16-row passes, $m" >> $L/count.txt.$m 2>&1
done
{
  r1off=$(md5 -q $L/r1_off.out); r16off=$(md5 -q $L/r16_off.safetensors)
  for m in route msg tail all; do
    a=$(md5 -q $L/r1_$m.out); b=$(md5 -q $L/r16_$m.safetensors)
    echo "BITWISE 1-row decode (64 greedy tokens, text md5): $m on vs all off: $( [ $a = $r1off ] && echo EQUAL || echo DIFFER ) ($a vs $r1off)"
    echo "BITWISE 16-row passes (G2 short, prefill chunk 16, self-golden md5): $m on vs all off: $( [ $b = $r16off ] && echo EQUAL || echo DIFFER ) ($b vs $r16off)"
  done
  grep -h "KL21\|mean KL\|PASS\|FAIL" $L/r16_off.log | tail -3
  cat $L/count.txt.off $L/count.txt.all
} > $L/bitwise.txt
# stress under real GPU load (coordinator condition 3): the 12k-call watchdog stress test in a loop for as long as the
# G2 / G3 model runs below keep the GPU busy; every round's result and slowest batch are logged
cargo test --release -p intarsia-backend-metal --test inkctrl --no-run > /dev/null 2>&1
touch $L/gates.running
( r=0; while [ -e $L/gates.running ]; do r=$((r+1)); INK_CTRL_STRESS=12000 /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/tmo.sh 900 cargo test --release -p intarsia-backend-metal --test inkctrl stress -- --nocapture --test-threads 1 > $L/stress_$r.log 2>&1; echo "STRESSLOAD round $r exit $? $(grep -h STRESS $L/stress_$r.log | sed 's/.*STRESS //' | tr '\n' ' ')" >> $L/stress_load.txt; done ) &
SL=$!
for g in short long; do
  kl=$( [ $g = short ] && echo 4e-3 || echo 6e-3 )
  env $ON $B accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g2-$g-self.safetensors > $L/g2-$g.log 2>&1; echo EXIT $? >> $L/g2-$g.log
  kl=$( [ $g = short ] && echo 1.05e-2 || echo 1.65e-2 )
  env $ON $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --golden /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g3-$g-self.safetensors > $L/g3-$g.log 2>&1; echo EXIT $? >> $L/g3-$g.log
  uv run --project /opt/code/intarsiaml/.worktrees/ink-dfuse/../ink-metal/tools python /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/confident.py $L/g2-$g-self.safetensors golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors > $L/conf-g2-$g.txt 2>&1
  uv run --project /opt/code/intarsiaml/.worktrees/ink-dfuse/../ink-metal/tools python /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/confident.py $L/g3-$g-self.safetensors /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors > $L/conf-g3-$g.txt 2>&1
done
rm -f $L/gates.running; wait $SL
cat $L/bitwise.txt; grep -h "EXIT" $L/g*.log; tail -n 2 $L/conf-*.txt; cat $L/stress_load.txt
