#!/bin/sh
# lane ink-d1row job L (gpu + cpu load, shares the box): build, bitwise tests, admission, G1, greedy-text identity
# (pipelined vs plain), G2 / G3. Writes .scratch/L/*.log; prints one line per step.
cd /opt/code/intarsiaml/.worktrees/ink-d1row || exit 1
S=.scratch/L
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
rm -f $S/build.ok
git rev-parse --short HEAD > $S/head.txt
cargo build --release --bin intarsia --features metal > $S/build.log 2>&1 || { echo "build FAILED"; tail -40 $S/build.log; exit 1; }
cargo build --release -p intarsia-backend-metal --examples >> $S/build.log 2>&1 || { echo "examples build FAILED"; tail -40 $S/build.log; exit 1; }
cargo test --release -p intarsia-backend-metal --test inkmoe --test inkling2_attn --test rows_exact --no-run >> $S/build.log 2>&1 || { echo "test build FAILED"; tail -40 $S/build.log; exit 1; }
cargo test --release -p intarsia-models --test inkling2 --no-run >> $S/build.log 2>&1 || { echo "G1 test build FAILED"; tail -40 $S/build.log; exit 1; }
touch $S/build.ok; echo "build ok $(cat $S/head.txt)"
cargo test --release -p intarsia-backend-metal --test inkmoe --test inkling2_attn --test rows_exact -- --nocapture > $S/t_bitwise.log 2>&1; echo "bitwise tests exit $?"
grep -E "bitwise|test result|panicked|FAILED" $S/t_bitwise.log | head -20
cargo test --release -p intarsia-backend-metal --lib > $S/t_lib.log 2>&1; echo "metal lib tests exit $?"; grep "test result" $S/t_lib.log
for op in rel_sink_attn ink_attn_decode moe_experts moe_experts_mxfp8; do echo "=== $op"; $B kernels test --backend metal --op $op 2>&1 | tail -8; done > $S/admission.log 2>&1
grep -E "ADMITTED|FAIL|REJECT" $S/admission.log
INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 -- --test-threads 1 > $S/g1_metal.log 2>&1; echo "G1 metal exit $?"; grep "test result" $S/g1_metal.log
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
G="$B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0"
for v in 0 1; do
  INTARSIA_INK_ATTN_PIPE=$v INTARSIA_INK_MOE_PIPE=$v $G --prompt "$P" --max-tokens 64 > $S/bw_short_$v.out 2> $S/bw_short_$v.err
  INTARSIA_INK_ATTN_PIPE=$v INTARSIA_INK_MOE_PIPE=$v $G --prompt "$(cat .scratch/prompt5k.txt)" --max-tokens 32 > $S/bw_5k_$v.out 2> $S/bw_5k_$v.err
done
md5 -q $S/bw_short_0.out $S/bw_short_1.out $S/bw_5k_0.out $S/bw_5k_1.out | tr '\n' ' '; echo "(short off, short on, 5k off, 5k on)"
for g in short long; do
  kl=$( [ $g = short ] && echo 4e-3 || echo 6e-3 )
  $B accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $S/g2-$g-self.safetensors > $S/g2-$g.log 2>&1; echo "G2 $g exit $?"
  kl=$( [ $g = short ] && echo 1.05e-2 || echo 1.65e-2 )
  $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --golden /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $S/g3-$g-self.safetensors > $S/g3-$g.log 2>&1; echo "G3 $g exit $?"
done
grep -h -E "KL21|golden gate" $S/g2-*.log $S/g3-*.log | head -20
