#!/bin/bash
# brief 35 job G1 (quiet): tree $1. OFF = defaults, ON = INTARSIA_INK_MOE_T1=1 INTARSIA_INK_MOE_DNU3=1 (one bundle).
# md5 64 / 256 off vs on; inkling2_batch (Metal) with ON; speed: solo short 4 pairs + 5k 3 pairs ABAB (decode tok/s of
# generate, 257 tokens); B 1/8/16 real text batch_step --digest A B B A B A A B.
set -m
trap 'kill 0' TERM
O=/opt/code/intarsiaml/.worktrees/ink-moe-35/.scratch; W=/opt/code/intarsiaml/.worktrees/ink-batch; R=$W/.scratch/base; T=$W/.scratch/base-target
L=$O/g1; mkdir -p $L
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
git -C $R checkout -q --detach "$1" || exit 1
echo "tree $(git -C $R rev-parse --short HEAD); modified files: $(git -C $R status --short --untracked-files=no | wc -l | tr -d ' ')"
cd $R && git ls-files -z crates kernels recipes | xargs -0 touch
export CARGO_TARGET_DIR=$T
cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E '^error|Finished'
cargo build --release -p intarsia-models --example batch_step 2>&1 | grep -E '^error|Finished'
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
export INTARSIA_METAL_TEST_MODEL=$CK
ON="INTARSIA_INK_MOE_T1=1 INTARSIA_INK_MOE_DNU3=1"; OFF="INTARSIA_INK_MOE_T1=0"
echo "== inkling2_batch ON"; env $ON INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2_batch -- --test-threads 1 2>&1 | grep -E 'test result|panicked|FAILED'
B=$T/release/intarsia; BS=$T/release/examples/batch_step
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
P5="$(cat $R/scripts/hip-decode/prompt4k.txt)"
g() { local e="$1"; shift; env $e $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$1" --max-tokens $2; }
for n in 64 256; do for m in OFF ON; do eval "E=\$$m"; g "$E" "$P" $n > $L/g${n}_$m.out 2> $L/g${n}_$m.err
  echo "md5 $n $m $(md5 -q $L/g${n}_$m.out) $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/g${n}_$m.err)"; done; done
echo "== warm-up"; g "$OFF" "$P" 257 > /dev/null 2>&1
run() { eval "E=\$$1"; local pr="$P"; [ $2 = 5k ] && pr="$P5"
  g "$E" "$pr" 257 > $L/s_$3.out 2> $L/s_$3.err
  echo "$2 $1 $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/s_$3.err) md5 $(md5 -q $L/s_$3.out | cut -c1-8)"; }
i=0; for pr in short short short short 5k 5k 5k; do i=$((i+1)); if [ $((i % 2)) = 1 ]; then run OFF $pr ${i}a; run ON $pr ${i}b; else run ON $pr ${i}a; run OFF $pr ${i}b; fi; done
echo "== B16 warm-up"; $BS --ctx 1024 --rounds 3 --steps 32 --batches 16 --prompts random > /dev/null 2>&1
n=0; for m in OFF ON ON OFF ON OFF OFF ON; do n=$((n+1)); eval "E=\$$m"
  echo "== run $n $m $(date +%T)"; env $E $BS --ctx 1024 --rounds 2 --steps 16 --batches 1,8,16 --prompts text --digest > $L/b$n.txt 2> $L/be$n.txt
  grep -E '^\| (1|8|16) |panicked' $L/b$n.txt; grep '^# B' $L/b$n.txt | awk '{print "  digest B"$3, $6}'; done
echo "g1 done"
