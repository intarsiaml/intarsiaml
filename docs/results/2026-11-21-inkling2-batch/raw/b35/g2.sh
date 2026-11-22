#!/bin/bash
# brief 35 job G2 (quiet): T1 default-on at tree $1 vs main $2. Base binary build; tip build; tests (inkmoe, rows_exact,
# inkling2, inkling2_batch, docs_env_vars); admission moe_experts tip default vs T1=0; md5 64/256 tip default vs T1=0;
# G2 (decode) / G3 (prefill) short + long, tip default vs T1=0, self-goldens byte-compared; plan show Qwen3-8B base vs tip.
set -m
trap 'kill 0' TERM
O=/opt/code/intarsiaml/.worktrees/ink-moe-35/.scratch; W=/opt/code/intarsiaml/.worktrees/ink-batch; R=$W/.scratch/base; T=$W/.scratch/base-target
L=$O/g2; mkdir -p $L $O/bin
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
export CARGO_TARGET_DIR=$T
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
export INTARSIA_METAL_TEST_MODEL=$CK
git -C $R checkout -q --detach "$2" || exit 1; (cd $R && git ls-files -z crates kernels recipes | xargs -0 touch && cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E '^error|Finished'); cp $T/release/intarsia $O/bin/intarsia-base
git -C $R checkout -q --detach "$1" || exit 1
echo "tree $(git -C $R rev-parse --short HEAD); modified files: $(git -C $R status --short --untracked-files=no | wc -l | tr -d ' ')"
cd $R && git ls-files -z crates kernels recipes | xargs -0 touch
cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E '^error|Finished'
echo "rustfmt: $(rustfmt --edition 2021 --check crates/intarsia-backend-metal/src/inkmoe.rs crates/intarsia-backend-metal/tests/inkmoe.rs >/dev/null 2>&1 && echo clean || echo DIFF)"
B=$T/release/intarsia; OFF="INTARSIA_INK_MOE_T1=0"; ON="INTARSIA_X=0"
f='test result|panicked|FAILED'
for t in inkmoe rows_exact; do echo "-- metal $t"; cargo test --release -p intarsia-backend-metal --test $t 2>&1 | grep -E "$f"; done
for t in inkling2 inkling2_batch; do echo "-- models $t"; INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test $t -- --test-threads 1 2>&1 | grep -E "$f"; done
echo "-- docs_env_vars"; cargo test --release -p intarsia-cli --features metal --test docs_env_vars 2>&1 | grep -E "$f"
for m in OFF ON; do eval "E=\$$m"; env $E $B kernels test --backend metal --op moe_experts > $L/kt_$m.log 2>&1
  echo "admission $m exit $? | ADMITTED $(grep -c ADMITTED $L/kt_$m.log) | $(grep -cE ' FAIL|REJECT|NO CASES' $L/kt_$m.log) fail/reject"; done
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
for n in 64 256; do for m in OFF ON; do eval "E=\$$m"; env $E $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens $n > $L/g${n}_$m.out 2> $L/g${n}_$m.err
  echo "md5 $n $m $(md5 -q $L/g${n}_$m.out)"; done; done
for g in short long; do
  for m in OFF ON; do eval "E=\$$m"
    kl=$( [ $g = short ] && echo 4e-3 || echo 6e-3 )
    env $E $B accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g2-$g-$m-self.safetensors > $L/g2-$g-$m.log 2>&1; echo EXIT $? >> $L/g2-$g-$m.log
    kl=$( [ $g = short ] && echo 1.05e-2 || echo 1.65e-2 )
    env $E $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --golden /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g3-$g-$m-self.safetensors > $L/g3-$g-$m.log 2>&1; echo EXIT $? >> $L/g3-$g-$m.log
  done
  for k in g2 g3; do
    echo "== $k-$g ON: $(grep -h '^OVERALL' $L/$k-$g-ON.log | tr -s ' ' | cut -c1-90 | tr '\n' '|') $(tail -1 $L/$k-$g-ON.log)"
    echo "   self-golden OFF vs ON: $(python3 -I $O/stcmp.py $L/$k-$g-OFF-self.safetensors $L/$k-$g-ON-self.safetensors 2>&1 | tail -2)"
  done
done
for m in Qwen3-8B; do
  $O/bin/intarsia-base plan show --model ~/models/$m --backend metal > $L/plan_base.out 2> $L/plan_base.err
  $B plan show --model ~/models/$m --backend metal > $L/plan_tip.out 2> $L/plan_tip.err
  if cmp -s $L/plan_base.out $L/plan_tip.out && cmp -s $L/plan_base.err $L/plan_tip.err; then v=byte-identical; else v=DIFFERS; fi
  echo "PLAN $m base vs tip: $v"
done
echo "g2 done"
