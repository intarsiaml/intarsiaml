#!/bin/bash
# ink-busy2048 gate A (load cpu): build main 0b8fab600 (base tree) and ink/busy2048 (own target, cloned from base-target), freeze both
# binaries, run the touched crates' tests (models, cli with docs_env_vars, core batch), plan show main vs new (Qwen3-8B, Llama-3.1-8B, Inkling; json + txt).
. /opt/code/intarsiaml/.worktrees/ink-batch/.scratch/b2048/common.sh
mkdir -p $O/bin/main $O/bin/new $O/plan
echo "new tree $(git -C $N rev-parse --short HEAD); modified: $(git -C $N status --short --untracked-files=no | wc -l | tr -d ' '); untracked: $(git -C $N status --short | grep -c '^??')"
run git -C $R checkout -q --detach $MAIN || exit 1
echo "main tree $(git -C $R rev-parse --short HEAD)"
(cd $R && git ls-files -z crates kernels recipes | xargs -0 touch)
[ -d $TN ] || { echo "cloning base-target"; run cp -cR $TB $TN; }
(cd $N && git ls-files -z crates kernels recipes | xargs -0 touch)
echo "== build main $(date +%T)"
cd $R; runlog $O/build-main.log env CARGO_TARGET_DIR=$TB cargo build --release --bin intarsia --features metal; grep -E '^error|Finished' $O/build-main.log
runlog $O/build-main2.log env CARGO_TARGET_DIR=$TB cargo build --release -p intarsia-models --example batch_step; grep -E '^error|Finished' $O/build-main2.log
cp $TB/release/intarsia $O/bin/main/intarsia; cp $TB/release/examples/batch_step $O/bin/main/batch_step
echo "== build new $(date +%T)"
cd $N; runlog $O/build-new.log env CARGO_TARGET_DIR=$TN cargo build --release --bin intarsia --features metal; grep -E '^error|Finished' $O/build-new.log
cp $TN/release/intarsia $O/bin/new/intarsia
echo "md5 main $(md5 -q $O/bin/main/intarsia) new $(md5 -q $O/bin/new/intarsia)"
echo "== tests $(date +%T)"
cd $N
echo "-- intarsia-models (all tests, incl. recipe_ink_busy_chunk, recipes_doc, recipe_gumbel, recipe_27b_chunk)"
runlog $O/test-models.log env CARGO_TARGET_DIR=$TN cargo test --release -p intarsia-models; grep -E 'FAILED|panicked|^error|test result|Running.*recipe' $O/test-models.log | head -80
echo "-- intarsia-cli --features metal (docs_env_vars, common::tests busy_chunk)"
runlog $O/test-cli.log env CARGO_TARGET_DIR=$TN cargo test --release -p intarsia-cli --features metal; grep -E 'FAILED|panicked|^error|test result|Running.*docs_env' $O/test-cli.log | head -40
echo "-- intarsia-core batch"
runlog $O/test-core.log env CARGO_TARGET_DIR=$TN cargo test --release -p intarsia-core --lib batch; grep -E 'FAILED|panicked|^error|test result' $O/test-core.log
echo "== plan show $(date +%T)"
for m in Qwen3-8B Llama-3.1-8B-Instruct; do
  for v in main new; do
    for fmt in txt json; do
      x=""; [ $fmt = json ] && x="--json"
      $O/bin/$v/intarsia plan show --model ~/models/$m --backend metal $x > $O/plan/$m-$v.$fmt 2> $O/plan/$m-$v.$fmt.err; echo "plan $m $v $fmt exit $?" > /dev/null
    done
  done
  for fmt in txt json; do
    cmp $O/plan/$m-main.$fmt $O/plan/$m-new.$fmt && echo "plan show $m $fmt: BYTE-IDENTICAL main vs new ($(wc -c < $O/plan/$m-new.$fmt) bytes)"
    cmp $O/plan/$m-main.$fmt.err $O/plan/$m-new.$fmt.err && echo "plan show $m $fmt stderr: IDENTICAL"
  done
done
for v in main new; do for fmt in txt json; do x=""; [ $fmt = json ] && x="--json"
  $O/bin/$v/intarsia plan show --model $CK --backend metal --max-seq 8192 $x > $O/plan/ink-$v.$fmt 2> $O/plan/ink-$v.$fmt.err; done; done
for fmt in txt json; do cmp $O/plan/ink-main.$fmt $O/plan/ink-new.$fmt && echo "plan show Inkling-2-Mini $fmt: BYTE-IDENTICAL (the plan does not carry the busy chunk)"; done
echo "gate A done"
