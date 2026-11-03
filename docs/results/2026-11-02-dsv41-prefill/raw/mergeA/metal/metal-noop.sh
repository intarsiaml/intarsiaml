#!/bin/bash
# Metal no-op of dsv41/prefill (merged head) against origin/main 4d9a5190e on endor. build = --load cpu, run = --quiet gpu,cpu.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/dsv41-prefill; H=$W/.scratch/head-clone; B=$W/.scratch/noop-base; O=$W/.scratch/metal; mkdir -p $O
TB=$W/.scratch/target-metal-base; TH=$W/.scratch/target-metal-head
P=/opt/code/deepeval/tests/data/prompts/natural-v1
case $1 in
build)
  [ -x $TB/release/intarsia ] || (cd $B && CARGO_TARGET_DIR=$TB cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  (cd $H && CARGO_TARGET_DIR=$TH cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  ls -la $TB/release/intarsia $TH/release/intarsia ;;
run)
  rc=0
  for b in $TB $TH; do [ -x $b/release/intarsia ] || { echo "missing binary $b"; exit 2; }; done
  for m in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B; do
    d=$HOME/models/$m
    $TB/release/intarsia plan show -m $d --backend metal > $O/$m-plan-base.txt 2>&1 < /dev/null
    $TH/release/intarsia plan show -m $d --backend metal > $O/$m-plan-head.txt 2>&1 < /dev/null
    cmp -s $O/$m-plan-base.txt $O/$m-plan-head.txt && echo "$m plan show unchanged ($(wc -l < $O/$m-plan-head.txt) lines)" || { echo "$m plan show DIFFERS"; rc=1; }
    for n in 128 4096; do
      for t in base head; do
        [ $t = base ] && bin=$TB || bin=$TH
        $bin/release/intarsia generate -m $d --backend metal --max-seq 8192 --profile-every 0 --ignore-eos --show-ids --max-tokens 256 --spec off -p "$(cat $P/n$n.txt)" > $O/$m-$t-n$n.out 2> $O/$m-$t-n$n.err < /dev/null
        grep -a '^ids:' $O/$m-$t-n$n.err > $O/$m-$t-n$n.ids
      done
      if [ -s $O/$m-head-n$n.ids ] && cmp -s $O/$m-base-n$n.ids $O/$m-head-n$n.ids; then echo "$m n$n: 256 ids identical"; else echo "$m n$n: ids DIFFER or empty"; rc=1; fi
    done
  done
  $TH/release/intarsia kernels test --backend metal > $O/kernels-full.txt 2>&1 < /dev/null
  echo "kernels test: $(grep -ac 'FAIL' $O/kernels-full.txt) FAIL lines, $(grep -aic 'pass\|ok' $O/kernels-full.txt) ok/pass lines; tail:"; tail -3 $O/kernels-full.txt
  exit $rc ;;
esac
