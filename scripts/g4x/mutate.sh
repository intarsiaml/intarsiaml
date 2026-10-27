#!/usr/bin/env bash
# G4X mutation checks (run on spark2 INSIDE a fleetq slot, --mem 8; the build loads the CPU, the sweep the GPU a little).
#   mutate.sh [mutation ...]      default: all of `mutate.py --list`
# Per mutation: a copy of the deployed tree (~/intarsiaml-g4x, sources only; its own copy of target/ in replay-out/mut) gets one planted
# bug (mutate.py), is rebuilt, and `intarsia kernels test --backend cuda --op <op> --kernel <kernel>` must FAIL
# (nonzero exit: the kernel is REJECTED). The unmutated tree is checked first and must pass. A build error is reported as a harness problem.
set -uo pipefail
R=$HOME/intarsiaml-g4x; MUT=$R/replay-out/mut;
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p $MUT
[ -d $MUT/target ] || cp -a $R/target $MUT/target
rsync -a --delete --exclude target --exclude replay-out --exclude DEPLOY_STAMP.json $R/ $MUT/
ALL=$(python3 -I $MUT/scripts/g4x/mutate.py --list | cut -d' ' -f1 | tr '\n' ' ')
MUTS=${*:-$ALL}
build() { (cd $MUT && CARGO_BUILD_JOBS=6 cargo build --release --locked -p intarsia-cli --features cuda 2>&1 | grep -E "^error|could not compile|nvcc" | head -5); }
run() { # run <kernel> <op>: output in $MUT/out.txt, returns the exit code of `kernels test` (nonzero: a kernel was REJECTED)
  $MUT/target/release/intarsia kernels test --backend cuda --op $2 --kernel $1 > $MUT/out.txt 2>&1
}
echo "== unmutated"; build
run attn_decode_bf16_splitkv_ext attn_decode; rc=$?; tail -n 4 $MUT/out.txt | cut -c1-200; echo "unmutated exit $rc"; [ $rc = 0 ] || exit 1
bad=0
for m in $MUTS; do
  read -r name kernel op < <(python3 -I $MUT/scripts/g4x/mutate.py --list | awk -v m=$m '$1==m')
  rsync -a $R/kernels/ $MUT/kernels/
  python3 -I $MUT/scripts/g4x/mutate.py $MUT $m || { echo "HARNESS: $m did not apply"; bad=1; continue; }
  b=$(build); if [ -n "$b" ]; then echo "HARNESS: $m does not build: $b"; bad=1; continue; fi
  run $kernel $op; rc=$?
  if [ $rc != 0 ]; then echo "KILLED   $m ($kernel): $(grep -E '^  FAIL|REJECTED' $MUT/out.txt | head -1 | cut -c1-170)"; else echo "SURVIVED $m ($kernel): exit 0, $(tail -n 2 $MUT/out.txt | tr '\n' ' ' | cut -c1-120)"; bad=1; fi
done
rsync -a $R/kernels/ $MUT/kernels/
exit $bad
