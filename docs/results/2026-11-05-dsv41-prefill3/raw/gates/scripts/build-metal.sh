#!/bin/zsh
# usage (inside a fleetq --load cpu slot): build-metal.sh  -> .scratch/gates/intarsia.new (this branch HEAD) and intarsia.base (9783878af, git archive), same flags, own target dirs
set -u
N=/opt/code/intarsiaml/.worktrees/dsv41-prefill3; G=$N/.scratch/gates; BASE=9783878af
mkdir -p $G/base && git -C $N archive $BASE | tar -x -C $G/base
for t in new base; do
  case $t in new) D=$N;; base) D=$G/base;; esac
  ( cd $D && CARGO_TARGET_DIR=$G/target-$t cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -5; cp $G/target-$t/release/intarsia $G/intarsia.$t ) &
done
wait
git -C $N rev-parse HEAD; git -C $N rev-parse $BASE
shasum -a 256 $G/intarsia.new $G/intarsia.base
