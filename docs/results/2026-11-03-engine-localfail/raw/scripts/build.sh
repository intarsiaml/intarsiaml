#!/bin/zsh
# usage (inside a fleetq --load cpu slot): build.sh  -> .scratch/intarsia.new (this branch) and .scratch/intarsia.base (origin/main), same flags, own target dirs
set -u
W=/opt/code/intarsiaml/.worktrees/engine-localfail
for t in new base; do
  case $t in new) D=$W;; base) D=$W/.scratch/base;; esac
  ( cd $D && CARGO_TARGET_DIR=$W/.scratch/target-$t cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -5; cp $W/.scratch/target-$t/release/intarsia $W/.scratch/intarsia.$t ) &
done
wait
git -C $W rev-parse HEAD; git -C $W/.scratch/base rev-parse HEAD
shasum -a 256 $W/.scratch/intarsia.new $W/.scratch/intarsia.base
