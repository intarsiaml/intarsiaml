#!/bin/zsh
# usage (inside a fleetq --load cpu slot): build.sh  -> .scratch/intarsia.new (this branch) and .scratch/intarsia.base (origin/main), same flags, own target dirs
set -u
W=/opt/code/intarsiaml/.worktrees/failpaths-gates; N=/opt/code/intarsiaml/.worktrees/dsv41-failpaths
for t in new base; do
  case $t in new) D=$N;; base) D=$W/base;; esac
  ( cd $D && CARGO_TARGET_DIR=$W/target-$t cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -5; cp $W/target-$t/release/intarsia $W/intarsia.$t ) &
done
wait
git -C $N rev-parse HEAD; git -C $W/base rev-parse HEAD
shasum -a 256 $W/intarsia.new $W/intarsia.base
