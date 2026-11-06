#!/bin/zsh
# Lane lmhead-fp8: Metal release builds on endor of this worktree (head) and of the base commit extracted to .scratch/base (main 5390e65b4).
#   fleetq run --host endor --load cpu --mem 24 --expect 40 --label "..." -- zsh scripts/lmhead/mbuild.sh
set -u
R=${0:A:h:h:h}
for t in head base; do
  d=$R; [[ $t == base ]] && d=$R/.scratch/base
  ( cd $d && INTARSIA_BUILD_STAMP="lmhead-$t $(git -C $R rev-parse --short=9 HEAD)" cargo build --release -p intarsia-cli --features metal 2>&1 | tail -4 )
  echo "build $t exit $?"; ls -la $d/target/release/intarsia
done
