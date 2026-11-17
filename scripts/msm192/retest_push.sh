#!/bin/zsh
# Lane msm192: the routing test file in release and debug (after the last test edit), then the push (its pre-push hook runs the docs tests).
# Under fleetq --load cpu.
set -u
W=/opt/code/intarsiaml/.worktrees/msm192; cd $W || exit 1
for prof in "--release" ""; do
  out=$(cargo test $=prof -p intarsia-backend-metal --test mxfp4_nax 2>&1); rc=$?
  echo "$out" | grep -E "^test |^test result|^error|FAILED"; echo "mxfp4_nax ${prof:-debug} exit $rc"
  [ $rc = 0 ] || exit 1
done
git push -u origin msm192
