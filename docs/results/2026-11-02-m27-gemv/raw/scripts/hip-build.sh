#!/bin/bash
# on endor, inside a tatooine fleetq slot (radiance stays up): ship head and origin/main, build each into its own target volume
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/m27-gemv
HEAD_SHA=f2690aa76d1aaccf0384d15acbe295e1de1c8bf6
MAIN_SHA=c269ded36
for t in head:$HEAD_SHA main:$MAIN_SHA; do
  n=${t%%:*}; r=${t#*:}
  echo "=== hip build $n $r $(date +%T)"
  ssh -n -o BatchMode=yes tatooine "rm -rf ~/intarsia-m27-gemv/$n && mkdir -p ~/intarsia-m27-gemv/$n"
  git archive --format=tar $r | ssh -o BatchMode=yes tatooine "tar -x -C ~/intarsia-m27-gemv/$n"
  ssh -o BatchMode=yes tatooine "bash -s $n" < .scratch/hip-remote-build.sh || echo "BUILD FAILED $n"
done
echo "=== hip build done $(date +%T)"
