#!/bin/bash
# build head and origin/main on spark2 (own trees under ~/intarsia-deploy, own targets), no smoke
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/m27-gemv
HEAD_SHA=f2690aa76d1aaccf0384d15acbe295e1de1c8bf6
MAIN_SHA=c269ded36
for t in head:$HEAD_SHA main:$MAIN_SHA; do
  n=${t%%:*}; r=${t#*:}
  echo "=== build $n $r $(date +%T)"
  scripts/deploy.sh spark2 --backend cuda --ref $r --tree m27-gemv-$n --smoke none || echo "BUILD FAILED $n rc=$?"
done
echo "=== build done $(date +%T)"
