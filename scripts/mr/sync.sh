#!/usr/bin/env bash
# Lane mr, LOCAL: copy the dev tree to tatooine:~/mr-hip and the base (origin/main, the PR base) to ~/mr-hip/.base.
# No GPU or build involved. usage: scripts/mr/sync.sh [base|dev|both]
set -euo pipefail
what=${1:-both}
cd "$(git rev-parse --show-toplevel)"
if [ "$what" != base ]; then
  rsync -a --exclude .git --exclude .worktrees --exclude target --exclude 'target-*' --exclude web --exclude .base --exclude .mrbin ./ tatooine:mr-hip/
fi
if [ "$what" != dev ]; then
  git fetch -q origin main
  ssh tatooine 'rm -rf ~/mr-hip/.base && mkdir -p ~/mr-hip/.base'
  git archive origin/main | ssh tatooine 'tar -x -C ~/mr-hip/.base'
fi
echo "synced ($what); base = $(git rev-parse --short origin/main)"
