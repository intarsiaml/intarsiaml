#!/usr/bin/env bash
# Lane mxr, LOCAL: copy the dev tree (this worktree) to tatooine:~/intarsiaml-mxr-dev and, with 'base', origin/main (the PR
# base) to ~/intarsiaml-mxr-dev/.base. No GPU or build involved. usage: sync.sh [dev|base|both] (default dev)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
what=${1:-dev}
if [ "$what" != base ]; then
  rsync -a --delete --exclude .git --exclude .worktrees --exclude target --exclude 'target-*' --exclude .mxrbin --exclude .mxrlogs --exclude .base --exclude .base-metal --exclude .targets ./ tatooine:intarsiaml-mxr-dev/
  echo "synced dev $(git rev-parse --short HEAD) (+ working tree) to tatooine:~/intarsiaml-mxr-dev"
fi
if [ "$what" != dev ]; then
  git fetch -q origin main
  ssh tatooine 'rm -rf ~/intarsiaml-mxr-dev/.base && mkdir -p ~/intarsiaml-mxr-dev/.base'
  git archive origin/main | ssh tatooine 'tar -x -C ~/intarsiaml-mxr-dev/.base'
  echo "synced base $(git rev-parse --short origin/main) to tatooine:~/intarsiaml-mxr-dev/.base"
fi
