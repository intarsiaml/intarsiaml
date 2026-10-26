#!/usr/bin/env bash
# Lane lb, LOCAL: copy the dev tree (the working tree) to tatooine:~/intarsiaml-lb/dev and the base (origin/main) to ~/intarsiaml-lb/base.
# No GPU or build involved. usage: scripts/lb/sync.sh [base|dev|both]
# rsync with --no-times --checksum so cargo sees changed files as new (no stale build), unchanged ones are not sent.
set -euo pipefail
what=${1:-both}
cd "$(git rev-parse --show-toplevel)"
ssh tatooine 'mkdir -p ~/intarsiaml-lb/dev ~/intarsiaml-lb/logs'
if [ "$what" != base ]; then
  rsync -r --no-times --checksum --delete --exclude .git --exclude .worktrees --exclude .lbscratch --exclude target --exclude 'target-*' --exclude .claude ./ tatooine:intarsiaml-lb/dev/
fi
if [ "$what" != dev ]; then
  git fetch -q origin main
  ssh tatooine 'rm -rf ~/intarsiaml-lb/base && mkdir -p ~/intarsiaml-lb/base'
  git archive origin/main | ssh tatooine 'tar -x -C ~/intarsiaml-lb/base'
fi
echo "synced ($what); base = $(git rev-parse --short origin/main), dev = $(git rev-parse --short HEAD) + working tree"
