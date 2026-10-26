#!/usr/bin/env bash
# Lane l7m, LOCAL: copy the dev tree to tatooine:~/intarsiaml-l7m-hip and the base (origin/main, the PR base) to ~/intarsiaml-l7m-base.
# No GPU or build involved. usage: scripts/l7m/sync.sh [base|dev|both]
# rsync keeps edit times by default and cargo skips a rebuild when the build is newer than the source (hr3 served a stale binary twice
# that way), so --no-times --checksum: changed files get a new mtime, unchanged ones are not sent.
set -euo pipefail
what=${1:-both}
cd "$(git rev-parse --show-toplevel)"
if [ "$what" != base ]; then
  rsync -r --no-times --checksum --delete --exclude .git --exclude .worktrees --exclude target --exclude 'target-*' --exclude .claude ./ tatooine:intarsiaml-l7m-hip/
fi
if [ "$what" != dev ]; then
  git fetch -q origin main
  ssh tatooine 'rm -rf ~/intarsiaml-l7m-base && mkdir -p ~/intarsiaml-l7m-base'
  git archive origin/main | ssh tatooine 'tar -x -C ~/intarsiaml-l7m-base'
fi
echo "synced ($what); base = $(git rev-parse --short origin/main), dev = $(git rev-parse --short HEAD)"
