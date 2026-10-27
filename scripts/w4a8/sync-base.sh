#!/usr/bin/env bash
# Lane W4A8, on the Mac, from the worktree root: copy a checkout of origin/main (BASE_DIR, a git worktree) to tatooine:~/intarsiaml-w4a8-base.
set -euo pipefail
: "${BASE_DIR:?BASE_DIR=<git worktree of origin/main>}"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees "$BASE_DIR"/ tatooine:~/intarsiaml-w4a8-base/
