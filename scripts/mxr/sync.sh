#!/usr/bin/env bash
# Lane mxr, LOCAL: copy the dev tree (this worktree) to tatooine:~/intarsiaml-mxr-dev. No GPU or build involved.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
rsync -a --delete --exclude .git --exclude .worktrees --exclude target --exclude 'target-*' --exclude .mxrbin ./ tatooine:intarsiaml-mxr-dev/
echo "synced dev $(git rev-parse --short HEAD) (+ working tree) to tatooine:~/intarsiaml-mxr-dev"
