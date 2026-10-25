#!/usr/bin/env bash
# Lane l8, LOCAL: copy the kernels + scripts of this worktree to tatooine:~/intarsiaml-l8-src. No GPU, no build.
# usage: scripts/l8/sync.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ssh tatooine 'mkdir -p ~/intarsiaml-l8-src'
rsync -a --delete --include='/kernels/***' --include='/scripts/' --include='/scripts/l8/***' --exclude='*' ./ tatooine:intarsiaml-l8-src/
echo "synced $(git rev-parse --short HEAD)"
