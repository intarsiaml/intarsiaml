#!/usr/bin/env bash
# Lane rocm102 (ON THE MAC): stamp this worktree (scripts/tree-stamp.sh) and copy it to tatooine:~/intarsiaml-rocm102/.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .rocm102-logs ./ tatooine:intarsiaml-rocm102/
echo "synced $(cat .stamp)"
