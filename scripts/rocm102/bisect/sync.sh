#!/usr/bin/env bash
# Lane rocm102-bisect (ON THE MAC): stamp this worktree and copy it to tatooine:~/intarsiaml-rocm102-bisect/ (keeps .bisect).
set -euo pipefail
cd "$(dirname "$0")/../../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .bisect --exclude .rocm102-logs ./ tatooine:intarsiaml-rocm102-bisect/
echo "synced $(cat .stamp)"
