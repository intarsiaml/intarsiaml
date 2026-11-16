#!/usr/bin/env bash
# Lane a4k-restore (ON THE MAC): stamp this worktree (scripts/tree-stamp.sh) and copy it to tatooine:~/intarsiaml-a4k-restore/.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .lane --exclude .a4k-restore-logs ./ tatooine:intarsiaml-a4k-restore/
echo "synced $(cat .stamp)"
