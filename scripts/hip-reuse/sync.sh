#!/usr/bin/env bash
# Lane hip-reuse (ON THE MAC): stamp this worktree (scripts/tree-stamp.sh) and copy it to tatooine:~/intarsiaml-hip-reuse/.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .lane --exclude .hip-reuse-logs ./ tatooine:intarsiaml-hip-reuse/
echo "synced $(cat .stamp)"
