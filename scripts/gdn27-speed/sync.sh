#!/usr/bin/env bash
# Lane gdn27-speed (ON THE MAC): stamp this worktree and copy it to tatooine:~/intarsiaml-gdn27-speed/.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .lane --exclude docs/results ./ tatooine:intarsiaml-gdn27-speed/
echo "synced $(cat .stamp)"
