#!/usr/bin/env bash
# Lane gdn-default (ON THE MAC): stamp this worktree (scripts/tree-stamp.sh) and copy it to tatooine:~/intarsiaml-gdn-default/.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .gdn-default-logs ./ tatooine:intarsiaml-gdn-default/
echo "synced $(cat .stamp)"
