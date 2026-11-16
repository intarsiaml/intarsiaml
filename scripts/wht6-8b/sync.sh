#!/usr/bin/env bash
# Lane wht6-8b (ON THE MAC, inside a fleetq job or lease): stamp this worktree (scripts/tree-stamp.sh) and copy it to tatooine:~/intarsiaml-wht6-8b/.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .lane --exclude .wht6-8b-logs --exclude 'docs/results/*/raw' ./ tatooine:intarsiaml-wht6-8b/
echo "synced $(cat .stamp)"
