#!/usr/bin/env bash
# Lane wire-probe (ON THE MAC, inside a fleetq job or lease): stamp this worktree (scripts/tree-stamp.sh) and copy it to tatooine:~/intarsiaml-wire-probe/.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .lane --exclude .wire-probe-logs --exclude 'docs/results/*/raw' ./ tatooine:intarsiaml-wire-probe/
echo "synced $(cat .stamp)"
