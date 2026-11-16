#!/usr/bin/env bash
# Lane q27-plan (ON THE MAC): stamp this worktree (scripts/tree-stamp.sh) and copy it to tatooine:~/intarsiaml-q27-plan/.
# The 27B golden (untracked on tatooine, copied into golden/ there by hand) and the logs survive the --delete.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .lane --exclude .q27-logs \
  --exclude 'golden/qwen3.8-27b-*.safetensors' ./ tatooine:intarsiaml-q27-plan/
echo "synced $(cat .stamp)"
