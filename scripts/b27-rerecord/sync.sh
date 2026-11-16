#!/usr/bin/env bash
# Lane b27-rerecord (ON THE MAC): stamp this worktree (scripts/tree-stamp.sh) and copy it to tatooine:~/intarsiaml-b27-rerecord/.
# The goldens (untracked on tatooine, copied into golden/ there by hand) and the logs survive the --delete.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/tree-stamp.sh > .stamp
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .lane --exclude .b27-logs \
  --exclude 'golden/qwen3.8-27b-*.safetensors' ./ tatooine:intarsiaml-b27-rerecord/
echo "synced $(cat .stamp)"
