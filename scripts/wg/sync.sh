#!/usr/bin/env bash
# Lane wg: copy this worktree to tatooine (~/intarsiaml-wg), the dev tree. `sync.sh base` copies origin/main to ~/intarsiaml-wg-base.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
if [[ "${1:-}" == base ]]; then
  git -C /opt/code/intarsiaml fetch -q origin
  tmp=$(mktemp -d); git -C /opt/code/intarsiaml archive origin/main | tar -x -C "$tmp"
  rsync -a --delete --exclude target "$tmp"/ tatooine:intarsiaml-wg-base/; rm -rf "$tmp"
else
  rsync -a --delete --exclude .git --exclude target --exclude 'docs/results' --exclude '.claude' "$here"/ tatooine:${DEST:-intarsiaml-wg}/
fi
echo synced
