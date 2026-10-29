#!/usr/bin/env bash
# Lane PF27b, on the Mac from the worktree root: copy the branch tree to tatooine (file copy, no slot). BASE=1 BASEREV=<rev>: main's tree
# (git archive of that revision) into ~/intarsiaml-pf27b/pf27b-base-tree.
set -euo pipefail
if [[ "${BASE:-0}" == 1 ]]; then
  rev=${BASEREV:?BASEREV}
  ssh -o BatchMode=yes tatooine "rm -rf ~/intarsiaml-pf27b/pf27b-base-tree && mkdir -p ~/intarsiaml-pf27b/pf27b-base-tree"
  git archive "$rev" | ssh -o BatchMode=yes tatooine "tar -x -C ~/intarsiaml-pf27b/pf27b-base-tree"
  echo "base tree = $rev"
  exit 0
fi
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .lane \
  --exclude '/pf27b-*' ./ tatooine:~/intarsiaml-pf27b/
