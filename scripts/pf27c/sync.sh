#!/usr/bin/env bash
# Lane PF27c (from ARX), on the Mac from the worktree root: copy the branch tree to tatooine (file copy, no slot). BASE=1 BASEREV=<rev>: main's tree
# (git archive of that revision) into ~/intarsiaml-pf27c/pf27c-base-tree.
set -euo pipefail
if [[ "${BASE:-0}" == 1 ]]; then
  rev=${BASEREV:?BASEREV}
  ssh -o BatchMode=yes tatooine "rm -rf ~/intarsiaml-pf27c/pf27c-base-tree && mkdir -p ~/intarsiaml-pf27c/pf27c-base-tree"
  git archive "$rev" | ssh -o BatchMode=yes tatooine "tar -x -C ~/intarsiaml-pf27c/pf27c-base-tree"
  echo "base tree = $rev"
  exit 0
fi
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .lane \
  --exclude '/pf27c-*' ./ tatooine:~/intarsiaml-pf27c/
