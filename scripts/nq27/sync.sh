#!/usr/bin/env bash
# Lane NQ27 (from ARX), on the Mac from the worktree root: copy the branch tree to tatooine (file copy, no slot). BASE=1 BASEREV=<rev>: main's tree
# (git archive of that revision) into ~/intarsiaml-nq27/nq27-base-tree.
set -euo pipefail
if [[ "${BASE:-0}" == 1 ]]; then
  rev=${BASEREV:?BASEREV}
  ssh -o BatchMode=yes tatooine "rm -rf ~/intarsiaml-nq27/nq27-base-tree && mkdir -p ~/intarsiaml-nq27/nq27-base-tree"
  git archive "$rev" | ssh -o BatchMode=yes tatooine "tar -x -C ~/intarsiaml-nq27/nq27-base-tree"
  echo "base tree = $rev"
  exit 0
fi
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .lane \
  --exclude '/nq27-*' ./ tatooine:~/intarsiaml-nq27/
