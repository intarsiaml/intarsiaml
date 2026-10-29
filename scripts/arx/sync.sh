#!/usr/bin/env bash
# Lane ARX, on the Mac from the worktree root: copy the branch tree to tatooine (file copy, no slot). BASE=1 BASEREV=<rev>: main's tree
# (git archive of that revision) into ~/intarsiaml-arx/arx-base-tree.
set -euo pipefail
if [[ "${BASE:-0}" == 1 ]]; then
  rev=${BASEREV:?BASEREV}
  ssh -o BatchMode=yes tatooine "rm -rf ~/intarsiaml-arx/arx-base-tree && mkdir -p ~/intarsiaml-arx/arx-base-tree"
  git archive "$rev" | ssh -o BatchMode=yes tatooine "tar -x -C ~/intarsiaml-arx/arx-base-tree"
  echo "base tree = $rev"
  exit 0
fi
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .worktrees --exclude .lane \
  --exclude '/arx-*' ./ tatooine:~/intarsiaml-arx/
