#!/usr/bin/env bash
# Lane wht6-ar (ON THE MAC, inside a fleetq job or lease): copy this worktree to tatooine:~/intarsiaml-wht6-ar/ (the lane's own tree).
cd "$(dirname "$0")/../.." && exec rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .wht6-ar-logs ./ tatooine:intarsiaml-wht6-ar/
