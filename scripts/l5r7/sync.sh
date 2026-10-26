#!/usr/bin/env bash
# Lane l5, LOCAL: copy this tree to tatooine:~/intarsiaml-l5-<name> (default name: hip). No GPU or build involved.
# usage: scripts/l5r7/sync.sh [name]   (a window runs from its own snapshot: use a new name per window)
# Run it inside a no-load slot: FLEETQ_OWNER=claude-f31ccf91-l5 fleetq run --host tatooine-r9700-2 --load none --expect 3 \
#   --label "l5: sync snapshot (no GPU)" -- scripts/l5r7/sync.sh <name>
set -euo pipefail
name=${1:-hip}
cd "$(git rev-parse --show-toplevel)"
ssh tatooine "mkdir -p ~/intarsiaml-l5-$name"
rsync -a --delete --exclude .git --exclude .worktrees --exclude target --exclude 'target-*' --exclude 'l5-out' --exclude node_modules ./ "tatooine:intarsiaml-l5-$name/"
echo "synced to tatooine:~/intarsiaml-l5-$name ($(git rev-parse --short HEAD) + working tree)"
