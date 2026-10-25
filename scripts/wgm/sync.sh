#!/usr/bin/env bash
# Lane wg merge gates, ON THE MAC: copy the merged head to tatooine ~/intarsiaml-wgm and main (the base worktree) to ~/intarsiaml-wgm-base,
# plus the wgm scripts to ~/wgm-*.sh.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
rsync -a --delete --exclude .git --exclude target --exclude 'docs/results' --exclude '.claude' --exclude '.worktrees' "$here"/ tatooine:intarsiaml-wgm/
rsync -a --delete --exclude .git --exclude target --exclude 'docs/results' --exclude '.claude' --exclude '.worktrees' "$here"/.worktrees/base/ tatooine:intarsiaml-wgm-base/
for f in prep win gates; do scp -q "$here/scripts/wgm/$f.sh" tatooine:wgm-$f.sh; done
echo "synced dev $(git -C "$here" rev-parse --short HEAD) base $(git -C "$here/.worktrees/base" rev-parse --short HEAD)"
