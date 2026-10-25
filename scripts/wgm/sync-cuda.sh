#!/usr/bin/env bash
# Lane wg merge gates, ON THE MAC: rsync the merged head and main to spark1 ~/intarsia-decfuse/wg-src-{dev,base}, and the script.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd); P=intarsia-decfuse
for pair in "dev:$here" "base:$here/.worktrees/base"; do
  rsync -a --delete --exclude .git --exclude target --exclude 'docs/results' --exclude '.claude' --exclude '.worktrees' "${pair#*:}"/ spark1:$P/wg-src-${pair%%:*}/
done
scp -q "$here/scripts/wgm/cuda-noop.sh" spark1:wg-cuda-noop.sh
echo "synced dev $(git -C "$here" rev-parse --short HEAD) base $(git -C "$here/.worktrees/base" rev-parse --short HEAD)"
