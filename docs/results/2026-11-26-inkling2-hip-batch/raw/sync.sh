#!/usr/bin/env bash
# From the Mac: copy the worktree to tatooine's lane directory (keeps logs/ and .scratch/bin/ there).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-batch
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch/bin --exclude /docs/results --exclude /.scratch/main-tree/docs/results --exclude /.scratch/main-tree/.git ./ tatooine:intarsiaml-ink-hip-batch/ < /dev/null
