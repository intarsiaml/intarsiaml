#!/usr/bin/env bash
# From the Mac: copy the worktree to tatooine's lane directory (keeps logs/ and .scratch/bin/ there).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-dec3
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch/bin ./ tatooine:intarsiaml-ink-hip-dec3/ < /dev/null
