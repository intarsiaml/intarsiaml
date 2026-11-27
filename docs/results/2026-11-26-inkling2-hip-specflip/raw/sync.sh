#!/usr/bin/env bash
# From the Mac: copy the worktree to tatooine's lane directory (keeps logs/ and .scratch/bin/ there).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-specflip
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch ./ tatooine:intarsiaml-ink-hip-specflip/ < /dev/null
