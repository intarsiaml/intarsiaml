#!/usr/bin/env bash
# From the Mac: copy the worktree to tatooine's lane directory (keeps logs/ and .scratch/bin/ there; docs/results not copied).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-verify
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch/bin --exclude /docs/results ./ tatooine:intarsiaml-ink-hip-verify/ < /dev/null
