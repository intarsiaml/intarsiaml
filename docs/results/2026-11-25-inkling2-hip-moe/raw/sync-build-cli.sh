#!/usr/bin/env bash
# From endor: copy the worktree to tatooine and build the intarsia binary and the test binary there (CPU only).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-moe
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs ./ tatooine:intarsiaml-ink-hip-moe/ < /dev/null || exit 90
exec ssh -o BatchMode=yes tatooine 'bash ~/intarsiaml-ink-hip-moe/.scratch/build-cli.sh' < /dev/null
