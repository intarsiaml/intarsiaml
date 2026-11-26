#!/usr/bin/env bash
# endor: rsync the origin/main checkout (.scratch/main-src) to the lane folder on tatooine, keeping its .scratch / logs.
set -euo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-pf2/.scratch/main-src
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch ./ tatooine:intarsiaml-ink-hip-pf2/ < /dev/null
