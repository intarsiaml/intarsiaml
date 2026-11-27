#!/usr/bin/env bash
# From endor: copy the worktree to tatooine's lane directory (keeps logs/, .scratch/bin/ there; no docs/results, no target), and
# with `main` also main's source (.scratch/main-src, a `git archive` of the base commit without docs/results).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-sampled
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch/bin --exclude /.scratch/main-src \
  --exclude /docs/results --exclude '/.scratch/*.log' --exclude /.scratch/metal ./ tatooine:intarsiaml-ink-sampled/ < /dev/null
if [ "${1:-}" = main ]; then
  rsync -a --delete .scratch/main-src/ tatooine:intarsiaml-ink-sampled/.scratch/main-src/ < /dev/null
fi
echo "synced $(date +%T)"
