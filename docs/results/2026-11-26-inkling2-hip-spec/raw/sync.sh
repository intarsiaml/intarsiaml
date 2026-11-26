#!/usr/bin/env bash
# From the Mac: copy the worktree to tatooine's lane directory (keeps logs/, .scratch/bin/ and .scratch/main-src there), and
# with `main` also main's source (.scratch/main-src, a `git archive` of the base commit) to .scratch/main-src there.
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-spec
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch/bin --exclude /.scratch/main-src ./ tatooine:intarsiaml-ink-hip-spec/ < /dev/null
if [ "${1:-}" = main ]; then
  rsync -a --delete .scratch/main-src/ tatooine:intarsiaml-ink-hip-spec/.scratch/main-src/ < /dev/null
fi
