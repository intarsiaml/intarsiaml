#!/usr/bin/env bash
# From endor (lane ink-sampled-adapt): copy this worktree to tatooine's lane folder (keeps logs/, .scratch/bin/ there; no docs/results,
# no target); with `main` also main's source at the branch's merged base aee06201c (.scratch/main-src, a `git archive`).
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-sampled-adapt
cd $W
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch/bin --exclude /.scratch/main-src \
  --exclude /docs/results --exclude /.scratch/logs --exclude /.scratch/sim ./ tatooine:intarsiaml-ink-sampled-adapt/ < /dev/null || exit 1
if [ "${1:-}" = main ]; then
  rm -rf .scratch/main-src && mkdir -p .scratch/main-src
  git -C /opt/code/intarsiaml archive aee06201c -- . ':!docs/results' | tar -x -C .scratch/main-src || exit 1
  rsync -a --delete .scratch/main-src/ tatooine:intarsiaml-ink-sampled-adapt/.scratch/main-src/ < /dev/null || exit 1
fi
echo "synced $(date +%T)"
