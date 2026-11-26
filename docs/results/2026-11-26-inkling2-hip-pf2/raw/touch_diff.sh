#!/usr/bin/env bash
# endor: after an rsync to tatooine (which keeps mtimes), touch there every file that differs between main 40b85a391 and the
# branch HEAD, so cargo / build.rs rebuild them whichever tree was synced last (window 4 found pf2d built stale = main2).
set -euo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-pf2
files=$(git diff --name-only 40b85a391 HEAD -- kernels crates src | tr '\n' ' ')
ssh -o BatchMode=yes tatooine "cd ~/intarsiaml-ink-hip-pf2 && for f in $files; do [ -e \"\$f\" ] && touch \"\$f\"; done; echo touched \$(echo $files | wc -w) files" < /dev/null
