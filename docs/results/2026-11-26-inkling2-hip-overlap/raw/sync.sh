#!/usr/bin/env bash
# From the Mac (lane ink-hip-overlap): copy this worktree to tatooine's lane directory, which is ink-hip-bpf's (the same cargo target
# volume, so builds stay incremental; logs/ and .scratch/bin/ are kept); raw/*.sh and *.py go to .scratch/.
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-overlap
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch --exclude /docs/results ./ tatooine:intarsiaml-ink-hip-bpf/ < /dev/null
rsync -a docs/results/2026-11-26-inkling2-hip-overlap/raw/*.sh tatooine:intarsiaml-ink-hip-bpf/.scratch/ < /dev/null
