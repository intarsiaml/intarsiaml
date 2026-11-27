#!/usr/bin/env bash
# From the Mac: copy the worktree to tatooine's lane directory (keeps logs/ and .scratch/bin/ there); the lane's raw/*.sh and
# *.py go to .scratch/ (window.sh, run.sh, stages).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-bpf
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch --exclude /docs/results ./ tatooine:intarsiaml-ink-hip-bpf/ < /dev/null
rsync -a docs/results/2026-11-26-inkling2-hip-bpf/raw/*.sh docs/results/2026-11-26-inkling2-hip-bpf/raw/*.py tatooine:intarsiaml-ink-hip-bpf/.scratch/ < /dev/null 2>/dev/null || rsync -a docs/results/2026-11-26-inkling2-hip-bpf/raw/*.sh tatooine:intarsiaml-ink-hip-bpf/.scratch/ < /dev/null
