#!/usr/bin/env bash
# From the Mac (lane ink-plain-tax): put commit $1 of this repo into tatooine's lane folder (keeps logs/ and .scratch/ there).
set -uo pipefail
C=${1:?commit}
cd /opt/code/intarsiaml/.worktrees/ink-plain-tax
ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-ink-plain-tax/logs ~/intarsiaml-ink-plain-tax/.scratch/bin && cd ~/intarsiaml-ink-plain-tax && find . -mindepth 1 -maxdepth 1 ! -name logs ! -name .scratch -exec rm -rf {} +' < /dev/null || exit 3
git archive "$C" | ssh -o BatchMode=yes tatooine 'tar -x -C ~/intarsiaml-ink-plain-tax' || exit 4
# the lane scripts as in the working tree (the red commit predates them)
tar -c docs/results/2026-11-27-inkling2-graph-taps/raw | ssh -o BatchMode=yes tatooine 'tar -x -C ~/intarsiaml-ink-plain-tax' || exit 4
ssh -o BatchMode=yes tatooine "echo $(git rev-parse --short $C) > ~/intarsiaml-ink-plain-tax/.scratch/commit" < /dev/null
