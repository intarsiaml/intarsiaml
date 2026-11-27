#!/usr/bin/env bash
# Mac side, inside `fleetq run --host tatooine-r9700 --load cpu` (lane ink-plain-tax): build the test executable at the test-only commit
# (red: the test on 418c122bd's code) and at the fix (green, plus the cpu tiny suites).
set -uo pipefail
RED=${RED:?}; GREEN=${GREEN:?}
R=docs/results/2026-11-27-inkling2-graph-taps/raw
D=/opt/code/intarsiaml/.worktrees/ink-plain-tax/$R
bash $D/sync.sh $RED || exit 3
ssh -o BatchMode=yes tatooine "bash ~/intarsiaml-ink-plain-tax/$R/build.sh red" < /dev/null || exit 5
bash $D/sync.sh $GREEN || exit 3
ssh -o BatchMode=yes tatooine "CPU=1 bash ~/intarsiaml-ink-plain-tax/$R/build.sh green" < /dev/null || exit 6
