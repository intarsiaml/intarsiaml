#!/usr/bin/env bash
# Mac side, inside `fleetq run --host tatooine-r9700 --load cpu`: put the exact tree of commit $1 (default origin/main cd16cac1b) into
# tatooine:~/intarsiaml-ink-hip-specflip (git archive, no .git; first use: the folder is new), add this lane's raw/ scripts, then setup.sh there (volume seeded from
# ink-dflash2's, build, copy the binary to .scratch/bin/intarsia-$2). Records the commit in .scratch/bin/intarsia-$2.commit.
set -uo pipefail
C=${1:-cd16cac1b}; NAME=${2:-main7}
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip
L=intarsiaml-ink-hip-specflip
R=docs/results/2026-11-26-inkling2-hip-specflip/raw
cd "$W"
git rev-parse --verify "$C^{commit}" || exit 2
ssh -o BatchMode=yes tatooine "mkdir -p ~/$L/.scratch/bin ~/$L/logs ~/$L/$R" < /dev/null || exit 3
git archive --format=tar "$C" | ssh -o BatchMode=yes tatooine "tar -x -C ~/$L" || exit 4
rsync -a "$W/$R/" "tatooine:$L/$R/" < /dev/null || exit 5
ssh -o BatchMode=yes tatooine "echo $(git rev-parse "$C") > ~/$L/.scratch/bin/intarsia-$NAME.commit; NAME=$NAME bash ~/$L/$R/setup.sh" < /dev/null
