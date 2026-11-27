#!/usr/bin/env bash
# Mac side, inside `fleetq run --host tatooine-r9700 --load cpu`: (1) the exact tree of main $1 (git archive) into the lane folder, build
# the hip CLI as .scratch/bin/intarsia-$2; (2) sync the branch back (sync.sh) and build .scratch/bin/intarsia-$3 + $MTESTS test exes.
set -uo pipefail
C=${1:?main commit}; MN=${2:?main name}; BN=${3:?branch name}
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip
L=intarsiaml-ink-hip-specflip
R=docs/results/2026-11-26-inkling2-hip-specflip/raw
cd "$W"
[ -z "$(git status --porcelain -- crates kernels recipes Cargo.toml Cargo.lock)" ] || { echo "uncommitted code changes: commit first"; exit 2; }
git archive --format=tar "$C" | ssh -o BatchMode=yes tatooine "tar -x -C ~/$L" || exit 4
rsync -a "$W/$R/" "tatooine:$L/$R/" < /dev/null || exit 5
ssh -o BatchMode=yes tatooine "echo $(git rev-parse "$C") > ~/$L/.scratch/bin/intarsia-$MN.commit; MTESTS= bash ~/$L/$R/build.sh $MN" < /dev/null || exit 6
bash "$W/$R/sync.sh" || exit 7
ssh -o BatchMode=yes tatooine "echo $(git rev-parse HEAD) > ~/$L/.scratch/bin/intarsia-$BN.commit; MTESTS='${MTESTS:-inkling2_sampled_real}' bash ~/$L/$R/build.sh $BN" < /dev/null
