#!/usr/bin/env bash
# Mac side, inside `fleetq run --host tatooine-r9700 --load cpu`: sync the worktree (committed state; sync.sh) to tatooine's lane folder,
# build the hip CLI as .scratch/bin/intarsia-$1 plus the test executables in $MTESTS (default inkling2_sampled_real), record the commit.
set -uo pipefail
NAME=${1:?name}
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip
L=intarsiaml-ink-hip-specflip
R=docs/results/2026-11-26-inkling2-hip-specflip/raw
cd "$W"
[ -z "$(git status --porcelain -- crates kernels recipes Cargo.toml Cargo.lock)" ] || { echo "uncommitted code changes: commit first"; exit 2; }
bash "$W/$R/sync.sh" || exit 3
ssh -o BatchMode=yes tatooine "echo $(git rev-parse HEAD) > ~/$L/.scratch/bin/intarsia-$NAME.commit; MTESTS='${MTESTS:-inkling2_sampled_real}' bash ~/$L/$R/build.sh $NAME" < /dev/null
