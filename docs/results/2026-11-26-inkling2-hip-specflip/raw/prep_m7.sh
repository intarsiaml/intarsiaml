#!/usr/bin/env bash
# Mac side, inside `fleetq run --host tatooine-r9700 --load cpu` (lane ink-hip-specflip, main7 check, coordinator 02:08): export exactly
# origin/main $1 (git archive, into .scratch/main7-src), sync it to tatooine's lane folder (--delete; logs/ and .scratch/ kept), lay this
# branch's raw/ scripts over it (docs only), build the hip CLI as .scratch/bin/intarsia-main7 and main7's seed harness exe.
set -uo pipefail
SHA=${1:?sha}; NAME=${2:-main7}
W=/opt/code/intarsiaml/.worktrees/ink-hip-specflip; A=$W/.scratch/wt-a; X=$W/.scratch/main7-src
L=intarsiaml-ink-hip-specflip; R=docs/results/2026-11-26-inkling2-hip-specflip/raw
mkdir -p "$X" && git -C "$A" archive "$SHA" | tar -x -C "$X" || exit 3
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch "$X/" tatooine:$L/ < /dev/null || exit 4
rsync -a "$A/$R/" tatooine:$L/$R/ < /dev/null || exit 5
ssh -o BatchMode=yes tatooine "echo '$(git -C "$A" rev-parse "$SHA") origin/main ($NAME, git archive)' > ~/$L/.scratch/bin/intarsia-$NAME.commit; cd ~/$L && grep -c . crates/intarsia-core/src/model.rs > /dev/null && MTESTS='inkling2_sampled_real' bash ~/$L/$R/build.sh $NAME" < /dev/null || exit 6
