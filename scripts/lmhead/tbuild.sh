#!/usr/bin/env bash
# Lane lmhead-fp8: CPU-slot build of this worktree on tatooine (no GPU, radiance stays up). Run FROM THE MAC (endor) inside a slot:
#   fleetq run --host tatooine --load cpu --mem 8 --expect 45 --label "lmhead-fp8: tatooine build" -- scripts/lmhead/tbuild.sh [NAME] [TREE]
# NAME (default lmhead): tree ~/intarsiaml-NAME, cargo volume intarsia-cargo-target-NAME (seeded as a copy of intarsia-cargo-target-g2b-dev
# for an incremental build), release CLI copied to ~/NAME-bin/intarsia (it needs the image's ROCm libs: run it in the dev image, tree at /work).
# TREE: the checkout to ship (default: this one). Removal when the lane is done: scripts/lmhead/tclean.sh NAME.
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; NAME=${1:-lmhead}; T=${2:-$(cd "$S/../.." && pwd)}
SHA=$(git -C "$T" rev-parse --short=9 HEAD)
V=intarsia-cargo-target-$NAME
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .scratch --exclude '*-intarsia' "$T/" tatooine:intarsiaml-$NAME/
ssh -n -o BatchMode=yes tatooine "docker volume inspect $V >/dev/null 2>&1 || { docker volume create $V >/dev/null && docker run --rm -v intarsia-cargo-target-g2b-dev:/s:ro -v $V:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'; }"
ssh -o BatchMode=yes tatooine "set -e; mkdir -p ~/$NAME-bin; export INTARSIA_REPO=\$HOME/intarsiaml-$NAME INTARSIA_TARGET_VOL=$V; D=\$INTARSIA_REPO/deploy/tatooine/dev.sh
  \$D env INTARSIA_BUILD_STAMP='head=$SHA $NAME' CARGO_BUILD_JOBS=20 cargo build --release -p intarsia-cli --features hip > ~/$NAME-bin/build.log 2>&1; echo build exit \$?; grep -E '^error|Finished' ~/$NAME-bin/build.log | tail -5
  \$D cp /target/release/intarsia /work/$NAME-intarsia && cp \$INTARSIA_REPO/$NAME-intarsia ~/$NAME-bin/intarsia; ls -la ~/$NAME-bin"
