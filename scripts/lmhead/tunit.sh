#!/usr/bin/env bash
# Lane lmhead-fp8: re-sync the tree to tatooine (tbuild.sh's) and run cargo test in the dev image (cpu only). tunit.sh NAME <cargo test args...>
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; T="$(cd "$S/../.." && pwd)"; NAME=$1; shift
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .scratch --exclude '*-intarsia' "$T/" tatooine:intarsiaml-$NAME/
ssh -o BatchMode=yes tatooine "export INTARSIA_REPO=\$HOME/intarsiaml-$NAME INTARSIA_TARGET_VOL=intarsia-cargo-target-$NAME; \$INTARSIA_REPO/deploy/tatooine/dev.sh cargo test --release $* 2>&1"
