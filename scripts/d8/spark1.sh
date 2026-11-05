#!/usr/bin/env bash
# Lane d8: build and gates on spark1 (CUDA, GB10). Everything lives in ~/intarsiaml-d8-{src,base,target,target-base,out} on spark1.
# Started from the Mac that holds the fleetq slot:
#   scripts/d8/spark1.sh build   (fleetq run --host spark1 --load cpu --expect 40 ...)  rsync this tree + export origin/main, build both CLIs
#   scripts/d8/spark1.sh gates   (fleetq run --host spark1 --quiet gpu,cpu --mem 28 --expect 40 ...)  ids, fallbacks, plan show
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
BASEREV=${BASEREV:-602a1fcfe}
ENVP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH'
case "${1:?}" in
build)
  rsync -a --delete --exclude target --exclude .git --exclude .worktrees --exclude .scratch --exclude tools/.venv --exclude 'docs/results' "$REPO/" spark1:intarsiaml-d8-src/ || exit 1
  git -C "$REPO" archive "$BASEREV" | ssh spark1 'rm -rf ~/intarsiaml-d8-base && mkdir -p ~/intarsiaml-d8-base && tar -x -C ~/intarsiaml-d8-base' || exit 1
  ssh spark1 "$ENVP; for t in target target-base; do [ -d \$HOME/intarsiaml-d8-\$t ] || cp -a \$HOME/intarsiaml/target \$HOME/intarsiaml-d8-\$t; done; \
    cd \$HOME/intarsiaml-d8-src && INTARSIA_BUILD_STAMP=d8-dev-\$(date +%m%d%H%M%S) CARGO_TARGET_DIR=\$HOME/intarsiaml-d8-target CARGO_BUILD_JOBS=8 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -5; \
    cd \$HOME/intarsiaml-d8-base && INTARSIA_BUILD_STAMP=d8-base-\$(date +%m%d%H%M%S) CARGO_TARGET_DIR=\$HOME/intarsiaml-d8-target-base CARGO_BUILD_JOBS=8 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -5; \
    echo dev-has-key:\$(strings \$HOME/intarsiaml-d8-target/release/intarsia | grep -c dflash_default_draft) base-has-key:\$(strings \$HOME/intarsiaml-d8-target-base/release/intarsia | grep -c dflash_default_draft); \
    strings \$HOME/intarsiaml-d8-target/release/intarsia | grep -m1 d8-dev-; strings \$HOME/intarsiaml-d8-target-base/release/intarsia | grep -m1 d8-base-" ;;
gates)
  rsync -a "$REPO/scripts/d8/gates.sh" spark1:intarsiaml-d8-src/scripts/d8/
  ssh spark1 "bash ~/intarsiaml-d8-src/scripts/d8/gates.sh ${*:2}" ;;
esac
