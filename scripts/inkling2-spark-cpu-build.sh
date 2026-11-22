#!/usr/bin/env bash
# ink-serve: sync this worktree to spark1 ~/intarsiaml-ink-serve/src and build the CPU CLI (no GPU features). Run under:
#   fleetq run --host spark1 --load cpu --mem 40 --expect 30 --label "intarsiaml inkling serve: cpu build" -- bash scripts/inkling2-spark-cpu-build.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
L='~/intarsiaml-ink-serve'; R="$L/src"
rsh() { ssh -n -o BatchMode=yes spark1 "export PATH=\$HOME/.cargo/bin:\$PATH; export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R $L/bin $L/out $L/model && rm -rf $L/stage && mkdir -p $L/stage"
( cd "$REPO" && COPYFILE_DISABLE=1 git ls-files -z --cached --others --exclude-standard | COPYFILE_DISABLE=1 tar --no-xattrs --null -T - -cf - ) | ssh -o BatchMode=yes spark1 "tar --warning=no-unknown-keyword -x -C $L/stage"
rsh "find $L/stage -name '._*' -delete; rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.toml' \) -print0 | xargs -0 touch"
date +%T
rsh "cd $R && env INTARSIA_BUILD_STAMP=ink-serve-$(git -C "$REPO" rev-parse --short=12 HEAD) CARGO_BUILD_JOBS=3 cargo build --release --locked -p intarsia-cli > $L/out/build.log 2>&1; grep -E '^error|Finished' $L/out/build.log | tail -20; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1"
rsh "df -h ~ | tail -1; free -g | sed -n 2p"
date +%T
