#!/usr/bin/env bash
# Lane dt1: copy of scripts/dw (lane dw), repointed to ~/intarsiaml-dt1, volume intarsia-cargo-dt1, CLI dt1-intarsia, container dt1-serve, output docs/results/2026-11-06-27b-dflash-hip-tp1/raw.
# Lane dh (from the Mac): put a committed tree on tatooine as ~/intarsiaml-dt1 (git archive, so exactly <rev>). Keeps the lane's
# own files inside the tree: .dt1/ (scripts, logs, window output) and the built CLIs dt1-intarsia* (the lane uses no other folder).
# tar -m: every file gets the extraction time (git archive carries commit times, older than an earlier build of another rev: cargo then
# skips the changed crates). Prints the build stamp for build.sh (TREES=dev=<stamp>).
#   scripts/dt1/sync.sh HEAD      scripts/dt1/sync.sh origin/main
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-dt1 && find ~/intarsiaml-dt1 -mindepth 1 -maxdepth 1 ! -name .dt1 ! -name .dt1 ! -name "dt1-intarsia*" ! -name "dt1-intarsia*" -exec rm -rf {} + && tar -x -m -C ~/intarsiaml-dt1'
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
