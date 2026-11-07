#!/usr/bin/env bash
# Lane ag1: copy of scripts/dt1 (lane dt1), repointed to ~/intarsiaml-ag1, volume intarsia-cargo-ag1, CLI ag1-intarsia, container ag1-serve, output docs/results/2026-11-07-spec-auto-graph/raw.
# Lane dh (from the Mac): put a committed tree on tatooine as ~/intarsiaml-ag1 (git archive, so exactly <rev>). Keeps the lane's
# own files inside the tree: .ag1/ (scripts, logs, window output) and the built CLIs ag1-intarsia* (the lane uses no other folder).
# tar -m: every file gets the extraction time (git archive carries commit times, older than an earlier build of another rev: cargo then
# skips the changed crates). Prints the build stamp for build.sh (TREES=dev=<stamp>).
#   scripts/ag1/sync.sh HEAD      scripts/ag1/sync.sh origin/main
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-ag1 && find ~/intarsiaml-ag1 -mindepth 1 -maxdepth 1 ! -name .ag1 ! -name .ag1 ! -name "ag1-intarsia*" ! -name "ag1-intarsia*" -exec rm -rf {} + && tar -x -m -C ~/intarsiaml-ag1'
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
