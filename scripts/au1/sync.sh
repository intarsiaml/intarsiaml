#!/usr/bin/env bash
# Lane au1: copy of scripts/dt1 (lane dt1), repointed to ~/intarsiaml-au1, volume intarsia-cargo-au1, CLI au1-intarsia, container au1-serve, output docs/results/2026-11-06-spec-auto-tp1/raw.
# Lane dh (from the Mac): put a committed tree on tatooine as ~/intarsiaml-au1 (git archive, so exactly <rev>). Keeps the lane's
# own files inside the tree: .au1/ (scripts, logs, window output) and the built CLIs au1-intarsia* (the lane uses no other folder).
# tar -m: every file gets the extraction time (git archive carries commit times, older than an earlier build of another rev: cargo then
# skips the changed crates). Prints the build stamp for build.sh (TREES=dev=<stamp>).
#   scripts/au1/sync.sh HEAD      scripts/au1/sync.sh origin/main
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-au1 && find ~/intarsiaml-au1 -mindepth 1 -maxdepth 1 ! -name .au1 ! -name .au1 ! -name "au1-intarsia*" ! -name "au1-intarsia*" -exec rm -rf {} + && tar -x -m -C ~/intarsiaml-au1'
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
