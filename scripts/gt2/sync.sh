#!/usr/bin/env bash
# Lane gt2: copy of scripts/ag1 repointed to ~/intarsiaml-gt2, volume intarsia-cargo-gt2, CLI gt2-intarsia, container gt2-serve, output docs/results/2026-11-07-spec-graph-tp2/raw.
# Lane dh (from the Mac): put a committed tree on tatooine as ~/intarsiaml-gt2 (git archive, so exactly <rev>). Keeps the lane's
# own files inside the tree: .gt2/ (scripts, logs, window output) and the built CLIs gt2-intarsia* (the lane uses no other folder).
# tar -m: every file gets the extraction time (git archive carries commit times, older than an earlier build of another rev: cargo then
# skips the changed crates). Prints the build stamp for build.sh (TREES=dev=<stamp>).
#   scripts/gt2/sync.sh HEAD      scripts/gt2/sync.sh origin/main
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-gt2 && find ~/intarsiaml-gt2 -mindepth 1 -maxdepth 1 ! -name .gt2 ! -name .gt2 ! -name "gt2-intarsia*" ! -name "gt2-intarsia*" -exec rm -rf {} + && tar -x -m -C ~/intarsiaml-gt2'
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
