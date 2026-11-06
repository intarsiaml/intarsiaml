#!/usr/bin/env bash
# Lane dh (from the Mac): put a committed tree on tatooine as ~/intarsiaml-dh (git archive, so exactly <rev>). Keeps the lane's
# own files inside the tree: .dh/ (scripts, logs, window output) and the built CLIs dh-intarsia* (the lane uses no other folder).
# Prints the build stamp for build.sh (TREES=dev=<stamp>).
#   scripts/dh/sync.sh HEAD      scripts/dh/sync.sh origin/main
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-dh && find ~/intarsiaml-dh -mindepth 1 -maxdepth 1 ! -name .dh ! -name "dh-intarsia*" -exec rm -rf {} + && tar -x -C ~/intarsiaml-dh'
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
