#!/usr/bin/env bash
# Lane mh2: copy of scripts/ag1 repointed to ~/intarsiaml-mh2, volume intarsia-cargo-mh2, CLI mh2-intarsia, container mh2-serve, output docs/results/2026-11-07-hip35-mtp-tp2/raw.
# Lane dh (from the Mac): put a committed tree on tatooine as ~/intarsiaml-mh2 (git archive, so exactly <rev>). Keeps the lane's
# own files inside the tree: .mh2/ (scripts, logs, window output) and the built CLIs mh2-intarsia* (the lane uses no other folder).
# tar -m: every file gets the extraction time (git archive carries commit times, older than an earlier build of another rev: cargo then
# skips the changed crates). Prints the build stamp for build.sh (TREES=dev=<stamp>).
#   scripts/mh2/sync.sh HEAD      scripts/mh2/sync.sh origin/main
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-mh2 && find ~/intarsiaml-mh2 -mindepth 1 -maxdepth 1 ! -name .mh2 ! -name .mh2 ! -name "mh2-intarsia*" ! -name "mh2-intarsia*" -exec rm -rf {} + && tar -x -m -C ~/intarsiaml-mh2'
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
