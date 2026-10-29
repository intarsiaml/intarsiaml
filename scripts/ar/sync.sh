#!/usr/bin/env bash
# Lane ars (from the Mac): put a committed rev on tatooine (git archive, so exactly <rev>). Prints the build stamp for build.sh.
#   scripts/ar/sync.sh <rev>              -> ~/intarsiaml-ars               (replaces the whole directory, .ars included)
#   scripts/ar/sync.sh <rev> btree        -> ~/intarsiaml-ars/.ars/btree    (the second tree, built beside the first by build-ab.sh)
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml-ars
if [[ -n ${2:-} ]]; then
  git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "rm -rf ~/$dst/.ars/$2 && mkdir -p ~/$dst/.ars/$2 && tar -x -C ~/$dst/.ars/$2"
else
  git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst"
fi
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
