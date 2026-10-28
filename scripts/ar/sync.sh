#!/usr/bin/env bash
# Lane ar (from the Mac): put a committed rev on tatooine as ~/intarsiaml-ars (git archive, so exactly <rev>). Prints the build stamp for build.sh.
#   scripts/ar/sync.sh HEAD
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml-ars
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst"
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
