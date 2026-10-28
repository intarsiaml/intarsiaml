#!/usr/bin/env bash
# Lane det (from the Mac): put a committed tree on tatooine as ~/intarsiaml/.worktrees/det (git archive, so exactly <rev>).
# Prints the build stamp for build.sh. After scripts/vat/sync.sh.
#   scripts/det/sync.sh HEAD
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml/.worktrees/det
git archive --format=tar "$rev" | ssh -o BatchMode=yes "${HOST:-tatooine}" "mkdir -p ~/intarsiaml/.worktrees && rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst"
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
