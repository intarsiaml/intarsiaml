#!/usr/bin/env bash
# Lane T2 (from the Mac): put a committed tree on tatooine as ~/intarsiaml-t2-<name> (git archive, so exactly <rev>) and copy
# P1's 27B golden (tatooine:~/intarsiaml-p1/p1-golden, not in git) into it. Prints the build stamp for build.sh.
#   scripts/t2/sync.sh head HEAD      scripts/t2/sync.sh base 0d2d6eda
set -euo pipefail
name=${1:?name}; rev=${2:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml-t2-$name
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst && cp -r ~/intarsiaml-p1/p1-golden ~/$dst/"
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
