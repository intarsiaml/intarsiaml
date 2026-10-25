#!/usr/bin/env bash
# Lane ac (from the Mac): put a committed tree on tatooine as ~/intarsiaml-ac (git archive, exactly <rev>) and copy the 27B goldens
# (tatooine:~/intarsiaml-tp2-batch/p1-golden, not in git) into it. Prints the build stamp for build.sh.
#   scripts/ac/sync.sh 5241377b
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml-ac
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst && cp -r ~/intarsiaml-tp2-batch/p1-golden ~/$dst/ && mkdir -p ~/$dst/ac-out"
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
