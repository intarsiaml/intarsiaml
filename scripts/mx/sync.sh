#!/usr/bin/env bash
# Lane mx (from the Mac): put a committed tree on tatooine as ~/intarsiaml-mx-<name> (git archive, exactly <rev>), plus P1's
# 27B golden (tatooine:~/intarsiaml-p1/p1-golden, not in git) and the hp prompts. Prints the build stamp for build.sh.
#   scripts/mx/sync.sh dev HEAD      scripts/mx/sync.sh base origin/main
set -euo pipefail
name=${1:?name}; rev=${2:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml-mx-$name
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst && cp -r ~/intarsiaml-p1/p1-golden ~/$dst/ && mkdir -p ~/$dst/mx-out && cp ~/hp-prompts/prompt*.txt ~/$dst/mx-out/"
tree=$(git rev-parse "$rev^{tree}")
stamp="$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
echo "$stamp" | ssh -o BatchMode=yes tatooine "cat > ~/$dst/.stamp"
echo "$name=$stamp"
