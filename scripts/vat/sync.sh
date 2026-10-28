#!/usr/bin/env bash
# Lane vat (from the Mac): put a committed tree on tatooine (or HOST=spark1) as ~/intarsiaml-vat-<name> (git archive, so exactly <rev>).
# Prints the build stamp for build.sh (TREES=<name>=<stamp>). After scripts/tp2-hip/sync.sh.
#   scripts/vat/sync.sh dev HEAD      scripts/vat/sync.sh base origin/main
set -euo pipefail
name=${1:?name}; rev=${2:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml-vat
git archive --format=tar "$rev" | ssh -o BatchMode=yes "${HOST:-tatooine}" "rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst"
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
