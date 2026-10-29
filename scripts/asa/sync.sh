#!/usr/bin/env bash
# Lane asa (from the Mac): put a committed tree on tatooine (or HOST=spark1) as ~/intarsiaml-asa-<name> (git archive, so exactly <rev>).
# Prints the build stamp for build.sh (TREES=<name>=<stamp>). After scripts/tp2-hip/sync.sh.
#   scripts/asa/sync.sh dev HEAD      scripts/asa/sync.sh base origin/main
set -euo pipefail
name=${1:?name}; rev=${2:?rev}
cd "$(git rev-parse --show-toplevel)"
dst=intarsiaml-asa
git archive --format=tar "$rev" | ssh -o BatchMode=yes "${HOST:-tatooine}" "rm -rf ~/$dst.new && mkdir -p ~/$dst.new && tar -x -C ~/$dst.new && rm -rf ~/$dst && mv ~/$dst.new ~/$dst"
tree=$(git rev-parse "$rev^{tree}")
echo "head=$(git rev-parse --short=7 "$rev")_code-tree=${tree:0:12}_dirty=no"
