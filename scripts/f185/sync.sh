#!/usr/bin/env bash
# Lane f185 (from the Mac): put a committed rev on a box as ~/intarsiaml-f185/src-<name> (git archive, so exactly <rev>). Everything the lane keeps on a box
# lives under ~/intarsiaml-f185 (and the docker volume intarsia-cargo-f185 on tatooine).
#   HOST=tatooine scripts/f185/sync.sh base fac1d6541      HOST=tatooine scripts/f185/sync.sh dev HEAD      HOST=spark1 scripts/f185/sync.sh dev HEAD
# base = the first parent of PR #185's merge (ac9ea9044^1 = fac1d6541), the last main before the PR.
set -euo pipefail
name=${1:?name}; rev=${2:?rev}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes "${HOST:?HOST}" "mkdir -p ~/intarsiaml-f185 && rm -rf ~/intarsiaml-f185/src-$name.new && mkdir ~/intarsiaml-f185/src-$name.new && tar -x -C ~/intarsiaml-f185/src-$name.new && rm -rf ~/intarsiaml-f185/src-$name && mv ~/intarsiaml-f185/src-$name.new ~/intarsiaml-f185/src-$name"
echo "synced $name = $(git rev-parse --short=9 "$rev") on $HOST"
