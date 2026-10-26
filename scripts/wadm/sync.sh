#!/usr/bin/env bash
# Lane wadm (from the Mac): put a committed tree on tatooine as ~/intarsiaml-wadm (git archive, so exactly <rev>).
#   scripts/wadm/sync.sh HEAD
set -euo pipefail
rev=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "rm -rf ~/intarsiaml-wadm.new && mkdir -p ~/intarsiaml-wadm.new && tar -x -C ~/intarsiaml-wadm.new && rm -rf ~/intarsiaml-wadm && mv ~/intarsiaml-wadm.new ~/intarsiaml-wadm"
echo "synced $(git rev-parse --short=9 "$rev") to tatooine:~/intarsiaml-wadm"
