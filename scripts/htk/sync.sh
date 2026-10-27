#!/usr/bin/env bash
# Lane htk (from the Mac): put a committed rev on tatooine as ~/intarsiaml-htk/src (git archive, so exactly <rev>).
# Everything the lane keeps on tatooine lives under ~/intarsiaml-htk and the docker volume intarsia-cargo-htk.
#   scripts/htk/sync.sh HEAD
set -euo pipefail
rev=${1:-HEAD}
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "mkdir -p ~/intarsiaml-htk && rm -rf ~/intarsiaml-htk/src.new && mkdir ~/intarsiaml-htk/src.new && tar -x -C ~/intarsiaml-htk/src.new && { [ -d ~/intarsiaml-htk/src/.bin ] && mv ~/intarsiaml-htk/src/.bin ~/intarsiaml-htk/src.new/.bin; true; } && rm -rf ~/intarsiaml-htk/src && mv ~/intarsiaml-htk/src.new ~/intarsiaml-htk/src"
echo "synced $(git rev-parse --short=9 "$rev") to tatooine:~/intarsiaml-htk/src"
