#!/usr/bin/env bash
# Lane l5, LOCAL (run inside a --load none fleetq slot): copy origin/main (the no-op gate's base) to tatooine:~/intarsiaml-l5-base (git archive; no GPU, no build).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin main
rev=$(git rev-parse origin/main)
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "set -e; rm -rf ~/intarsiaml-l5-base.new && mkdir -p ~/intarsiaml-l5-base.new && tar -x -C ~/intarsiaml-l5-base.new && rm -rf ~/intarsiaml-l5-base && mv ~/intarsiaml-l5-base.new ~/intarsiaml-l5-base && echo $rev > ~/intarsiaml-l5-base/.rev"
echo "base = origin/main $rev"
