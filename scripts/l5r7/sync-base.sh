#!/usr/bin/env bash
# Lane l5, LOCAL (run inside a --load none fleetq slot): copy origin/main (the no-op gate's base) to tatooine:~/intarsiaml-l5-zbase (git archive; no GPU, no build).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin main
rev=$(git merge-base HEAD origin/main)
git archive --format=tar "$rev" | ssh -o BatchMode=yes tatooine "set -e; rm -rf ~/intarsiaml-l5-zbase.new && mkdir -p ~/intarsiaml-l5-zbase.new && tar -x -C ~/intarsiaml-l5-zbase.new && rm -rf ~/intarsiaml-l5-zbase && mv ~/intarsiaml-l5-zbase.new ~/intarsiaml-l5-zbase && echo $rev > ~/intarsiaml-l5-zbase/.rev"
echo "base = merge-base with origin/main $rev"
