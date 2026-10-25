#!/usr/bin/env bash
# Lane tp2 (from the Mac): put the lane's base (origin/main 2c2d9941) and HEAD on spark1 as ~/intarsia-decfuse/tp2-src-{base,dev}.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for spec in base=2c2d9941 dev=HEAD; do
  n=${spec%%=*}; r=${spec#*=}
  git archive --format=tar "$r" | ssh -o BatchMode=yes spark1 "rm -rf ~/intarsia-decfuse/tp2-src-$n && mkdir -p ~/intarsia-decfuse/tp2-src-$n && tar -x -C ~/intarsia-decfuse/tp2-src-$n"
  echo "$n=$(git rev-parse --short "$r")"
done
