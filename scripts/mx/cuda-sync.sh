#!/usr/bin/env bash
# Lane mx (from the Mac): put origin/main and the branch head on spark1 as ~/intarsia-decfuse/mx-src-{base,dev} (git archive)
# for scripts/mx/cuda-noop.sh. Then: fleetq run --host spark1 --mem 40 --expect 60 -- ssh spark1 'bash ~/intarsia-decfuse/mx-src-dev/scripts/mx/cuda-noop.sh'
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for spec in base=origin/main dev=HEAD; do
  n=${spec%%=*}; r=${spec#*=}
  git archive --format=tar "$r" | ssh -o BatchMode=yes spark1 "rm -rf ~/intarsia-decfuse/mx-src-$n && mkdir -p ~/intarsia-decfuse/mx-src-$n && tar -x -C ~/intarsia-decfuse/mx-src-$n"
  echo "$n=$(git rev-parse --short "$r")"
done
