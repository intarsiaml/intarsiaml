#!/usr/bin/env bash
# Lane mxr (from the Mac): put origin/main and the branch head on spark1 as ~/intarsiaml-mxr-cuda/src-{base,dev} (git archive)
# for scripts/mxr/cuda-noop.sh. Then:
#   fleetq run --host spark1 --mem 40 --expect 70 --label "..." -- ssh spark1 'bash ~/intarsiaml-mxr-cuda/src-dev/scripts/mxr/cuda-noop.sh'
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin main
for spec in base=origin/main dev=HEAD; do
  n=${spec%%=*}; r=${spec#*=}
  git archive --format=tar "$r" | ssh -o BatchMode=yes spark1 "rm -rf ~/intarsiaml-mxr-cuda/src-$n && mkdir -p ~/intarsiaml-mxr-cuda/src-$n && tar -x -C ~/intarsiaml-mxr-cuda/src-$n"
  echo "$n=$(git rev-parse --short "$r")"
done
