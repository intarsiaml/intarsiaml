#!/usr/bin/env bash
# Lane wmr: put HEAD of this worktree on tatooine as ~/intarsiaml-tp2-wmr (git archive) and build the HIP CLI plus the HIP crate's GPU tests
# (rows_exact etc.) and the intarsia-models batch tests, compiled only (CPU, no GPU). Run FROM THE MAC in the worktree, as its own CPU job:
#   fleetq run --host tatooine-r9700 --load cpu --mem 0 --expect 45 --label "WMR build HIP + tests" -- scripts/wmr/build.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
stamp=$(scripts/tp2-hip/sync.sh wmr HEAD)
echo "stamp: $stamp"
ssh -o BatchMode=yes tatooine "HIPTEST=1 HIPTEST_MODELS=1 TREES='wmr=$stamp' bash -s" < scripts/tp2-hip/build.sh
