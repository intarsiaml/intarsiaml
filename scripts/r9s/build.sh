#!/usr/bin/env bash
# Lane r9s: put HEAD of this worktree on tatooine as ~/intarsiaml-tp2-r9s (git archive) and build the HIP CLI there
# (CPU only, no GPU), via the lane tp2 helpers. Run FROM THE MAC in the worktree, as its own CPU job:
#   fleetq run --host tatooine-r9700 --load cpu --mem 0 --expect 35 --label "R9S build HIP" -- scripts/r9s/build.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
stamp=$(scripts/tp2-hip/sync.sh r9s HEAD)
echo "stamp: $stamp"
ssh -o BatchMode=yes tatooine "TREES='r9s=$stamp' bash -s" < scripts/tp2-hip/build.sh
