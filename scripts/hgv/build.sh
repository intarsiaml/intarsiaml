#!/usr/bin/env bash
# Lane hgv: put two commits on tatooine and build both (CPU only, no GPU): ~/intarsiaml-tp2-hgvbase (BASE: main's kernels, the new
# test shapes only) and ~/intarsiaml-tp2-hgv (HEAD), each with the HIP crate's test binaries (HIPTEST=1). Run FROM THE MAC in the worktree,
# as its own CPU job:
#   BASE=<rev> fleetq run --host tatooine-r9700 --load cpu --mem 0 --expect 60 --label "HGV build base + hgv HIP" -- scripts/hgv/build.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
base=$(scripts/tp2-hip/sync.sh hgvbase "${BASE:?BASE=<rev>}")
head=$(scripts/tp2-hip/sync.sh hgv HEAD)
echo "stamp base: $base"
echo "stamp hgv: $head"
ssh -o BatchMode=yes tatooine "HIPTEST=1 TREES='hgvbase=$base hgv=$head' bash -s" < scripts/tp2-hip/build.sh
