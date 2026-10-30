#!/usr/bin/env bash
# Run on the GPU box: compile the microbench binaries of window 1 in parallel
# (bench-main = tree kernels; abl* / w16 = the scratch/abl copy with -D hooks).
set -uo pipefail
T=$HOME/intarsiaml-tp2-k4fp8; cd "$T"
one() {
  local tag=$1 inc=$2; shift 2
  [[ $inc == - ]] && inc=
  INC=$inc bash scripts/k4fp8/bench.sh "$T" "$tag" "$@"
}
export -f one; export T
{
  echo "main -"
  echo "abl0 scratch/abl"
  for a in 1 2 3 4 5 6 7; do echo "abl$a scratch/abl -DK4_ABL=$a"; done
  echo "w16 scratch/abl -DK4_WARPS=16"
} | xargs -P 5 -L 1 bash -c 'one "$@"' _
