#!/usr/bin/env bash
# window stage of lane ink-hip-mix (ON tatooine, via window.sh): the GPU gates of the mix / projection kernels on card 0.
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-mix/docs/results/2026-11-25-inkling2-hip-mix/raw
export INTARSIA_REQUIRE_GPU=1
for t in "$@"; do
  echo "=== $t"
  bash "$R/run.sh" bash -c "$t" || echo "=== FAILED ($?): $t"
done
