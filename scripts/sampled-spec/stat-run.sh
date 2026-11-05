#!/usr/bin/env bash
# Statistical exactness gate of sampled speculation (docs/results/2026-11-05-sampled-spec.md, rule 2):
# stat-run.sh <test binary of tests/sampled_spec_real.rs> <model dir> <out dir> [n per arm] [parallel]
# Runs configs A and B x arms off / reject / match, n seeds each (disjoint seed ranges per arm), in shards of n / P.
set -euo pipefail
bin=$1; model=$2; out=$3; n=${4:-3000}; par=${5:-8}
mkdir -p "$out"
tasks=()
for cfg in A B; do
  for arm in 0 reject match; do
    case $arm in 0) base=0;; reject) base=1000000;; match) base=2000000;; esac
    per=$(( (n + par - 1) / par ))
    for ((s = 0; s < par; s++)); do
      seed0=$(( base + s * per )); cnt=$(( s == par - 1 ? n - s * per : per ))
      tasks+=("$cfg $arm $seed0 $cnt")
    done
  done
done
printf '%s\n' "${tasks[@]}" | xargs -P "$par" -L 1 bash -c '
  cfg=$0 arm=$1 seed0=$2 cnt=$3
  INTARSIA_TEST_REAL_MODEL=1 INTARSIA_SS_MODEL='"$model"' INTARSIA_SPEC_SAMPLED=$arm INTARSIA_SS_CFG=$cfg \
  INTARSIA_SS_SEED0=$seed0 INTARSIA_SS_N=$cnt INTARSIA_SS_OUT='"$out"'/$cfg-$arm-$seed0.txt \
  '"$bin"' --nocapture --test-threads 1 >'"$out"'/$cfg-$arm-$seed0.log 2>&1 || echo "FAILED $cfg $arm $seed0"'
echo "all shards done"
