#!/usr/bin/env bash
# Lane tp2: compile (CPU slot, ON tatooine) the models GPU tests of the mutant tree ~/intarsiaml-tp2-mut-batch-peer-no-table, so the
# window (stages-w12.sh) only runs them. Warm-copies the batch tree's hiptest volume first.
set -uo pipefail
V=intarsia-cargo-target-tp2-mutb-hiptest
docker volume inspect $V > /dev/null 2>&1 || { docker volume create $V > /dev/null; docker run --rm -v intarsia-cargo-target-tp2-batch-hiptest:/from:ro -v $V:/to alpine sh -c "cp -a /from/. /to/"; }
M=$HOME/intarsiaml-tp2-mut-batch-peer-no-table
INTARSIA_REPO=$M INTARSIA_TARGET_VOL=$V "$M/deploy/tatooine/dev.sh" cargo test --release -p intarsia-models --test batch_cuda_real --no-run 2>&1 | grep -E '^error|Finished|Executable' | tail -3
df -h / | tail -1
