#!/usr/bin/env bash
# Flash BQ lever on spark1 inside the caller's slot (vLLM up: relative numbers): build r6-dev, tests/flash_bq.rs (bitwise
# BQ 128 vs 64) + tests/attention.rs (reference sweep), then attention_throughput 35B prefill rows with BQ 64 / 128, ROUNDS.
#   fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round6/flash-ab.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS="--test flash_bq --test attention" "$S/build-test.sh"
TREE=r6-dev NOBUILD=1 STEPS=att ROUNDS=${ROUNDS:-2} ATT_FILTER="prefill 35B|panicked" ATT_ENVS="INTARSIA_CUDA_FLASH_BQ=64;INTARSIA_CUDA_FLASH_BQ=128" "$S/dev.sh"
