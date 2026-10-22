#!/usr/bin/env bash
# Lane FX5c, second prompt (long_code_4k) in ONE spark1 slot: CUDA generates the continue sequence and dumps its rows, then the
# CPU oracle runs on that sequence (rows scored offline by score.py). Endor was blocked behind the #4051 reservation.
#   fleetq run --host spark1 --mem 45 --expect 40 --label "..." -- env DUMPS=<dir> scripts/fx5c/code-chain.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
RAW=docs/results/2026-10-22-fx5c-reuse-vs-oracle/raw
BUILD=0 PROMPT=long_code_4k OUT=cuda-code scripts/fx5c/cuda-dump.sh || exit 1
uv run --project tools python scripts/fx5c/make_oracle_input.py $RAW/cuda-code/dump/seq.json /opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors "$DUMPS/oracle-code-input.safetensors" || exit 1
ORACLE_OUT=oracle-code scripts/fx5c/oracle.sh "$DUMPS/oracle-code-input.safetensors"
