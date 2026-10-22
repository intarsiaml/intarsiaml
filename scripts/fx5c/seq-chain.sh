#!/usr/bin/env bash
# Lane FX5c: freeze a continue sequence. In ONE spark1 slot: CUDA generates the sequence (PROMPT at prefill chunk CHUNK, the
# split follows from the chunk) and dumps its rows, then the CPU oracle runs on it. Outputs raw/$OUT (dump, seq.json) and raw/$OUT-oracle.
#   fleetq run --host spark1 --mem 45 --expect 40 --load cpu --label "..." -- env DUMPS=<dir> PROMPT=long_docs_4k CHUNK=2048 OUT=cuda-s3072 scripts/fx5c/seq-chain.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
: "${DUMPS:?}"; : "${OUT:?}"; PROMPT=${PROMPT:-long_docs_4k}; CHUNK=${CHUNK:-1024}
RAW=docs/results/2026-10-22-fx5c-reuse-vs-oracle/raw
BUILD=${BUILD:-0} PROMPT=$PROMPT CHUNK=$CHUNK OUT=$OUT scripts/fx5c/cuda-dump.sh || exit 1
G=${GOLDEN:-/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors}
uv run --project tools python scripts/fx5c/make_oracle_input.py $RAW/$OUT/dump/seq.json "$G" "$DUMPS/$OUT-oracle-input.safetensors" || exit 1
ORACLE_OUT=$OUT-oracle scripts/fx5c/oracle.sh "$DUMPS/$OUT-oracle-input.safetensors"
