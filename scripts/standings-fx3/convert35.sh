#!/usr/bin/env bash
# Try to convert the ModelOpt NVFP4 checkpoint nvidia/Qwen3.6-35B-A3B-NVFP4 to a GGUF keeping NVFP4 (outtype auto) with llama.cpp's converter.
#   fleetq run --host spark2 --mem 8 --expect 20 --label "..." -- scripts/standings-fx3/convert35.sh
set -u
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark2}
rsh() { ssh -n -o BatchMode=yes "$HOST" "export PATH=/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SNAP=$(rsh 'ls -d ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1')
CSNAP="/root/.cache/huggingface/${SNAP#*/.cache/huggingface/}"
echo "snapshot: $SNAP"
rsh "mkdir -p ~/fx3/gguf; docker run --rm --entrypoint python3 -v \$HOME/fx3/llama.cpp:/lcpp -v \$HOME/.cache/huggingface:/root/.cache/huggingface -v \$HOME/fx3/gguf:/out -e PYTHONPATH=/lcpp/gguf-py vllm/vllm-openai:nightly-aarch64 /lcpp/convert_hf_to_gguf.py $CSNAP --outtype auto --outfile /out/Qwen3.6-35B-A3B-NVFP4.gguf 2>&1 | tail -25; ls -la ~/fx3/gguf"
