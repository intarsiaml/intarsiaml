#!/usr/bin/env bash
# Build llama.cpp (CUDA, sm_121) on a Spark and convert Qwen3-8B (HF BF16 safetensors) to a BF16 GGUF. Runs inside a fleetq slot
# (non-exclusive, vLLM up):  fleetq run --host spark2 --mem 8 --label "..." -- scripts/standings-fx3/build-llamacpp.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark2}
rsh() { ssh -n -o BatchMode=yes "$HOST" "export PATH=/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh 'mkdir -p ~/fx3; cd ~/fx3; test -d llama.cpp || git clone --depth 1 https://github.com/ggml-org/llama.cpp; cd llama.cpp; git log -1 --format="llama.cpp %H %cd"'
rsh 'cd ~/fx3/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > ~/fx3/llama-cmake.log 2>&1; tail -3 ~/fx3/llama-cmake.log'
rsh 'cd ~/fx3/llama.cpp && cmake --build build --config Release -j 12 --target llama-server llama-bench > ~/fx3/llama-build.log 2>&1; tail -3 ~/fx3/llama-build.log; ls -la build/bin/llama-server'
[[ "${SKIP8:-0}" == 1 ]] && exit 0
SNAP=$(rsh 'ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*' | head -1)
echo "snapshot: $SNAP"
CSNAP="/root/.cache/huggingface/${SNAP#*/.cache/huggingface/}"
rsh "mkdir -p ~/fx3/gguf; test -s ~/fx3/gguf/Qwen3-8B-BF16.gguf || docker run --rm --entrypoint python3 -v \$HOME/fx3/llama.cpp:/lcpp -v \$HOME/.cache/huggingface:/root/.cache/huggingface -v \$HOME/fx3/gguf:/out -e PYTHONPATH=/lcpp/gguf-py vllm/vllm-openai:nightly-aarch64 /lcpp/convert_hf_to_gguf.py $CSNAP --outtype bf16 --outfile /out/Qwen3-8B-BF16.gguf 2>&1 | tail -5; ls -la ~/fx3/gguf"
