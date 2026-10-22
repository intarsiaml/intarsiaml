#!/usr/bin/env bash
# fleetq run --host tatooine --mem 0 --label "fx6 build llama HIP runtime image (CPU/net)" -- bash scripts/standings-fx6/build-rocm-llama-image.sh
set -euo pipefail
cd "$(dirname "$0")"
ssh tatooine 'docker build -t fx6-rocm-llama:7.2.4 -' < Dockerfile.rocm-llama 2>&1 | tail -15
ssh tatooine 'docker run --rm --entrypoint bash -e LD_LIBRARY_PATH=/lcpp -v /home/josef/bench-r9700/llama.cpp-v0.5.0-hip/build/bin:/lcpp:ro fx6-rocm-llama:7.2.4 -c "ldd /lcpp/libggml-hip.so.0 | grep -ci \"not found\"; ls /lcpp/llama-server"' </dev/null
