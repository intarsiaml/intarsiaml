#!/usr/bin/env bash
ssh tatooine 'docker run --rm --entrypoint bash -e LD_LIBRARY_PATH=/lcpp:/opt/rocm/lib -v /home/josef/bench-r9700/llama.cpp-v0.5.0-hip/build/bin:/lcpp:ro fx6-rocm-llama:7.2.4 -c "ldd /lcpp/libggml-hip.so.0 | grep -i \"not found\"; ldd /lcpp/llama-server | grep -i \"not found\""' </dev/null
