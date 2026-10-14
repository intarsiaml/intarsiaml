#!/bin/bash
source ~/.cargo/env
export CUDA_PATH=/usr/local/cuda
mkdir -p /tmp/cc-logs
cd /tmp/intarsia-cc-head && CARGO_TARGET_DIR=/tmp/intarsia-cc-head-target cargo test --release -p intarsia-backend-cuda > /tmp/cc-logs/cargo-test-cuda.log 2>&1; echo "EXIT $?" >> /tmp/cc-logs/cargo-test-cuda.log
