#!/bin/bash
# usage: cc-build.sh <tree> 
source ~/.cargo/env
export CUDA_PATH=/usr/local/cuda
cd /tmp/$1 && CARGO_TARGET_DIR=/tmp/$1-target cargo build --release -p intarsia-cli --features cuda > /tmp/$1-build.log 2>&1; echo "EXIT $?" >> /tmp/$1-build.log
