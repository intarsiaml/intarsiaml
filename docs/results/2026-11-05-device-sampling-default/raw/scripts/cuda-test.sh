#!/usr/bin/env bash
# Lane lane-ds: the in-process CUDA draws test on spark1 with the new default (no INTARSIA_DEVICE_SAMPLING set). Inside a fleetq --quiet gpu,cpu slot.
set -uo pipefail
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
ssh -n spark1 "cd ~/intarsiaml-ds/src && MAX_JOBS=4 INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 ~/.cargo/bin/cargo test -p intarsia-models --release --test batch_cuda_real real_35b_device_sampling_draws_the_host_tokens -- --test-threads=1 --nocapture 2>&1 | tail -25"
