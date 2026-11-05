#!/usr/bin/env bash
# Lane lane-ds: sync the worktree to spark1 ~/intarsiaml-ds/src and build the CUDA binary. Run inside a fleetq --load cpu job.
set -euo pipefail
WT=/opt/code/intarsiaml/.worktrees/device-sampling
ssh spark1 'mkdir -p ~/intarsiaml-ds/src'
rsync -rlpt --delete --exclude .git --exclude .scratch --exclude '/target*' --exclude .worktrees --exclude docs/results "$WT"/ spark1:intarsiaml-ds/src/
ssh spark1 'cd ~/intarsiaml-ds/src && MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 ~/.cargo/bin/cargo build --release --locked --features cuda -p intarsia-cli 2>&1 | tail -15; ls -la target/release/intarsia; md5sum target/release/intarsia'
