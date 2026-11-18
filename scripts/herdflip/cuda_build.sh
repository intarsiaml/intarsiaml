#!/usr/bin/env bash
# usage (on the Spark, under `setsid -w` so $$ is the process-group leader): cuda_build.sh <stamp> <log>
# Builds the CUDA CLI in ~/intarsiaml-herdflip/src with the Spark caps; kills its own process group if MemAvailable < 15 GB.
L=$HOME/intarsiaml-herdflip; R=$L/src; mem=$L/out/mem-build-$1.txt
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
me=$$
( while kill -0 "$me" 2>/dev/null; do
    a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); echo "$(date +%T) $a" >> "$mem"
    if [ "$a" -lt 15360 ]; then echo "KILLED low mem $a" >> "$mem"; kill -9 -- "-$me"; fi
    sleep 5
  done ) &
cd "$R" || exit 3
env INTARSIA_BUILD_STAMP="$1" CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli -j 4 > "$2" 2>&1
echo "exit $?" >> "$2"
