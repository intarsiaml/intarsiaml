#!/usr/bin/env bash
# usage (on the Spark, under `setsid -w` so $$ is the process-group leader): build.sh <stamp>
# Builds origin/main's CUDA CLI in ~/intarsiaml-flipev2/src with the Spark caps; kills its own process group if MemAvailable < 15 GB.
L=$HOME/intarsiaml-flipev2; R=$L/src; mem=$L/out/mem-build.txt
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
me=$$
( while kill -0 "$me" 2>/dev/null; do
    a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); echo "$(date +%T) $a" >> "$mem"
    if [ "$a" -lt 15360 ]; then echo "KILLED low mem $a" >> "$mem"; kill -9 -- "-$me"; fi
    sleep 5
  done ) &
cd "$R" || exit 3
env INTARSIA_BUILD_STAMP="$1" CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli -j 3 > "$L/out/build.log" 2>&1
echo "exit $?" >> "$L/out/build.log"
