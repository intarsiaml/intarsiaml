#!/usr/bin/env bash
# Lane l7m, ON tatooine inside a fleetq job (--load cpu, a few minutes): compile the new HIP translation unit(s) with the build.rs flags, object
# discarded, so a syntax or type error shows before the long build queues. usage: syntax.sh [file.hip ...] (default moe.hip moe_block_grouped.hip)
set -uo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
files=("$@"); [ ${#files[@]} -eq 0 ] && files=(moe.hip moe_block_grouped.hip)
for f in "${files[@]}"; do
  echo "== $f"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HOME/intarsiaml-l7m-hip":/work -w /work "$IMG" \
    hipcc -c -O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 -I /work/kernels -o /tmp/l7-syntax.o "/work/kernels/hip/$f" 2>&1 < /dev/null | head -40
  echo "rc=${PIPESTATUS[0]}"
done
