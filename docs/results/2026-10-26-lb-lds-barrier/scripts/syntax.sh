#!/usr/bin/env bash
# Lane lb, ON tatooine inside a fleetq job (--load cpu): compile every kernels/hip/*.hip of a tree with the build.rs flags (object
# discarded) so an error shows before the long build queues. usage: syntax.sh [tree=dev] [file.hip ...]
set -uo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
t=${1:-dev}; shift || true
cd "$HOME/intarsiaml-lb/$t/kernels/hip" || exit 1
files=("$@"); [ ${#files[@]} -eq 0 ] && files=(*.hip)
for f in "${files[@]}"; do
  cu=""; [ "$f" = gdn.hip ] && cu="-mcumode"
  out=$(docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HOME/intarsiaml-lb/$t":/work -w /work "$IMG" \
    hipcc -c -O3 -std=c++17 -fPIC -Wall -Wno-unused-function $cu --offload-arch=gfx1201 -I /work/kernels -o /tmp/lb-syntax.o "/work/kernels/hip/$f" 2>&1 < /dev/null | grep -E "rror" | head -5)
  echo "$f ${out:-ok}"
done
