#!/usr/bin/env bash
# Lane lb, ON tatooine inside a fleetq job (--load cpu): device assembly + resource usage of kernels/hip/<f>.hip for one tree, with the
# build.rs flags (gdn.hip in CU mode), into ~/intarsiaml-lb/isa/<tree>/<f>.{s,res.txt}.   isa.sh <tree> <f> [<f> ...]   (f without .hip)
set -uo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
t=${1:?tree}; shift
mkdir -p "$HOME/intarsiaml-lb/isa/$t"
for f in "$@"; do
  cu=""; [ "$f" = gdn ] && cu="-mcumode"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HOME/intarsiaml-lb/$t":/work -v "$HOME/intarsiaml-lb/isa/$t":/out -w /work "$IMG" \
    hipcc -S -O3 -std=c++17 -fPIC $cu --offload-arch=gfx1201 --cuda-device-only -Rpass-analysis=kernel-resource-usage -I /work/kernels \
    -o /out/$f.s /work/kernels/hip/$f.hip > "$HOME/intarsiaml-lb/isa/$t/$f.res.txt" 2>&1 < /dev/null
  echo "$t/$f: rc=$? $(grep -c 's_barrier_signal' "$HOME/intarsiaml-lb/isa/$t/$f.s") s_barrier_signal, $(grep -c global_inv "$HOME/intarsiaml-lb/isa/$t/$f.s") global_inv"
done
