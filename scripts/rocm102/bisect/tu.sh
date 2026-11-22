#!/usr/bin/env bash
# Lane rocm102-bisect (ON tatooine, CPU only): tu.sh <arm> <variant> <tu> [extra hipcc flags...]: compile kernels/hip/<tu>.hip in
# that arm's image with build.rs's flags (+ extras) -> $B/objs/<variant>/<tu>.o, plus the device IR (<tu>.ll, after -O3) and ISA
# (<tu>.s) for gfx1201 -> $B/isa/<variant>/. DEVLIBS=<dir under /work> adds --rocm-device-lib-path=<dir>.
set -u
. "$(dirname "$0")/lib.sh"
a=$1 v=$2 tu=$3; shift 3
dk "$a" bash -c '
  set -e; v=$1 tu=$2 dl=$3; shift 3
  cu=; [[ $tu == gdn || $tu == gdn_chunk ]] && cu=-mcumode
  ex=("$@"); [[ -n $dl ]] && ex+=("--rocm-device-lib-path=$dl")
  F=(-O3 -std=c++17 -fPIC -Wall -Wno-unused-function $cu --offload-arch=gfx1201 -I /work/kernels "${ex[@]}")
  mkdir -p /work/.bisect/objs/$v /work/.bisect/isa/$v
  hipcc -c "${F[@]}" -o /work/.bisect/objs/$v/$tu.o /work/kernels/hip/$tu.hip
  hipcc --offload-device-only -S -emit-llvm "${F[@]}" -o /work/.bisect/isa/$v/$tu.ll /work/kernels/hip/$tu.hip
  hipcc --offload-device-only -S "${F[@]}" -o /work/.bisect/isa/$v/$tu.s /work/kernels/hip/$tu.hip
  echo "$v: hipcc ${F[*]}"' _ "$v" "$tu" "${DEVLIBS:-}" "$@" < /dev/null
