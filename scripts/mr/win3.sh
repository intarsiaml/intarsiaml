#!/usr/bin/env bash
# Lane mr, ON tatooine inside a fleetq slot that takes radiance down: one window for
#   (a) lgbench: logits rows kernel register-shape variants (LGR_VARIANT) on the 27B head 248320x5120,
#   (b) mxbench: MXFP4 rows kernel schedule variants (MXR_VARIANT: rows per barrier group 1/2/4, reduce-scatter epilogue off/on).
# The trap starts radiance again. usage: win3.sh "<lgr variants>" "<mxr variants>" "<NxK shapes for mxbench>"
trap 'bash $HOME/mr-hip/scripts/mr/win.sh restore' EXIT
bash $HOME/mr-hip/scripts/mr/win.sh stop
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
LGV=${1:-"0 1 3"}; MXV=${2:-"0 1 2 3 4 5"}; SHAPES=${3:-"17408x5120 5120x17408 5120x6144"}
gpu() { # gpu <env VAR=VAL> <binary> <args...>
  e=$1; b=$2; shift 2
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e HIP_VISIBLE_DEVICES=0 \
    -e "$e" -v "$HOME/mr-hip":/work -w /work "$IMG" "/work/.mrbin/$b" "$@"
}
for v in $LGV; do echo "## lgbench variant $v 248320x5120"; gpu LGR_VARIANT=$v lgbench 248320 5120 40; done
for sh in $SHAPES; do
  echo "## mxbench gemv baseline $sh"; gpu MXR_VARIANT=0 mxbench ${sh%x*} ${sh#*x} 200 gemv
  for v in $MXV; do echo "## mxbench variant $v $sh"; gpu MXR_VARIANT=$v mxbench ${sh%x*} ${sh#*x} 200 rows; done
done
