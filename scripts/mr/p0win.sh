#!/usr/bin/env bash
# Lane mr, ON tatooine inside a fleetq job that takes radiance down: run the P0 ablation (.mrbin/mxp0) on card 0, then start radiance again.
# usage: p0win.sh
trap 'bash $HOME/mr-hip/scripts/mr/win.sh restore' EXIT
bash $HOME/mr-hip/scripts/mr/win.sh stop
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
for sh in 17408x5120 5120x17408 5120x6144; do
  N=${sh%x*}; K=${sh#*x}
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e HIP_VISIBLE_DEVICES=0 \
    -v "$HOME/mr-hip":/work -w /work "$IMG" /work/.mrbin/mxp0 "$N" "$K" 200 3
done
