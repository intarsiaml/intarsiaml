#!/usr/bin/env bash
# Lane mr, ON tatooine, inside a fleetq slot that takes radiance down: stop / restore radiance, and run the harness.
#   win.sh stop                         systemctl --user stop radiance
#   win.sh restore                      start radiance, wait until it answers (<= 3 min), print its state (idempotent)
#   win.sh run "<variants>" "<NxK..>" [rows|gemv] [base 0|1]     mxbench per shape and MXR_VARIANT on card 0
cmd=${1:?}; shift
case "$cmd" in
  stop) systemctl --user stop radiance; sleep 3; echo "[mr] radiance $(systemctl --user is-active radiance)" ;;
  restore)
    systemctl --user start radiance
    for i in $(seq 1 36); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
    echo "[mr] radiance $(systemctl --user is-active radiance) after $((i * 5)) s" ;;
  run)
    IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
    hb() { # hb <env VAR=VAL...> -- <args>: mxbench inside the dev image (ROCm libraries), card 0
      docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
        --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e HIP_VISIBLE_DEVICES=0 \
        -e MXR_VARIANT="${MXR_VARIANT:-2}" -v "$HOME/mr-hip":/work -w /work "$IMG" /work/.mrbin/mxbench "$@"
    }
    variants=${1:-2}; shapes=${2:-17408x5120}; kernel=${3:-rows}; base=${4:-0}
    for sh in $shapes; do
      [ "$base" = 1 ] && { echo "## gemv baseline $sh"; hb ${sh%x*} ${sh#*x} 200 gemv; }
      for v in $variants; do echo "## variant $v $sh"; MXR_VARIANT=$v hb ${sh%x*} ${sh#*x} 200 $kernel; done
    done ;;
esac
