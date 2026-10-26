#!/usr/bin/env bash
# Lane l8, ON tatooine inside a fleetq job that takes radiance down (--quiet gpu,cpu --takes-down radiance-rank0):
#   win.sh bench <N> <binary-name> [args...]   snapshot ~/intarsiaml-l8-src to ~/intarsiaml-l8-w<N>, run .l8bin/<binary> from
#                                              the snapshot on card 0 (HIP_VISIBLE_DEVICES=0); radiance is stopped for the run
#                                              and always started again (trap) and waited for (<= 3 min).
#   win.sh restore                             start radiance and wait for /v1/models (idempotent)
# Every window appends to ~/intarsiaml-l8-w<N>/downtime.log (copy it to the results raw/ dir afterwards).
set -u
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
restore() {
  systemctl --user start radiance
  local i
  for i in $(seq 1 36); do
    curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break
    sleep 5
  done
  echo "[l8] radiance $(systemctl --user is-active radiance) after $((i * 5)) s"
}
cmd=${1:?}; shift
case "$cmd" in
  restore) restore ;;
  bench)
    n=${1:?}; bin=${2:?}; shift 2
    snap=$HOME/intarsiaml-l8-w$n
    rm -rf "$snap" && cp -a "$HOME/intarsiaml-l8-src" "$snap"
    t0=$(date +%T)
    trap 'restore; echo "window w'$n' down $t0 -> $(date +%T)" >> "$snap/downtime.log"; cat "$snap/downtime.log"' EXIT
    systemctl --user stop radiance
    sleep 3
    echo "[l8] radiance $(systemctl --user is-active radiance); window w$n starts $(date +%T)"
    docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" \
      --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video --ipc host --security-opt seccomp=unconfined \
      --user "$(id -u):$(id -g)" -e HOME=/tmp -e HIP_VISIBLE_DEVICES=0 -v "$snap":/work -w /work "$IMG" "/work/.l8bin/$bin" "$@"
    ;;
esac
