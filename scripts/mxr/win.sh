#!/usr/bin/env bash
# Lane mxr, ON tatooine inside a fleetq job that takes radiance down (--takes-down radiance --quiet gpu,cpu):
# stop radiance, run the Phase 0 harness on card 0 (timing, edge-case bitwise, rocprofv3), start radiance again and wait
# until it answers. usage: [H=mxr1] [W=w2] win.sh [timing] [edge] [prof]   (default: all three). Logs: ~/intarsiaml-mxr-dev/.mxrbin/$W-*.log
set -o pipefail
D=$HOME/intarsiaml-mxr-dev
H=${H:-mxr1}; W=${W:-w2}
restore() {
  systemctl --user start radiance
  for i in $(seq 1 48); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
  echo "[mxr] radiance $(systemctl --user is-active radiance) after $((i * 5)) s; models: $(curl -s -m 3 http://localhost:8080/v1/models | head -c 120)"
}
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[mxr] radiance $(systemctl --user is-active radiance)"
what=${*:-timing edge prof}
dr() { # dr <image> <cmd...>: card 0, ROCm image
  local img=$1; shift
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e HIP_VISIBLE_DEVICES=0 \
    -v "$D":/work -w /work "$img" "$@"
}
DEV=intarsia-hip-dev:rocm7.2.4-rust1.90.0
PROF=intarsia-hip-prof:rocm7.2.4-rust1.90.0
rocm-smi --showclocks --showtemp 2>/dev/null | grep -E "sclk|mclk|Temperature" | head -8
for w in $what; do case $w in
  timing)
    for sh in 17408x5120 5120x17408 12288x5120 10240x5120 6144x5120 5120x6144 1024x5120; do
      dr $DEV /work/.mxrbin/$H ${sh%x*} ${sh#*x} 200 3 0
    done 2>&1 | tee $D/.mxrbin/$W-timing.log ;;
  edge)
    for sh in 17408x5120 5120x17408 12288x5120 10240x5120 6144x5120 5120x6144 1024x5120 1x5120 9x2080 33x32 300x4096 300x6176 301x64 7x17408; do
      for e in 0 1 2; do dr $DEV /work/.mxrbin/$H ${sh%x*} ${sh#*x} 2 1 $e 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; echo "[mxr] exit $? $sh edge=$e"; done
    done 2>&1 | tee $D/.mxrbin/$W-edge.log ;;
  prof)
    dr $PROF rocprofv3 --kernel-trace --stats -d /work/.mxrbin/$W-prof-kt -o kt -- /work/.mxrbin/$H 17408 5120 50 1 0 1 8 16 2>&1 | tail -2
    dr $PROF rocprofv3 --kernel-trace --stats -d /work/.mxrbin/$W-prof-kt2 -o kt -- /work/.mxrbin/$H 5120 6144 50 1 0 1 8 16 2>&1 | tail -2
    true ;;
esac; done
