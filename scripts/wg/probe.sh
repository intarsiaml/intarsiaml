#!/usr/bin/env bash
# Lane wg, ON tatooine: build / run the one-instruction hardware probe (probe_wmma.cpp) in the HIP dev container.
#   probe.sh build   compile only (cpu job; stamps the tree as "probe" on success)
#   probe.sh run     run inside a radiance-down window (as a tail of window D): refuses, without failing the window, unless the build stamp matches the tree
# Output: ~/wg-logs/probe/{probe-crafted.bin,probe-crafted.tags,probe-random.bin,probe-chain.bin,probe-summary.txt}
set -uo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
REPO=${REPO:-$HOME/intarsiaml-wg}
BUILD=${PROBE_BUILD:-$HOME/wg-build-probe}; mkdir -p "$BUILD"  # not ${BUILD}: window callers export BUILD=wg-build-v2 (window D skipped the probe because of that)
OUT=$HOME/wg-logs/probe; mkdir -p "$OUT"
mode=${1:-run}
if [ "$mode" = build ]; then
  rm -f "$BUILD/probe"
  docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$REPO":/work -v "$BUILD":/b -w /b "$IMG" \
    hipcc -O2 -std=c++17 -Wall -Wno-unused-function --offload-arch=gfx1201 -o probe /work/scripts/wg/probe_wmma.cpp 2>&1 | tail -n 20
  [ -x "$BUILD/probe" ] || { echo "probe build FAILED"; exit 2; }
  bash "$HOME/wg-win.sh" --stamp probe
  echo "probe built"
  exit 0
fi
want=$(bash "$HOME/wg-win.sh" --sha); have=$(cat "$HOME/wg-logs/stamp-probe" 2>/dev/null || echo none)
if [ "$have" != "$want" ] || [ ! -x "$BUILD/probe" ]; then echo "[probe] SKIPPED: stamp $have, tree $want, binary $([ -x "$BUILD/probe" ] && echo yes || echo no)"; exit 0; fi
echo "[probe] running $(date +%T)"
docker run --rm -i --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 \
  -v "$BUILD":/b -v "$OUT":/out -w /b "$IMG" timeout 120 ./probe /out 2000000 2>&1 | tee "$OUT/probe-summary.txt" | tail -n 400
echo "[probe] exit=${PIPESTATUS[0]} $(date +%T)"
exit 0
