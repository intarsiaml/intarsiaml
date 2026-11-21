#!/usr/bin/env bash
# Lane rocm102 (ROCm 10.2.0 trial, docs/results/2026-11-21-hip-rocm102.md): helpers ON tatooine (sourced). One tree
# ~/intarsiaml-rocm102, two arms that differ only in the image: ARM=r7 (7.2.4, the default image) and ARM=r10 (10.2.0,
# deploy/tatooine/Dockerfile.hip-dev-rocm10), each with its own cargo target volume (never share one across images).
# Every run: stdout -> $O/<tag>.out, stderr -> $O/<tag>.err; E="K=V ..." sets its environment.
TREE=${TREE:-$HOME/intarsiaml-rocm102}
IMG_r7=intarsia-hip-dev:rocm7.2.4-rust1.90.0;  VOL_r7=intarsia-cargo-target-rocm102-r7
IMG_r10=intarsia-hip-dev:rocm10.2.0-rust1.90.0; VOL_r10=intarsia-cargo-target-rocm102-r10
arm() { # arm r7|r10: point dev.sh at that arm's image and target volume
  local i="IMG_$1" v="VOL_$1"
  export INTARSIA_REPO=$TREE INTARSIA_IMAGE=${!i} INTARSIA_TARGET_VOL=${!v} INTARSIA_CONTAINER_NAME=rocm102-$1-$RANDOM
}
O=${O:-$TREE/.rocm102-logs/$(date +%m%d-%H%M)}; mkdir -p "$O"
D="$TREE/deploy/tatooine/dev.sh"
BIN=/target/release/intarsia
M35=/models/Qwen3.6-35B-A3B-NVFP4
STAMP="$(cat "$TREE/.stamp")"
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# run <arm> <gpus> <timeout> <tag> <command in the container...>
run() {
  local a=$1 gpus=$2 to=$3 tag=$4; shift 4
  arm "$a"
  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$D" env ${E:-} "$@" > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}"
  docker rm -f "$INTARSIA_CONTAINER_NAME" > /dev/null 2>&1   # a timeout kills only the docker client
  return $rc
}
# kill every container of this lane (both images); window.sh / guard.sh call it before radiance starts again
killall_lane() { docker ps -a --format '{{.Names}}' | grep -E '^rocm102-' | xargs -r docker rm -f > /dev/null 2>&1; }
