#!/usr/bin/env bash
# Lane rocm102-bisect (why ROCm 10.2 moves the 35B HIP logits, docs/results/2026-11-21-hip-rocm102-bisect.md): helpers ON tatooine
# (sourced). Tree ~/intarsiaml-rocm102-bisect, the two lane images of scripts/rocm102/lib.sh, target volumes of its own
# (rocm102b-*). Everything the lane makes lives under $B = $TREE/.bisect (the container sees it at /work/.bisect):
#   objs/<r7|r10>/<tu>.o    each compiler's kernel objects (build.rs's own, copied out of the build's OUT_DIR)
#   link-<arm>.txt          the CLI's final link command (rustc --print link-args, with -C save-temps so its inputs stay)
#   bins/<arm>/intarsia     the cargo-built CLI of each image;  mix/<name>/intarsia  relinked CLIs with mixed kernel archives
TREE=${TREE:-$HOME/intarsiaml-rocm102-bisect}
IMG_r7=intarsia-hip-dev:rocm7.2.4-rust1.90.0;  VOL_r7=intarsia-cargo-target-rocm102b-r7
IMG_r10=intarsia-hip-dev:rocm10.2.0-rust1.90.0; VOL_r10=intarsia-cargo-target-rocm102b-r10
B=$TREE/.bisect
O=${O:-$B/logs/$(date +%m%d-%H%M)}; mkdir -p "$O"
W=/work/${O#"$TREE"/}
D="$TREE/deploy/tatooine/dev.sh"
M35=/models/Qwen3.6-35B-A3B-NVFP4
STAMP="$(cat "$TREE/.stamp")"
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# dk <arm> [command...]: a container of that arm's image with its target volume (names start rocm102- so the lane kill finds them)
dk() {
  local a=$1; shift; local i="IMG_$a" v="VOL_$a"
  INTARSIA_REPO=$TREE INTARSIA_IMAGE=${!i} INTARSIA_TARGET_VOL=${!v} INTARSIA_CONTAINER_NAME=rocm102-bis-$a-$RANDOM "$D" "$@"
}
killall_lane() { docker ps -a --format '{{.Names}}' | grep -E '^rocm102-' | xargs -r docker rm -f > /dev/null 2>&1; }
