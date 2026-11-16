#!/usr/bin/env bash
# Lane hip-reuse (HIP hybrid prefix reuse test + 35B acc-4k baseline, 2026-11-15): helpers ON tatooine (sourced). Tree ~/intarsiaml-hip-reuse
# with its own cargo target volume. Every run: stdout -> $O/<tag>.out, stderr -> $O/<tag>.err; E="K=V ..." sets its environment.
TREE=${TREE:-$HOME/intarsiaml-hip-reuse}
export INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-hip-reuse}
O=${O:-$TREE/.hip-reuse-logs/$(date +%m%d-%H%M)}; mkdir -p "$O"
D="$TREE/deploy/tatooine/dev.sh"
BIN=/target/release/intarsia
M35=/models/Qwen3.6-35B-A3B-NVFP4
STAMP="$(cat "$TREE/.stamp")"
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# run <gpus> <timeout> <tag> <command in the container...>
run() {
  local gpus=$1 to=$2 tag=$3; shift 3
  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$D" env ${E:-} "$@" > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}"
  return $rc
}
