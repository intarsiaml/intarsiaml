#!/usr/bin/env bash
# Lane b27-rerecord (re-record the four Qwen3.8-27B HIP accuracy baselines after the chunked GDN default, 2026-11-16): helpers
# ON tatooine (sourced). Tree ~/intarsiaml-b27-rerecord with its own cargo target volume. Every run: stdout -> $O/<tag>.out,
# stderr -> $O/<tag>.err; E="K=V ..." sets its environment inside the container.
TREE=${TREE:-$HOME/intarsiaml-b27-rerecord}
export INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-b27-rerecord}
export INTARSIA_MODELS=${INTARSIA_MODELS:-/mnt/models}   # the 27B checkpoint is not under the default models mount
O=${O:-$TREE/.b27-logs/$(date +%m%d-%H%M)}; mkdir -p "$O"
D="$TREE/deploy/tatooine/dev.sh"
BIN=/target/release/intarsia
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
