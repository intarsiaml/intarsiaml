#!/usr/bin/env bash
# Lane wht6-8b (docs/results/2026-11-16-hip-wht8-ar.md): helpers for the window scripts (ON tatooine, sourced). The tree is
# ~/intarsiaml-wht6-8b with its own cargo target volume, so no other lane's /target/release/intarsia is touched. Every run:
# stdout -> $O/<tag>.out, stderr -> $O/<tag>.err. Environment for a run: E="K=V K2=V2" (passed with `env` inside the container).
TREE=${TREE:-$HOME/intarsiaml-wht6-8b}
export INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-wht6-8b}
O=${O:-$TREE/.wht6-8b-logs/$(date +%m%d-%H%M)}; mkdir -p "$O"
BIN=/target/release/intarsia
M35=/models/Qwen3.6-35B-A3B-NVFP4; G=/models/golden
STAMP="$(cat "$TREE/.stamp" 2>/dev/null)"
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# run <gpus> <timeout> <tag> <command in the container...>   (an intarsia subcommand goes as: run 0,1 300 tag $BIN generate ...)
run() {
  local gpus=$1 to=$2 tag=$3; shift 3
  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$TREE/deploy/tatooine/dev.sh" env ${E:-} "$@" > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}"
  return $rc
}
