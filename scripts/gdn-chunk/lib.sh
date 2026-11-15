#!/usr/bin/env bash
# Lane gdn-chunk (docs/results/2026-11-14-hip-gdn-chunk.md): helpers for the window scripts (ON tatooine, sourced). The tree is
# ~/intarsiaml-gdn-chunk with its own cargo target volume, so no other lane's /target/release/intarsia is touched. Every run:
# stdout -> $O/<tag>.out, stderr -> $O/<tag>.err. Environment for a run: E="K=V K2=V2" (passed with `env` inside the container).
TREE=${TREE:-$HOME/intarsiaml-gdn-chunk}
export INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-gdn-chunk}
O=${O:-$TREE/.gdn-chunk-logs/$(date +%m%d-%H%M)}; mkdir -p "$O"
BIN=/target/release/intarsia
M35=/models/Qwen3.6-35B-A3B-NVFP4; M8=/models/Qwen3-8B; G=/models/golden
SG=$G/qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# run <gpus> <timeout> <tag> <intarsia args...>
run() {
  local gpus=$1 to=$2 tag=$3; shift 3
  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$TREE/deploy/tatooine/dev.sh" env ${E:-} $BIN "$@" > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}"
  return $rc
}
