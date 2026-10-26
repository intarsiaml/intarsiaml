#!/usr/bin/env bash
# Lane spt: helpers for the stages scripts (ON tatooine, sourced; after scripts/tp2-hip/lib.sh). Trees: ~/intarsiaml-spt-<name>,
# CLI at /work/spt-intarsia inside the HIP dev image (deploy/tatooine/dev.sh mounts the tree at /work and /data/models at /models).
# Every run: stdout -> $O/<tag>.out, stderr -> $O/<tag>.err. Environment for a run: E="K=V K2=V2".
O=${O:?out dir}
export INTARSIA_MODELS=/data/models
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
DRAFT=/models/tensorfold/Qwen3.8-27B-DFlash2
M35=/models/intarsia/Qwen3.6-35B-A3B-NVFP4; M8=/models/intarsia/Qwen3-8B
SHORT='Write a long story about a robot who learns to paint.'
TP2="--tp 2 --devices 0,1"
T() { echo "$HOME/intarsiaml-spt-${RUN:+run-$RUN-}$1"; }
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# DRY=1 (a cpu slot, radiance up, before a window): run/gen print the command and its environment instead of running it, mtest runs
# the test with INTARSIA_VERIFY_DRYRUN=1 (prints PROBE_CONFIG, loads nothing) and without a GPU; clear() passes.
# run <tree> <gpus> <timeout> <tag> <intarsia args...>
run() {
  local tree=$1 gpus=$2 to=$3 tag=$4; shift 4
  if [[ ${DRY:-0} == 1 ]]; then log "DRY run $tag gpus=$gpus E=[${E:-}] intarsia $*"; : > "$O/$tag.err"; return 0; fi
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T "$tree") HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$(T "$tree")/deploy/tatooine/dev.sh" env ${E:-} /work/spt-intarsia "$@" \
    > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}: $(grep -ahE '^Error|error:|decode .*tok/s|tok/s|spec|accept' "$O/$tag.err" | tail -4 | cut -c1-200 | tr '\n' '|')"
  return $rc
}
# gen <tree> <gpus> <tag> <model> <short|4k> [args]: 256 greedy ids -> $O/gen-<tag>.ids
gen() {
  local tree=$1 gpus=$2 tag=$3 m=$4 kind=$5; shift 5
  local p="$SHORT"; [[ $kind == 4k ]] && p="$(cat "$(T "$tree")/scripts/hip-decode/prompt4k.txt")"
  run "$tree" "$gpus" 300 "gen-$tag" generate -m "$m" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$p" --max-tokens 256 "$@"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
}
# mtest <tree> <tag> <gpus> <timeout> <cargo test args...>: intarsia-models GPU tests, prebuilt in the -hiptest volume
mtest() {
  local tree=$1 tag=$2 gpus=$3 to=$4; shift 4
  if [[ ${DRY:-0} == 1 ]]; then gpus=""; E="${E:-} INTARSIA_VERIFY_DRYRUN=1"; fi
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T "$tree") INTARSIA_TARGET_VOL=intarsia-cargo-target-spt-$tree-hiptest HIP_VISIBLE_DEVICES=$gpus timeout "$to" \
    "$(T "$tree")/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 ${E:-} \
    cargo test --release -p intarsia-models "$@" > "$O/mtest-$tag.log" 2>&1 < /dev/null
  local rc=$?
  log "mtest $tag exit $rc${E:+ [$E]}: $(grep -aE '^test result|FAILED|panicked|SKIPPED|PROBE_|DIFF' "$O/mtest-$tag.log" | head -14 | cut -c1-200 | tr '\n' '|')"
  return $rc
}
# htest <tree> <tag> <gpus> <timeout> <cargo test args...>: intarsia-backend-hip GPU tests, prebuilt in the -hiptest volume
htest() {
  local tree=$1 tag=$2 gpus=$3 to=$4; shift 4
  if [[ ${DRY:-0} == 1 ]]; then log "DRY htest $tag gpus=$gpus E=[${E:-}] $*"; return 0; fi
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T "$tree") INTARSIA_TARGET_VOL=intarsia-cargo-target-spt-$tree-hiptest HIP_VISIBLE_DEVICES=$gpus timeout "$to" \
    "$(T "$tree")/deploy/tatooine/dev.sh" env ${E:-} cargo test --release -p intarsia-backend-hip "$@" > "$O/htest-$tag.log" 2>&1 < /dev/null
  local rc=$?
  log "htest $tag exit $rc: $(grep -aE '^test result|FAILED|panicked|skipping|dyn_conv|attn_decode_bf16_window' "$O/htest-$tag.log" | head -12 | cut -c1-200 | tr '\n' '|')"
  return $rc
}
# plan <tree> <gpus> <tag> <model> [args]: plan show, JSON and text
plan() {
  local tree=$1 gpus=$2 tag=$3 m=$4; shift 4
  run "$tree" "$gpus" 300 "plan-$tag" plan show --backend hip --model "$m" --mem-budget 29 --max-seq 4096 --json "$@"
  run "$tree" "$gpus" 300 "plantext-$tag" plan show --backend hip --model "$m" --mem-budget 29 --max-seq 4096 "$@"
}
same() { if cmp -s "$O/$1" "$O/$2"; then log "SAME $3 ($1 == $2)"; else log "DIFF $3 ($1 != $2)"; fi; }
vram() { local m=0 v c; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done; echo $m; }
clear() { [[ ${DRY:-0} == 1 ]] && return 0; local i; for i in 1 2 3 4 5 6; do (( $(vram) < 1073741824 )) && return 0; sleep 10; done
  log "CONTENDED before '$1': $(( $(vram) / 1048576 )) MiB in use on a card: NO VERDICT"; return 1; }
