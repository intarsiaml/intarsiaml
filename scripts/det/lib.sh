#!/usr/bin/env bash
# Lane det: helpers for the stages scripts (ON tatooine, sourced; after scripts/vat/lib.sh). Tree ~/intarsiaml/.worktrees/det, CLI at
# /work/det-intarsia inside the HIP dev image. Every run: stdout -> $O/<tag>.out, stderr -> $O/<tag>.err; E="K=V ..." sets its environment.
O=${O:?out dir}
export INTARSIA_MODELS=/data/models
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
TREE=$HOME/intarsiaml/.worktrees/det
P=$TREE/scripts/dfh/ids-prompts
DRAFT=/models/tensorfold/Qwen3.8-27B-DFlash2
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# g <tag> <gpus> <prompt name> <repeat> [intarsia args]: greedy ${MT:-256} ids (prompt from scripts/dfh/ids-prompts or scripts/det/prompts), --repeat runs in one process. E="..." environment;
# TAP=1 also writes the layer hashes (INTARSIA_DET_TAP) to $O/<tag>.tap ($O is under the tree, which the container sees at /work).
g() {
  local tag=$1 gpus=$2 pr=$3 n=$4; shift 4
  local tap=""; [[ ${TAP:-0} == 1 ]] && tap="INTARSIA_DET_TAP=${O/#$TREE//work}/$tag.tap"
  # shellcheck disable=SC2086
  INTARSIA_REPO=$TREE HIP_VISIBLE_DEVICES=$gpus timeout 900 "$TREE/deploy/tatooine/dev.sh" env ${E:-} $tap /work/det-intarsia generate -m "$M27" \
    --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(cat "$P/$pr.txt" 2>/dev/null || cat "$TREE/scripts/det/prompts/$pr.txt")" --max-tokens "${MT:-256}" --repeat "$n" "$@" > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}: $(grep -aE '^Error|error:|decode .*tok/s|steps=' "$O/$tag.err" | tail -2 | cut -c1-160 | tr '\n' '|')"
  return $rc
}
vram() { local m=0 v c; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done; echo $m; }
clear() { local i; for i in 1 2 3 4 5 6; do (( $(vram) < 1073741824 )) && return 0; sleep 10; done
  log "CONTENDED before '$1': $(( $(vram) / 1048576 )) MiB in use on a card: NO VERDICT"; return 1; }
