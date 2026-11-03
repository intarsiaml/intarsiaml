#!/usr/bin/env bash
# Lane dsv41/rdma: HIP at tp=2 on tatooine, the Qwen3.8-27B MXFP4 with its recipe defaults (`[tuning.hip] mtp_default_tp_k = 2`: MTP on
# by itself at --tp 2): 256 greedy ids at a short and a 4k prompt, binary `main` against binary `head` (HIP CLIs built by
# scripts/dsv41-tp2/hipbuild-notes: ~/intarsiaml-rdma-<arm>/rdma-intarsia in the HIP dev image). In-process tensor parallelism: this
# path does not use crates/intarsia-core/src/tp_remote.rs (that is --tp-remote / tp-worker, CUDA only).
# Runs ON tatooine inside a fleetq slot that takes radiance down; stops radiance and starts it again (also on error):
#   fleetq run --host tatooine --takes-down radiance --mem 30 --expect 18 --label "..." -- bash -c 'ssh tatooine "export FLEETQ_JOB=$FLEETQ_JOB; bash -s" < scripts/dsv41-tp2/hip-tp2-compare.sh'
set -uo pipefail
O=${O:-$HOME/rdma-out/hip-$(date +%H%M)}; mkdir -p "$O"
ARMS=${ARMS:-"main head"}
export INTARSIA_MODELS=/data/models
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
SHORT='Write a long story about a robot who learns to paint.'
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
restore() { log "restarting radiance"; systemctl --user start radiance; sleep 20; log "radiance: $(systemctl --user is-active radiance), VRAM in use $(( $(vram) / 1048576 )) MiB"; }
trap restore EXIT
log "stopping radiance (VRAM in use $(( $(vram) / 1048576 )) MiB)"
systemctl --user stop radiance
for i in $(seq 1 30); do [ "$(vram)" -lt 1073741824 ] && break; sleep 2; done
log "radiance stopped: VRAM in use $(( $(vram) / 1048576 )) MiB, active=$(systemctl --user is-active radiance)"
P4K="$(cat "$HOME/intarsiaml-rdma-head/scripts/hip-decode/prompt4k.txt")"
for arm in $ARMS; do
  for kind in short 4k; do
    p="$SHORT"; [ $kind = 4k ] && p="$P4K"
    INTARSIA_REPO=$HOME/intarsiaml-rdma-$arm HIP_VISIBLE_DEVICES=0,1 timeout 420 "$HOME/intarsiaml-rdma-$arm/deploy/tatooine/dev.sh" \
      env INTARSIA_SPEC_TRACE=1 /work/rdma-intarsia generate -m "$M27" --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 \
      --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$p" --max-tokens 256 > "$O/gen-$arm-$kind.out" 2> "$O/gen-$arm-$kind.err" < /dev/null
    rc=$?
    grep -a '^ids:' "$O/gen-$arm-$kind.err" > "$O/gen-$arm-$kind.ids"
    log "$arm $kind exit $rc: $(wc -w < "$O/gen-$arm-$kind.ids") words in ids line; spec: $(grep -ac 'spec round' "$O/gen-$arm-$kind.err") round lines; $(grep -aiE 'speculative|mtp' "$O/gen-$arm-$kind.err" | head -2 | cut -c1-160 | tr '\n' '|')"
  done
done
for kind in short 4k; do
  if [ -s "$O/gen-main-$kind.ids" ] && cmp -s "$O/gen-main-$kind.ids" "$O/gen-head-$kind.ids"; then log "COMPARE $kind: IDENTICAL head vs main ($(grep -ao 'prompt=[0-9]*' "$O/gen-head-$kind.err" | tail -1))"
  else log "COMPARE $kind: DIFFER or empty"; fi
done
